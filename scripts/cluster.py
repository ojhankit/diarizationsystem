import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from config import logger
from sklearn.preprocessing import normalize

def estimate_speakers_eigengap(sim_matrix, max_speakers):
    """
    Estimate number of speakers using eigengap heuristic on similarity matrix.
    Fallback to sqrt(n_segments) if eigengap fails.
    """

    n = sim_matrix.shape[0]
    if n < 2:
        logger.warning("Too few segments for eigengap. Defaulting to 1 speaker.")
        return 1

    try:
        # Eigen decomposition
        eigvals = np.linalg.eigvalsh(sim_matrix)
        eigvals = np.sort(eigvals)[::-1]  # sort descending

        # Compute eigengaps
        gaps = np.diff(eigvals)
        if len(gaps) == 0:
            logger.warning("No eigengap available. Falling back to sqrt heuristic.")
            return max(2, int(np.sqrt(n)))

        # Choose largest eigengap (but limit max_speakers)
        k = np.argmax(gaps[:max_speakers - 1]) + 1
        return max(2, min(k, max_speakers))

    except Exception as e:
        logger.error(f"Eigengap estimation failed: {e}. Falling back to sqrt heuristic.")
        return max(2, int(np.sqrt(n)))


def first_pass_clustering(
    embeddings,
    min_seg_duration=2.5,
    max_speakers=20,
    #affinity_threshold=0.3,
    affinity_threshold=0.05,
    under_count_factor=2.5,
):
    """
    First-pass clustering with eigengap heuristic for speaker estimation.
    """

    if not embeddings:
        logger.warning("No embeddings provided for first-pass clustering.")
        return []

    clean_embeddings, clean_segments = [], []
    for seg, emb in embeddings:
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        emb = np.nan_to_num(emb)
        if np.linalg.norm(emb) == 0:
            continue
        clean_embeddings.append(emb)
        clean_segments.append(seg)

    if not clean_embeddings:
        logger.warning("All embeddings invalid after cleaning.")
        return []

    X = np.stack(clean_embeddings)
    X = normalize(X)
    segments = clean_segments

    if len(X) < 2:
        logger.warning("Not enough embeddings for clustering. Assigning all segments to single cluster.")
        return [0] * len(X)

    n_segments = len(X)
    # est_speakers = max(2, int(np.sqrt(n_segments)))
    # n_speakers = min(max_speakers, int(est_speakers * under_count_factor))
    # logger.info(f"First-pass clustering: estimated={est_speakers}, over-counted={n_speakers}")

    # --- Similarity matrix ---
    dist_matrix = cosine_distances(X)
    sim_matrix = 1 - dist_matrix
    sim_matrix = np.nan_to_num(sim_matrix)
    sim_matrix = (sim_matrix + 1.0) / 2.0
    np.fill_diagonal(sim_matrix, 1.0)
    
    # --- Affinity thresholding ---
    sim_matrix[sim_matrix < affinity_threshold] = 0.0

    # --- Estimate number of speakers using eigengap ---
    est_speakers = estimate_speakers_eigengap(sim_matrix, max_speakers)
    logger.info(f"Eigengap estimated speakers: {est_speakers}")

    # --- Spectral clustering ---
    clustering = SpectralClustering(
        n_clusters=est_speakers,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42
    )
    labels = clustering.fit_predict(sim_matrix)

    # Filter short segments
    filtered_labels = []
    for seg, lbl in zip(segments, labels):
        if seg.duration < min_seg_duration:
            filtered_labels.append(-1)
        else:
            filtered_labels.append(lbl)

    logger.info(
        f"First-pass clustering done. "
        f"Clusters: {len(np.unique(labels))}, "
        f"Unassigned: {filtered_labels.count(-1)}"
    )

    return filtered_labels


def recluster_embeddings(
    embeddings,
    first_pass_labels,
    #distance_threshold=0.5,
    #similarity_merge_threshold=0.7,
    distance_threshold=0.3,
    similarity_merge_threshold=0.9,
    min_cluster_size=2,
    max_iterations=3,
):
    """
    Second-pass HAC clustering with iterative refinement.

    Args:
        embeddings: list of (segment, embedding) pairs
        first_pass_labels: labels from first-pass clustering
        distance_threshold: float, HAC merge threshold
        similarity_merge_threshold: float, min cosine similarity to merge clusters
        min_cluster_size: int, small clusters below this will be absorbed
        max_iterations: int, how many refinement passes

    Returns:
        refined_labels: list of final cluster labels per segment
    """

    if not embeddings:
        logger.warning("No embeddings for re-clustering.")
        return []

    # --- Clean embeddings ---
    clean_embeddings = []
    for seg, emb in embeddings:
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        emb = np.nan_to_num(emb)
        if np.linalg.norm(emb) == 0:
            emb = np.zeros_like(emb)
        clean_embeddings.append(emb)

    X = np.stack(clean_embeddings)
    labels = np.array(first_pass_labels, dtype=int)
    refined_labels = labels.copy()

    for iteration in range(max_iterations):
        logger.info(f"Refinement iteration {iteration+1}/{max_iterations}")

        unique_labels = [lbl for lbl in np.unique(refined_labels) if lbl != -1]
        if len(unique_labels) <= 1:
            break

        # --- Compute cluster embeddings ---
        speaker_embeddings = []
        cluster_sizes = {}
        for lbl in unique_labels:
            idxs = np.where(refined_labels == lbl)[0]
            cluster_embs = X[idxs]
            speaker_emb = cluster_embs.mean(axis=0)
            speaker_embeddings.append(speaker_emb)
            cluster_sizes[lbl] = len(idxs)

        speaker_embeddings = np.stack(speaker_embeddings)

        # --- HAC merge ---
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
            metric="cosine"
        )
        new_labels = agg.fit_predict(speaker_embeddings)

        # --- Map old → new labels ---
        label_mapping = {old: new for old, new in zip(unique_labels, new_labels)}
        for i, lbl in enumerate(refined_labels):
            if lbl != -1:
                refined_labels[i] = label_mapping[lbl]

        # --- Merge only if similarity is high enough ---
        sims = cosine_similarity(speaker_embeddings)
        merged = False
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                if sims[i, j] >= similarity_merge_threshold:
                    refined_labels[refined_labels == j] = i
                    merged = True
        if merged:
            logger.info("Some clusters merged based on similarity.")

        # --- Absorb tiny clusters ---
        for lbl in np.unique(refined_labels):
            idxs = np.where(refined_labels == lbl)[0]
            if len(idxs) < min_cluster_size:
                emb = X[idxs].mean(axis=0).reshape(1, -1)
                sims = cosine_similarity(emb, speaker_embeddings)
                nearest = np.argmax(sims)
                refined_labels[idxs] = nearest
                logger.info(f"Cluster {lbl} absorbed into {nearest} due to small size.")

        # --- Re-assign unassigned segments ---
        for i, lbl in enumerate(refined_labels):
            if lbl == -1:
                emb = X[i].reshape(1, -1)
                sims = cosine_similarity(emb, speaker_embeddings)
                refined_labels[i] = np.argmax(sims)

    logger.info(f"Re-clustering done. Final clusters: {len(np.unique(refined_labels))}")
    return refined_labels