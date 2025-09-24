import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from config import logger

# First-pass Spectral clustering
def first_pass_clustering(embeddings, min_seg_duration=2.5, sv_threshold=0.15,
                          max_speakers=10, under_count_factor=1.2):
    """
    Perform first-pass NMESC-inspired clustering on embeddings.
    """
    if not embeddings:
        logger.warning("No embeddings provided for first-pass clustering.")
        return []

    # 1. Stack embeddings into matrix
    segments, X = zip(*embeddings)
    X = np.stack(X)
    X = normalize(X)

    # 2. Estimate number of speakers
    n_segments = len(X)
    #est_speakers = max(1, int(np.sqrt(n_segments)))  # heuristic
    est_speakers = max(2, min(n_segments // 5, max_speakers))
    n_speakers = min(max_speakers, int(est_speakers * under_count_factor))
    logger.info(f"First-pass clustering: estimated={est_speakers}, "
                f"over-counted n_speakers={n_speakers}")

    # 3. Similarity matrix
    dist_matrix = pairwise_distances(X, metric="euclidean")
    sim_matrix = 1 - dist_matrix
    logger.info(f"Similarity matrix stats: min={sim_matrix.min()}, max={sim_matrix.max()}, mean={sim_matrix.mean()}")

    # 4. Spectral Clustering
    clustering = SpectralClustering(
        n_clusters=n_speakers,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=42
    )
    labels = clustering.fit_predict(sim_matrix)

    # 5. Mark short-duration segments as unassigned (-1)
    filtered_labels = []
    for seg, lbl in zip(segments, labels):
        if seg.duration < min_seg_duration:
            filtered_labels.append(-1)
        else:
            filtered_labels.append(lbl)

    logger.info(f"First-pass clustering done. "
                f"Clusters: {len(np.unique(labels))}, "
                f"Unassigned: {filtered_labels.count(-1)}")

    return filtered_labels


# Second-pass reclustering
def recluster_embeddings(embeddings, first_pass_labels, distance_threshold=0.2):
    """
    Second-pass clustering to refine speaker assignments.
    """
    if not embeddings:
        logger.warning("No embeddings for re-clustering.")
        return []

    X = np.stack([emb[1] for emb in embeddings])
    labels = np.array(first_pass_labels)
    refined_labels = labels.copy()

    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    # 1. Speaker-level embeddings (mean per cluster)
    speaker_embeddings = []
    speaker_map = {}
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        cluster_embs = X[idxs]
        speaker_emb = cluster_embs.mean(axis=0)
        speaker_embeddings.append(speaker_emb)
        speaker_map[lbl] = idxs

    if len(speaker_embeddings) > 1:
        speaker_embeddings = np.stack(speaker_embeddings)

        # 2. Merge similar speakers
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="average",
            metric="euclidean"
        )
        new_labels = agg.fit_predict(speaker_embeddings)

        # 3. Map old labels to refined ones
        label_mapping = {old: new for old, new in zip(unique_labels, new_labels)}
        for i, lbl in enumerate(refined_labels):
            if lbl != -1:
                refined_labels[i] = label_mapping[lbl]

    logger.info(f"Re-clustering done. Final clusters: {len(np.unique(refined_labels))}")
    return refined_labels
