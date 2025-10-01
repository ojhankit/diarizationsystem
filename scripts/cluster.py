import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from config import logger
from sklearn.preprocessing import normalize

def estimate_speakers_eigengap(sim_matrix, max_speakers, method):
    """
    Estimate number of speakers using multiple heuristics.
    
    Args:
        sim_matrix: precomputed similarity matrix
        max_speakers: maximum allowed speakers
        method: 'eigengap', 'sqrt', 'combined' (recommended)
    """
    n = sim_matrix.shape[0]
    if n < 2:
        logger.warning("Too few segments for eigengap. Defaulting to 1 speaker.")
        return 1

    try:
        # Method 1: Eigengap on Laplacian (more reliable)
        # Convert similarity to affinity matrix
        affinity = sim_matrix.copy()
        np.fill_diagonal(affinity, 0)  # Remove self-loops for Laplacian
        
        # Compute normalized Laplacian
        degree = np.sum(affinity, axis=1)
        degree[degree == 0] = 1  # Avoid division by zero
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
        
        # Get eigenvalues
        eigvals = np.linalg.eigvalsh(L_norm)
        eigvals = np.sort(eigvals)  # ascending for Laplacian
        
        # Find eigengap (looking for small eigenvalues)
        gaps = np.diff(eigvals[:max_speakers])
        if len(gaps) > 0:
            k_eigengap = np.argmax(gaps) + 1
        else:
            k_eigengap = 2
            
        # Method 2: Enhanced sqrt heuristic with scaling
        k_sqrt = max(2, int(np.sqrt(n) * 2.5))  # Scale up sqrt estimate
        
        # Method 3: Connectivity-based estimate
        # Count number of distinct "communities" based on strong connections
        threshold = np.percentile(affinity[affinity > 0], 75) if np.any(affinity > 0) else 0.5
        strong_connections = (affinity > threshold).astype(int)
        np.fill_diagonal(strong_connections, 0)
        # Estimate from average connectivity
        avg_connections = np.mean(np.sum(strong_connections, axis=1))
        if avg_connections > 0:
            k_connectivity = max(2, int(n / max(2, avg_connections)))
        else:
            k_connectivity = k_sqrt
        
        if method == 'eigengap':
            k = k_eigengap
        elif method == 'sqrt':
            k = k_sqrt
        else:  # combined
            # Take the maximum of the methods (to avoid under-estimation)
            # But weight eigengap higher if it suggests more clusters
            k = max(k_eigengap, int((k_sqrt + k_connectivity) / 2))
            logger.info(f"Speaker estimates - eigengap: {k_eigengap}, sqrt: {k_sqrt}, connectivity: {k_connectivity}")
        
        k = max(2, min(k, max_speakers))
        logger.info(f"Final speaker estimate: {k}")
        return k

    except Exception as e:
        logger.error(f"Eigengap estimation failed: {e}. Falling back to enhanced sqrt heuristic.")
        return max(2, min(int(np.sqrt(n) * 1.5), max_speakers))


def first_pass_clustering(
    embeddings,
    min_seg_duration=2.5,
    max_speakers=25,
    affinity_threshold=0.08,  # Keep low threshold
    estimation_method='sqrt',
):
    """
    First-pass clustering with improved speaker estimation.
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

    # --- Similarity matrix ---
    dist_matrix = cosine_distances(X)
    sim_matrix = 1 - dist_matrix
    sim_matrix = np.nan_to_num(sim_matrix)
    sim_matrix = (sim_matrix + 1.0) / 2.0
    np.fill_diagonal(sim_matrix, 1.0)
    
    # --- Affinity thresholding ---
    affinity_matrix = sim_matrix.copy()
    affinity_matrix[affinity_matrix < affinity_threshold] = 0.0

    # --- Estimate number of speakers ---
    est_speakers = estimate_speakers_eigengap(affinity_matrix, max_speakers, method=estimation_method)
    logger.info(f"Estimated speakers: {est_speakers} (from {n_segments} segments)")

    # --- Spectral clustering ---
    clustering = SpectralClustering(
        n_clusters=est_speakers,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=10  # Multiple initializations for stability
    )
    labels = clustering.fit_predict(affinity_matrix)

    # Filter short segments
    filtered_labels = []
    for seg, lbl in zip(segments, labels):
        if seg.duration < min_seg_duration:
            filtered_labels.append(-1)
        else:
            filtered_labels.append(lbl)

    logger.info(
        f"First-pass clustering done. "
        f"Clusters: {len(np.unique([l for l in labels if l != -1]))}, "
        f"Unassigned: {filtered_labels.count(-1)}"
    )

    return filtered_labels


def recluster_embeddings(
    embeddings,
    first_pass_labels,
    distance_threshold=0.35,  # Slightly increased for less aggressive merging
    similarity_merge_threshold=0.85,  # Lower threshold = easier to keep separate
    min_cluster_size=1,  # Allow smaller clusters (was 2)
    max_iterations=2,  # Reduce iterations to preserve clusters
):
    """
    Second-pass HAC clustering with less aggressive merging.
    
    Key changes:
    - Increased distance_threshold to merge less
    - Lowered similarity_merge_threshold to preserve distinctions
    - Reduced min_cluster_size to keep small but valid clusters
    - Fewer iterations to avoid over-merging
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

        # --- HAC merge (less aggressive) ---
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

        # --- Merge ONLY if similarity is very high ---
        sims = cosine_similarity(speaker_embeddings)
        merged = False
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                if sims[i, j] >= similarity_merge_threshold:
                    # Only merge if both clusters are small OR similarity is extremely high
                    if (cluster_sizes.get(unique_labels[i], 0) < 5 and 
                        cluster_sizes.get(unique_labels[j], 0) < 5) or sims[i, j] >= 0.95:
                        refined_labels[refined_labels == unique_labels[j]] = unique_labels[i]
                        merged = True
                        logger.info(f"Merged clusters {unique_labels[i]} and {unique_labels[j]} (sim: {sims[i, j]:.3f})")
        
        # --- Handle tiny clusters more carefully ---
        for lbl in np.unique(refined_labels):
            if lbl == -1:
                continue
            idxs = np.where(refined_labels == lbl)[0]
            if len(idxs) < min_cluster_size:
                emb = X[idxs].mean(axis=0).reshape(1, -1)
                sims = cosine_similarity(emb, speaker_embeddings)
                max_sim = np.max(sims)
                # Only absorb if very similar to another cluster
                if max_sim >= 0.80:
                    nearest = np.argmax(sims)
                    refined_labels[idxs] = unique_labels[nearest]
                    logger.info(f"Small cluster {lbl} absorbed into {unique_labels[nearest]} (sim: {max_sim:.3f})")

        # --- Re-assign unassigned segments ---
        unassigned_count = 0
        for i, lbl in enumerate(refined_labels):
            if lbl == -1:
                emb = X[i].reshape(1, -1)
                sims = cosine_similarity(emb, speaker_embeddings)
                refined_labels[i] = unique_labels[np.argmax(sims)]
                unassigned_count += 1
        
        if unassigned_count > 0:
            logger.info(f"Re-assigned {unassigned_count} unassigned segments")

    final_count = len(np.unique(refined_labels[refined_labels != -1]))
    logger.info(f"Re-clustering done. Final clusters: {final_count}")
    return refined_labels