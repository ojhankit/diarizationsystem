# clustering.py

import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from config import logger

# First-pass Spectral

def auto_spectral_clustering(embeddings,max_speaker=10):
    """

    """
    if not embeddings:
        logger.warning("No embeddings")
        return [],0

    X = np.stack([emb[1] for emb in embeddings])

    # compute pairwise dist
    dist_matrix = pairwise_distances(X, metric="cosine")
    # estimate speakers using heurestics- eigen-gap
    n_segments = len(X)
    n_speakers = min(max(1,int(np.sqrt(n_segments))),max_speaker)
    logger.info(f"Estimated number of speakers: {n_speakers}")

    clustering = SpectralClustering(
        n_clusters = n_speakers,
        affinity = "precomputed",
        assign_labels = "discretize",
        random_state = 42
    )
    labels = clustering.fit_predict(1-dist_matrix)
    return labels,n_speakers

# second cluster
def recluster_embeddings(embeddings, labels, distance_threshold=0.3):
    """

    """
    if not embeddings:
        return []
    
    X = np.stack([emb[1] for emb in embeddings])
    refined_labels = labels.copy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        idx = np.where(labels==label)[0]
        if len(idx) <= 1:
            continue
        
        cluster_embs = X[idx]

        # Agglomerative re-clustering inside cluster
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )
        sub_labels = agg.fit_predict(cluster_embs)

        # Reassign refined labels (ensure global uniqueness)
        max_label = refined_labels.max() + 1
        for i, ix in enumerate(idx):
            if sub_labels[i] != 0:
                refined_labels[ix] = max_label + sub_labels[i] - 1

    logger.info(f"Re-clustering done. Total clusters after refinement: {len(np.unique(refined_labels))}")
    return refined_labels