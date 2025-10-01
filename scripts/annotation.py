from pyannote.core import Annotation

def build_annotation(embeddings, labels):
    """
    Build pyannote.core.Annotation object from embeddings and cluster labels.

    Args:
        embeddings: list of (segment, embedding_vector)
        labels: list of cluster labels (ints, -1 means unassigned)

    Returns:
        Annotation object
    """
    annotation = Annotation()

    for (segment, _), label in zip(embeddings, labels):
        if label == -1:  # skip unassigned
            continue
        # Add speaker label (e.g., SPEAKER_0, SPEAKER_1, ...)
        annotation[segment] = f"spk{label}"

    return annotation
