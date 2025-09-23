import torch
from pyannote.audio import Model
from config import HF_ACCESS_TOKEN, logger

# --------------------------
# Load embedding model
# --------------------------
logger.info("Loading embedding model 'pyannote/embedding'...")
embedding_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=HF_ACCESS_TOKEN
)
logger.info("Embedding model loaded.")


# --------------------------
# Extract embeddings
# --------------------------
def extract_embeddings(vad_result, audio_file):
    """
    Extract speaker embeddings for all VAD segments in an audio file.

    Args:
        vad_result (Annotation): Pyannote annotation with speech segments
        audio_file (str): Path to the audio file

    Returns:
        list of (Segment, embedding_vector)
    """
    embeddings = []

    for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), 1):
        try:
            # Crop waveform for the given segment
            waveform, sample_rate = embedding_model.audio.crop(audio_file, segment)

            # Generate embedding
            with torch.inference_mode():
                emb = embedding_model(waveform[None])

            embeddings.append((segment, emb.squeeze().numpy()))

        except Exception as e:
            logger.error(f"[{audio_file}] Failed on segment {idx} ({segment}): {e}")

    logger.info(f"[{audio_file}] Extracted {len(embeddings)} embeddings")
    return embeddings
