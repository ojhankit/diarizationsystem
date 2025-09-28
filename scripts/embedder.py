# import torch
# from pyannote.audio import Model
# from config import HF_ACCESS_TOKEN, logger

# # --------------------------
# # Load embedding model
# # --------------------------
# logger.info("Loading embedding model 'pyannote/embedding'...")
# embedding_model = Model.from_pretrained(
#     "pyannote/embedding",
#     use_auth_token=HF_ACCESS_TOKEN
# )
# logger.info("Embedding model loaded.")


# # --------------------------
# # Extract embeddings
# # --------------------------
# def extract_embeddings(vad_result, audio_file):
#     """
#     Extract speaker embeddings for all VAD segments in an audio file.

#     Args:
#         vad_result (Annotation): Pyannote annotation with speech segments
#         audio_file (str): Path to the audio file

#     Returns:
#         list of (Segment, embedding_vector)
#     """
#     embeddings = []

#     for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), 1):
#         try:
#             # Crop waveform for the given segment
#             waveform, sample_rate = embedding_model.audio.crop(audio_file, segment)

#             # Generate embedding
#             with torch.inference_mode():
#                 emb = embedding_model(waveform[None])

#             embeddings.append((segment, emb.squeeze().numpy()))

#         except Exception as e:
#             logger.error(f"[{audio_file}] Failed on segment {idx} ({segment}): {e}")

#     logger.info(f"[{audio_file}] Extracted {len(embeddings)} embeddings")
#     return embeddings

## Embedder.py
import torch
import numpy as np
from pyannote.audio import Model, Inference
from pyannote.core import Annotation, Segment
from config import HF_ACCESS_TOKEN, logger

# --------------------------
# Load embedding model
# --------------------------
logger.info("Loading embedding model 'pyannote/embedding'...")
embedding_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=HF_ACCESS_TOKEN
)
inference = Inference(embedding_model, window="whole")
logger.info("Embedding model loaded.")

# --------------------------
# Extract embeddings
# --------------------------
def extract_embeddings(vad_result: Annotation, audio_file: str):
    """
    Extract speaker embeddings for all VAD segments in an audio file.
    Splits long segments into smaller chunks.
    Filters out very short segments and ensures no NaN/Inf embeddings.

    Args:
        vad_result (Annotation)
        audio_file (str)

    Returns:
        list of (Segment, embedding_vector)
    """
    embeddings = []
    MIN_SEG_DURATION = 0.5      # skip tiny segments
    MAX_SEG_DURATION = 3.0      # split long segments into max 3s chunks

    for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), 1):
        # Skip too short segments
        if segment.duration < MIN_SEG_DURATION:
            logger.warning(f"Skipping short segment {segment}")
            continue

        # Split long segments into smaller chunks
        start = segment.start
        while start < segment.end:
            end = min(start + MAX_SEG_DURATION, segment.end)
            sub_segment = Segment(start, end)

            try:
                with torch.inference_mode():
                    emb = inference.crop(audio_file, sub_segment)
                emb = emb.squeeze()

                # Handle NaN / Inf
                if np.isnan(emb).any() or np.isinf(emb).any():
                    logger.warning(f"Segment {sub_segment} has NaN/Inf in embedding, replacing with 0")
                    emb = np.nan_to_num(emb)

                # Skip zero vectors
                if np.linalg.norm(emb) == 0:
                    logger.warning(f"Segment {sub_segment} has zero embedding, skipping.")
                    start += MAX_SEG_DURATION
                    continue

                embeddings.append((sub_segment, emb))

            except Exception as e:
                logger.error(f"[{audio_file}] Failed on sub-segment ({sub_segment}): {e}")

            start += MAX_SEG_DURATION

    logger.info(f"[{audio_file}] Extracted {len(embeddings)} embeddings")
    return embeddings

