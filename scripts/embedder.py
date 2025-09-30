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
#inference = Inference(embedding_model, window="whole")
inference = Inference(embedding_model, window="sliding",duration=2.0,step=1.0)
logger.info("Embedding model loaded.")

# --------------------------
# Extract embeddings
# --------------------------
def split_segment_safely(segment, max_duration, min_duration):
    """
    Split a segment into chunks, ensuring all chunks meet minimum duration.
    
    Args:
        segment: pyannote Segment object
        max_duration: maximum chunk size
        min_duration: minimum acceptable chunk size
        
    Returns:
        list of Segment objects
    """
    if segment.duration <= max_duration:
        return [segment]
    
    # Calculate how many full chunks we can create
    num_full_chunks = int(segment.duration // max_duration)
    remaining_time = segment.duration - (num_full_chunks * max_duration)
    
    chunks = []
    start = segment.start
    
    # Create full-sized chunks
    for i in range(num_full_chunks):
        end = start + max_duration
        chunks.append(Segment(start, end))
        start = end
    
    # Handle remaining time
    if remaining_time > 0:
        if remaining_time >= min_duration:
            # Remaining time is long enough to be its own chunk
            chunks.append(Segment(start, segment.end))
        else:
            # Remaining time too short - extend the last chunk or merge
            if chunks:
                # Extend the last chunk to include remaining time
                last_chunk = chunks[-1]
                chunks[-1] = Segment(last_chunk.start, segment.end)
                logger.debug(f"Extended last chunk to include remaining {remaining_time:.3f}s")
            else:
                # This shouldn't happen if segment > max_duration, but just in case
                logger.warning(f"Unexpected case: segment {segment} couldn't be split properly")
    
    return chunks


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
    #MIN_SEG_DURATION = 1.0      # Minimum duration to avoid kernel size issues
    #MAX_SEG_DURATION = 3.0      # Maximum chunk size
    MIN_SEG_DURATION = 2.0
    MAX_SEG_DURATION = 8.0
    logger.info(f"Processing {len(list(vad_result.itersegments()))} segments from VAD")

    for idx, (segment, _, _) in enumerate(vad_result.itertracks(yield_label=True), 1):
        # Skip too short segments entirely
        if segment.duration < MIN_SEG_DURATION:
            logger.warning(f"Skipping short segment {segment} (duration: {segment.duration:.3f}s < {MIN_SEG_DURATION}s)")
            continue

        # Split segment safely
        sub_segments = split_segment_safely(segment, MAX_SEG_DURATION, MIN_SEG_DURATION)
        logger.debug(f"Segment {segment} split into {len(sub_segments)} sub-segments")

        # Process each sub-segment
        for sub_segment in sub_segments:
            # Double-check duration (should not be necessary with safe splitting, but just in case)
            if sub_segment.duration < MIN_SEG_DURATION:
                logger.warning(f"Sub-segment {sub_segment} still too short (duration: {sub_segment.duration:.3f}s), skipping")
                continue

            try:
                with torch.inference_mode():
                    emb = inference.crop(audio_file, sub_segment)
                
                # Convert to numpy array and average over frames
                emb = emb.data
                if emb.ndim > 1:
                    emb = np.mean(emb, axis=0)

                # Handle NaN / Inf
                if np.isnan(emb).any() or np.isinf(emb).any():
                    logger.warning(f"Segment {sub_segment} has NaN/Inf in embedding, replacing with 0")
                    emb = np.nan_to_num(emb)

                # Skip zero vectors
                if np.linalg.norm(emb) == 0:
                    logger.warning(f"Segment {sub_segment} has zero embedding, skipping.")
                    continue

                embeddings.append((sub_segment, emb))
                logger.debug(f"Successfully extracted embedding for {sub_segment} (duration: {sub_segment.duration:.3f}s)")

            except Exception as e:
                logger.error(f"[{audio_file}] Failed on sub-segment ({sub_segment}) with duration {sub_segment.duration:.3f}s: {e}")
                logger.error(f"This should not happen with safe splitting - please check your VAD output")
                continue

    logger.info(f"[{audio_file}] Extracted {len(embeddings)} embeddings from {idx} original segments")
    return embeddings