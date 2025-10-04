import os
import glob
from tqdm import tqdm
from config import logger
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from embedder import extract_embeddings
#from ecapa_embedder import extract_embeddings
from cluster import first_pass_clustering, recluster_embeddings
from annotation import build_annotation

if __name__ == "__main__":
    # Base project dir (audio/ and vad_outputs/ are siblings of current dir)
    base_dir = os.path.dirname(os.getcwd())

    # Input / Output dirs
    audio_dir = os.path.join(base_dir, "audio")
    vad_dir = os.path.join(os.getcwd(), "vad_outputs")
    out_dir = os.path.join(os.getcwd(), "after_clustering_v2")
    os.makedirs(out_dir, exist_ok=True)

    # Find all wav files
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        logger.error(f"No .wav files found in {audio_dir}")
        exit(1)

    logger.info(f"Found {len(wav_files)} wav files in {audio_dir}")

    # Process each audio file
    for wav_path in tqdm(wav_files[:], desc="Embedding & Clustering"):
        audio_name = os.path.basename(wav_path).replace(".wav", "")
        rttm_path = os.path.join(vad_dir, f"{audio_name}.rttm")

        if not os.path.exists(rttm_path):
            logger.warning(f"No RTTM found for {audio_name}, skipping.")
            continue

        logger.info(f"\n=== Processing {audio_name} ===")
        logger.info(f"Audio file: {wav_path}")
        logger.info(f"VAD RTTM: {rttm_path}")

        # Step 1: Load VAD RTTM → Annotation
        try:
            # Load RTTM file using pyannote's load_rttm function
            rttm_data = load_rttm(rttm_path)
            # Get the annotation for this specific audio file
            vad_result = rttm_data[audio_name]
            logger.info(f"Loaded {len(list(vad_result.itersegments()))} speech segments from RTTM.")
        except Exception as e:
            logger.error(f"Failed to load RTTM for {audio_name}: {e}")
            continue

        # Step 2: Extract embeddings
        embeddings = extract_embeddings(vad_result, wav_path)
        if not embeddings:
            logger.warning(f"No embeddings extracted for {audio_name}, skipping.")
            continue

        logger.info(f"Extracted {len(embeddings)} embeddings. "
                    f"First embedding shape: {embeddings[0][1].shape}")

        # Step 3: First-pass clustering
        first_pass_labels = first_pass_clustering(embeddings)
        logger.info(f"First-pass produced {len(set(first_pass_labels))} unique labels.")

        # Step 4: Re-cluster refinement
        refined_labels = recluster_embeddings(embeddings, first_pass_labels)
        logger.info(f"Refined clustering produced {len(set(refined_labels))} unique labels.")

        # Step 5: Build final annotation
        final_annotation = build_annotation(embeddings, refined_labels)
        logger.info("Final annotation built successfully.")

        # Step 6: Save RTTM output
        out_path = os.path.join(out_dir, f"{audio_name}.rttm")
        with open(out_path, "w") as f:
            final_annotation.write_rttm(f)

        logger.info(f"✅ Saved diarization RTTM: {out_path}")