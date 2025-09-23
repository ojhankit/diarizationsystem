from vad import voice_activity_detection
from embedder import extract_embeddings
from config import wav_files, logger
from tqdm import tqdm
from cluster import auto_spectral_clustering, recluster_embeddings

if __name__ == "__main__":

    # Step 1: Run VAD
    vad_results = voice_activity_detection(wav_files[:1])
    for audio_file, vad_result in tqdm(vad_results.items(), desc="Embedding & Diarization"):

        # Step 2: Extract embeddings
        embeddings = extract_embeddings(vad_result, audio_file)

        if not embeddings:
            logger.warning(f"No speech segments found in {audio_file}")
            continue

        logger.info(
            f"First 2 embeddings for {audio_file}: "
            f"{[emb[1].shape for emb in embeddings[:2]]}"
        )

        # Step 3: Auto Spectral Clustering
        labels, n_speakers = auto_spectral_clustering(embeddings)
        logger.info(f"Estimated speakers in {audio_file}: {n_speakers}")

        #Step 4: Re-cluster for refinement
        refined_labels = recluster_embeddings(embeddings, labels)

        # Step 5: Build final annotation
        # final_annotation = build_annotation(embeddings, refined_labels)

        # Optional: Save RTTM
        # out_path = audio_file.replace(".wav", "_diarization.rttm")
        # with open(out_path, "w") as f:
        #     final_annotation.write_rttm(f)
        # logger.info(f"Saved final diarization RTTM: {out_path}")