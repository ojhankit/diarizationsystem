import os
import glob
from tqdm import tqdm
from config import logger
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
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
    reference_dir = os.path.join(base_dir, "dev")  # Add your reference RTTM directory
    os.makedirs(out_dir, exist_ok=True)

    # Find all wav files
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        logger.error(f"No .wav files found in {audio_dir}")
        exit(1)

    logger.info(f"Found {len(wav_files)} wav files in {audio_dir}")
    print(f"Found {len(wav_files)} wav files in {audio_dir}")

    # Initialize metrics
    der_metric = DiarizationErrorRate()
    jer_metric = JaccardErrorRate()
    
    processed_files = []
    successful_evaluations = []

    # Process each audio file
    for wav_path in tqdm(wav_files[:5], desc="Embedding & Clustering"):
        audio_name = os.path.basename(wav_path).replace(".wav", "")
        rttm_path = os.path.join(vad_dir, f"{audio_name}.rttm")

        if not os.path.exists(rttm_path):
            logger.warning(f"No RTTM found for {audio_name}, skipping.")
            print(f"WARNING: No RTTM found for {audio_name}, skipping.")
            continue

        logger.info(f"\n=== Processing {audio_name} ===")
        print(f"\n=== Processing {audio_name} ===")
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
            print(f"ERROR: Failed to load RTTM for {audio_name}: {e}")
            continue

        # Step 2: Extract embeddings
        embeddings = extract_embeddings(vad_result, wav_path)
        if not embeddings:
            logger.warning(f"No embeddings extracted for {audio_name}, skipping.")
            print(f"WARNING: No embeddings extracted for {audio_name}, skipping.")
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
        print(f"✅ Saved diarization RTTM: {out_path}")
        
        processed_files.append(audio_name)
        
        # Step 7: Evaluate if reference exists
        reference_rttm_path = os.path.join(reference_dir, f"{audio_name}.rttm")
        if os.path.exists(reference_rttm_path):
            try:
                # Load reference annotation
                ref_data = load_rttm(reference_rttm_path)
                reference_annotation = ref_data[audio_name]
                
                # Calculate metrics
                der_value = der_metric(reference_annotation, final_annotation)
                jer_value = jer_metric(reference_annotation, final_annotation)
                
                # Print and log metrics for this file
                metrics_msg = f"📊 Metrics for {audio_name}:"
                der_msg = f"   DER: {der_value:.4f} ({der_value*100:.2f}%)"
                jer_msg = f"   JER: {jer_value:.4f} ({jer_value*100:.2f}%)"
                
                logger.info(metrics_msg)
                logger.info(der_msg)
                logger.info(jer_msg)
                
                print(metrics_msg)
                print(der_msg)
                print(jer_msg)
                
                successful_evaluations.append({
                    'file': audio_name,
                    'der': der_value,
                    'jer': jer_value
                })
                
            except Exception as e:
                error_msg = f"Failed to evaluate {audio_name}: {e}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
        else:
            warning_msg = f"No reference RTTM found for {audio_name} at {reference_rttm_path}"
            logger.warning(warning_msg)
            print(f"WARNING: {warning_msg}")

    # Summary of metrics across all processed files
    separator = "=" * 60
    logger.info("\n" + separator)
    logger.info("EVALUATION SUMMARY")
    logger.info(separator)
    
    print("\n" + separator)
    print("EVALUATION SUMMARY")
    print(separator)
    
    summary_msg = f"Total files processed: {len(processed_files)}"
    eval_msg = f"Files with evaluation: {len(successful_evaluations)}"
    
    logger.info(summary_msg)
    logger.info(eval_msg)
    print(summary_msg)
    print(eval_msg)
    
    if successful_evaluations:
        avg_der = sum(item['der'] for item in successful_evaluations) / len(successful_evaluations)
        avg_jer = sum(item['jer'] for item in successful_evaluations) / len(successful_evaluations)
        
        avg_header = "\nAverage Metrics:"
        avg_der_msg = f"   Average DER: {avg_der:.4f} ({avg_der*100:.2f}%)"
        avg_jer_msg = f"   Average JER: {avg_jer:.4f} ({avg_jer*100:.2f}%)"
        
        logger.info(avg_header)
        logger.info(avg_der_msg)
        logger.info(avg_jer_msg)
        
        print(avg_header)
        print(avg_der_msg)
        print(avg_jer_msg)
        
        results_header = "\nIndividual Results:"
        logger.info(results_header)
        print(results_header)
        
        for item in successful_evaluations:
            result_msg = f"   {item['file']}: DER={item['der']:.4f} ({item['der']*100:.2f}%), JER={item['jer']:.4f} ({item['jer']*100:.2f}%)"
            logger.info(result_msg)
            print(result_msg)
    else:
        warning_msg = "No evaluations were performed. Check if reference RTTM files exist."
        logger.warning(warning_msg)
        print(f"WARNING: {warning_msg}")
    
    logger.info(separator)
    print(separator)