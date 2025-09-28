from pyannote.audio import Pipeline
from pyannote.core import Timeline, Segment
from tqdm import tqdm
import os
from config import HF_ACCESS_TOKEN, wav_files, GAP_THRESHOLD, MIN_SPEECH_DURATION, output_dir, logger

pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=HF_ACCESS_TOKEN
)

def voice_activity_detection(wav_files):
    print("inside function")
    results = {}
    for file in tqdm(wav_files, desc="Running VAD"):
        print("inside loop")
        try:
            logger.info(f"Processing file: {file}")
            vad_result = pipeline(file)

            # --- Post-processing ---
            speech_timeline = vad_result.get_timeline()
            merged_timeline = Timeline()
            prev = None
            for seg in speech_timeline:
                if prev is None:
                    prev = seg
                    continue
                if seg.start - prev.end < GAP_THRESHOLD:
                    prev = Segment(prev.start, seg.end)
                else:
                    merged_timeline.add(prev)
                    prev = seg
            if prev is not None:
                merged_timeline.add(prev)

            final_timeline = Timeline([seg for seg in merged_timeline if seg.duration >= MIN_SPEECH_DURATION])
            processed_annotation = vad_result.crop(final_timeline)
            results[file] = processed_annotation

            # Save RTTM
            base_name = os.path.splitext(os.path.basename(file))[0] + ".rttm"
            out_path = os.path.join(output_dir, base_name)
            with open(out_path, "w") as f:
                processed_annotation.write_rttm(f)

            logger.info(f"Saved RTTM VAD result to: {out_path}")

        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")

    #return results


if __name__ == "__main__":

    voice_activity_detection(wav_files)