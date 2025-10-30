import os
import glob
import logging
from dotenv import load_dotenv

# --------------------------
# Directories
# --------------------------
curr_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(curr_dir)
dev_audio_dir = os.path.join(root_dir, "audio")
test_audio_dir = os.path.join(root_dir, "test_audio")
output_dir = os.path.join(curr_dir, "vad_outputs")
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# Logger setup (file only)
# --------------------------
logger = logging.getLogger("diarization_logger")
logger.setLevel(logging.DEBUG)  # capture everything from DEBUG upwards

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File handler only
fh = logging.FileHandler("diarization_check_1.log", mode="a")
fh.setLevel(logging.DEBUG)  # log DEBUG and above
fh.setFormatter(formatter)
logger.addHandler(fh)

# --------------------------
# HuggingFace Token
# --------------------------
dotenv_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path)
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
if HF_ACCESS_TOKEN is None:
    logger.error("HF_ACCESS_TOKEN not found in .env")
    raise ValueError("HF_ACCESS_TOKEN not found in .env")

# --------------------------
# Fetch all .wav files
# --------------------------
wav_files = glob.glob(os.path.join(dev_audio_dir, "*.wav")) \
            + glob.glob(os.path.join(test_audio_dir, "*.wav"))

if not wav_files:
    logger.warning("No .wav files found in audio directories.")
    exit(0)
else:
    logger.info(f"Found {len(wav_files)} audio files.")

# --------------------------
# VAD post-processing params
# --------------------------
GAP_THRESHOLD = 0.2
MIN_SPEECH_DURATION = 0.3
