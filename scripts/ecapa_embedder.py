import os
import torchaudio
import torch
from speechbrain.pretrained import EncoderClassifier
from pyannote.core import Segment, Annotation

# --- Load ECAPA-TDNN model ---
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# --- Directories ---
script_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(script_dir)
audio_dir = os.path.join(root_dir, "audio")
print(audio_dir)
vad_dir = os.path.join(script_dir, "vad_outputs")
print(vad_dir)
sample_rate = 16000

def read_rttm_segments(rttm_file):
    """
    Read RTTM file and return list of pyannote.core.Segment
    """
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                segments.append(Segment(start, start + duration))
    return segments

def extract_ecapa_embeddings(audio_path, segments=None):
    """
    Extract ECAPA embeddings for given audio and segments.
    """
    signal, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        signal = torchaudio.functional.resample(signal, sr, sample_rate)

    embeddings = []

    if not segments:  # whole file
        emb = classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        return [(None, emb)]

    for seg in segments:
        start = int(seg.start * sample_rate)
        end = int(seg.end * sample_rate)
        seg_signal = signal[:, start:end]
        if seg_signal.numel() == 0:
            continue
        emb = classifier.encode_batch(seg_signal).squeeze().detach().cpu().numpy()
        embeddings.append((seg, emb))

    return embeddings

if __name__ == "__main__":
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    all_embeddings = {}

    for audio_file in audio_files[:]:
        audio_path = os.path.join(audio_dir, audio_file)
        rttm_file = os.path.join(vad_dir, audio_file.replace(".wav", ".rttm"))
        if not os.path.exists(rttm_file):
            print(f"No RTTM found for {audio_file}, skipping.")
            continue

        segments = read_rttm_segments(rttm_file)
        embeddings = extract_ecapa_embeddings(audio_path, segments)
        all_embeddings[audio_file] = embeddings
        # print(f"{audio_file} : {all_embeddings[audio_file]}")
        with open("ecapa-embeddings.txt","a") as file:
            file.write(f"{audio_file} - {all_embeddings[audio_file]}")
        print(f"Extracted {len(embeddings)} embeddings for {audio_file}")
