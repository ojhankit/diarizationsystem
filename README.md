# 🎙️ Speaker Diarization Pipeline

An end-to-end **speaker diarization system** built with [Pyannote](https://github.com/pyannote/pyannote-audio), **PyTorch**, and **scikit-learn**.  
This repo takes raw audio files → detects speech → extracts embeddings → clusters → outputs RTTM files.  

> **Recent Improvements 🚀**  
> - ⚡ GPU acceleration for fast embedding extraction  
> - 📈 Eigengap heuristic for smarter speaker estimation  
> - 🧹 Affinity thresholding to reduce noise in similarity matrices  
> - 🔒 Robust handling of short / invalid segments  
> - 📝 Automatic RTTM file renaming + filename consistency  
> - 🔄 Re-clustering refinement to reduce speaker confusion  

---

## ✨ Features

- 🔊 **VAD (Voice Activity Detection)** → detects speech regions from audio  
- 🧩 **Speaker Embedding Extraction** with `pyannote/embedding`  
  - Runs on **GPU** (CUDA) for maximum speed  
  - Handles NaN/Inf and zero vectors safely  
  - Skips very short segments (<0.5s)  
- 📊 **First-Pass Clustering** (Spectral Clustering)  
  - **Eigengap heuristic** to dynamically estimate speaker count  
  - Affinity thresholding to ignore weak connections  
- 🔄 **Re-clustering Refinement** (Hierarchical Agglomerative Clustering)  
  - Iterative merging of small/close clusters  
  - Reduces **confusion errors** in DER/JER  
- 📁 **RTTM Handling**  
  - Automatically names RTTM outputs as `<audio_name>_diarization.rttm`  
  - Updates `SPEAKER` lines to include correct file IDs  
- 🖥️ **Detailed Logging** for full transparency at each stage  

---
## Repo Structure
.
├── __pycache__/                 # Python cache files
├── vad_outputs/                 # VAD-detected speech segments
├── annotation.py                # Handles RTTM parsing, annotation merging, and segment management
├── cluster.py                   # Implements spectral + hierarchical clustering logic
├── config.py                    # Centralized configuration (logging, paths, model parameters)
├── diarization.log              # Log file (main diarization run)
├── diarizationv2.log            # Log file for v2 run (refined pipeline)
├── ecapa_embedder.py            # ECAPA-TDNN embedder (SpeechBrain implementation)
├── embedder.py                  # Pyannote-based speaker embedding extractor
├── evaluate_diarization.py      # Script to evaluate diarization outputs (DER/JER computation)
├── fix_name.py                  # Fixes and renames RTTM file IDs for consistency
├── helper.py                    # Utility helper functions used across modules
├── run.sh                       # Shell script to launch diarization pipeline end-to-end
├── runner.py                    # Main pipeline orchestrator combining VAD → Embedding → Clustering → RTTM
├── utils.py                     # Shared utility functions (I/O, signal handling, error management)
├── vad.py                       # Voice Activity Detection (Pyannote-based)
└── clean_log.sh                 # Cleans old logs and temporary files

