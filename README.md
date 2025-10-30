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



