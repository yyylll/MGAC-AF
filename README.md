# MGAC-AF: Multi-Granular Attribute Completion with Attention Fusion for Heterogeneous Graphs

This repository contains the official PyTorch implementation of the paper:
**"MGAC-AF: Multi-Granular Attribute Completion with Attention Fusion for Heterogeneous Graphs"**

---

### 📢 Code Release Notice
> **Update:** The source code, pre-trained models, and dataset preprocessing scripts for MGAC-AF are currently being organized. To comply with the double-blind review process and intellectual property policies, **the full implementation will be made publicly available here upon the official publication of the paper.** 

Stay tuned for updates! 🚀

---

## 📝 Abstract
Heterogeneous Graph Neural Networks (HGNNs) have demonstrated exceptional performance in modeling complex relational data. However, real-world heterogeneous graphs often suffer from the **missing attribute problem**, which significantly hinders the effective aggregation of semantic information. 

**MGAC-AF** is a novel framework designed to address this challenge through:
1. **Multi-granular Subgraph Construction:** Leveraging Cosine similarity, Euclidean distance, and Pointwise Mutual Information (PMI) to capture distinct semantic perspectives.
2. **Hierarchical Attention Mechanism:** A 2-hop aggregation strategy guided by topology-aware embeddings to complete missing attributes with high-quality localized context.
3. **Denoising Autoencoder (DAE) Enhancement:** A robust refinement module to purify completed attributes and mitigate structural noise.
4. **Adaptive Attention Fusion:** A multi-view fusion decision module to dynamically integrate representations for downstream tasks.

---

## 🏗️ Model Architecture
MGAC-AF follows a systematic pipeline as illustrated below:
*   **Stage 1:** Initial node encoding using pre-trained Hierarchical BERT.
*   **Stage 2:** Meta-path guided topological embedding via Metapath2Vec.
*   **Stage 3:** Attribute completion through hierarchical attention and Top-K selection.
*   **Stage 4:** DAE-based enhancement and Multi-granular view fusion.

**
![image](https://github.com/yyylll/MGAC-AF/blob/main/MGAC-AF-model.jpg)
---

## 📊 Benchmarks
We evaluate MGAC-AF on three major benchmarks. Our model achieves state-of-the-art results, especially on the industrial-scale USPTO-2M dataset.

| Dataset | Metric | MGAC-AF | Best Baseline | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **ACM** | Macro-F1 | **90.56%** | 89.35% | +1.21% |
| **DBLP** | Macro-F1 | **55.03%** | 49.97% | +5.06% |
| **USPTO-2M** | Macro-F1 | **76.52%** | 72.94% | +3.58% |

---

## 🛠️ Requirements (Preliminary)
The following environment will be required to run the code:
*   Python 3.8+
*   PyTorch 1.12.0+
*   DGL (Deep Graph Library) / PyG (PyTorch Geometric)
*   Transformers (HuggingFace)
*   NVIDIA A100 (40GB/80GB) is recommended for the USPTO-2M dataset.

---

## 📂 Repository Structure (Planned)
```text
.
├── data/               # Processed datasets (ACM, DBLP, USPTO-2M)
├── models/             # MGAC-AF core architecture
│   ├── bert_encoder.py
│   ├── hierarchical_attn.py
│   └── dae_refine.py
├── scripts/            # Pre-computation scripts
├── train.py            # End-to-end training pipeline
├── eval.py             # Evaluation and metrics
└── requirements.txt    # Environment dependencies
```

---
