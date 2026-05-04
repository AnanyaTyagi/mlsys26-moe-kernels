# Optimizing Fused Mixture-of-Experts Inference Kernels

This project focuses on optimizing Mixture-of-Experts (MoE) inference for large language models on NVIDIA Blackwell GPUs using expert-selective dequantization.

---

## Author

- Ananya Tyagi  

---

## Acknowledgements

We thank Modal for providing NVIDIA B200 GPU compute credits used for benchmarking.

---

## Overview

Mixture-of-Experts models improve efficiency by activating only a subset of experts for each input. However, standard implementations still process all experts, leading to wasted computation and memory usage.

This project improves performance by:

- Exploiting sparsity (processing only active experts)
- Reducing CPU–GPU synchronization overhead
- Optimizing memory usage using FP8 dequantization and broadcasting

---

## Framework and Tools

- FlashInfer-Bench → Benchmarking framework  
- FlashInfer → GPU inference library  
- PyTorch → Implementation  
- Modal → Cloud GPU execution (NVIDIA B200)

---

## Setup Instructions

### 1. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal

### 2. Download Dataset
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/flashinfer-trace

### 3. Run Benchmark

modal run scripts/run_modal.py
