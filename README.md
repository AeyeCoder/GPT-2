# GPT-2 (124M) Pre-training on an RTX 3060

This repository contains the code, data processing pipeline, and training logs for pre-training a standard GPT-2 (124M parameter) model from scratch. The primary challenge and achievement of this project was scaling the training process to work efficiently on a consumer-grade GPU (NVIDIA RTX 3060, 12GB VRAM) using modern PyTorch optimizations.

# Dataset
The model was trained on a high-quality educational dataset, specifically a custom subset of FineWeb-Edu.

Base Data: codelion/fineweb-edu-1B dataset (approx. 1 billion tokens).

Extended Data: Added 2 raw parquet files manually downloaded from the massive FineWeb-Edu 350B dataset.

Total Tokens: Tokenization yielded roughly ~3.25 billion tokens across the combined datasets.

Processing: The data was tokenized using tiktoken (GPT-2 BPE) and saved directly to raw uint16 binary files (.bin). To accommodate memory constraints, the token streams were partitioned into discrete 50/50 chunks (train_tokens_1.bin, train_tokens_2.bin) allowing sequential, low-RAM data loading during training.

# Hardware Limitations & Optimizations
Training a 124M parameter model on an RTX 3060 requires strict VRAM management and throughput optimizations. Here is how the hardware limitations were handled:

Gradient Accumulation: A true batch size of 65,536 tokens per step was achieved without OOM (Out of Memory) errors by using grad_accum_steps=4.

Vocabulary Padding: The standard GPT-2 vocabulary size (50,257) was padded to 50,304. Making the vocab size a multiple of 64 unlocks significantly faster matrix multiplications on NVIDIA tensor cores.

torch.compile: The model was compiled using PyTorch 2.0's inductor (torch.compile), which fuses operations and heavily optimizes the computational graph (as seen by the _orig_mod. prefix in the saved state dictionary).

Fused AdamW: Used the fused implementation of the AdamW optimizer to reduce the number of memory read/writes during the weight update step, saving both time and VRAM bandwidth.

Mixed Precision: Handled via PyTorch autocast to keep the memory footprint small while maintaining numerical stability.

# Training Results
The model was trained using a cosine learning rate decay schedule (peaking around ~1e-3).

Initial Loss: Started at ~10.97.

Final Validation Loss: Dropped steadily down to ~3.5 by the end of the logged steps. It could go even better with more compute + data (basic scaling law)

Checkpoints: The best model weights were saved dynamically whenever validation loss reached a new low (model_best_final.pt).

# Inference
To run inference, the compiled model weights are stripped of their compile prefixes and loaded into the base GPT architecture.

# Reference

Single most Import reference : Lets reproduce GPT-2(124M) by andrej karpathy on YT!
