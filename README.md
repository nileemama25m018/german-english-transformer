#  Implementing German-to-English Neural Machine Translation using Transformer



## Overview

This project implements a **Transformer-based Neural Machine Translation (NMT)** system for translating German sentences into English.

The model is implemented using **PyTorch** and trained on the **Multi30k** dataset. The project focuses on implementing the core Transformer architecture and evaluating its translation performance.

## Problem Statement

Machine Translation is the task of automatically converting text from one language into another.

The objective of this project is to translate German sentences into meaningful English sentences using a **Transformer Encoder-Decoder architecture**.

## Dataset

The project uses the **Multi30k German-English parallel dataset**.

Each sample contains:

- German sentence as the source language
- English sentence as the target language

The dataset is loaded using the Hugging Face `datasets` library.

## Model Architecture

```text
German Sentence
       |
       v
   Tokenization
       |
       v
 Token Embeddings
       |
       v
 Positional Encoding
       |
       v
 Transformer Encoder
       |
       v
 Transformer Decoder
       |
       v
 Linear + Softmax
       |
       v
English Translation
## Project Structure

```text

├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── lr_scheduler.py    # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```

