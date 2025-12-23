# RCF: Real-Time Confidence Filtering for Event Stream Denoising

RCF is a lightweight, training-free event stream denoising framework designed for real-time and continuous operation.
The method suppresses background activity noise while preserving structural information by combining local spatio-temporal confidence with global distribution-aware confidence.

---

## 1. Motivation

Event cameras provide asynchronous event streams with high temporal resolution, but these streams often contain significant background activity noise.
Existing denoising approaches typically suffer from one or more of the following limitations:

- Learning-based methods are computationally expensive and difficult to deploy in real time
- Traditional filters rely on fixed decision rules and are fragile under distribution changes
- Aggressive filtering strategies often remove useful structural events

RCF is designed to address these issues by providing a fast, adaptive, and explainable denoising mechanism.

---

## 2. Method Overview

Each event is assigned a final confidence score by combining two components:

### Local Confidence (score1)

- Computed from a local spatio-temporal neighborhood
- Reflects short-term event support
- Suppresses isolated and temporally sparse noise events

### Global Confidence (score2)

- Computed at the block level within a temporal bin
- Each block is represented by a statistical feature vector
- Blocks are ranked according to local confidence statistics
- Top-ranked blocks are treated as signal-like, bottom-ranked blocks as noise-like
- Events inherit the global confidence of their corresponding block

### Final Decision

score = score1 × score2

An event is retained if:

score ≥ η

where η ∈ (0,1) is the only tunable parameter.

---

## 3. Design Principles

- No training required
- Single-pass processing
- Causal (only past events are used)
- Distribution-aware relative decision
- Explainable at each stage
- Suitable for real-time deployment

---

## 4. Data Format

During experiments, raw .aedat4 files are converted into .npz format with the following fields:

- t: timestamp in microseconds
- x: pixel x-coordinate
- y: pixel y-coordinate
- p: polarity (optional, currently unused)

Intermediate confidence values can also be stored:

- score1: local event confidence
- score2: global block confidence

This allows fast threshold sweeping without recomputation.

---

## 5. Project Structure (Planned)

rcf/
├── core/
├── utils/
├── experiments/
├── visualization/
├── configs/
├── README.md
└── .gitignore

---

## 6. Status

- Design finalized
- Core implementation in progress
- Parameter sweep and dataset evaluation planned

---

## 7. Notes

This repository is research-oriented and under active development.
The implementation emphasizes clarity, robustness, and reproducibility over heavy model complexity.
