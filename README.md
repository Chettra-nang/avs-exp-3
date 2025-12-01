# Vision-Language Cooperative Driving (Highway-Env)

This repository contains every stage of the ambulance-yielding thesis project built on **highway-env**, **Stable-Baselines3**, and **CLIP/DINOv2**. It covers ego-centric dataset generation, CLIP-based classification, hybrid PPO training, MARL baselines, and evaluation utilities that run on a single RTX 3050 (4 GB).

---

## 1. Dataset Generation

| Script | Description |
| --- | --- |
| `generate_dataset.py` | Original global-view generator with MAPPO wrappers and forced ambulance interactions. |
| `generate_data_ego.py` | First ego-view attempt (kept for reference). |
| `generate_data_ego_v2.py` | **Final ego-centric generator**. For each controlled vehicle: centers the camera, paints the ambulance yellow, teleports it behind random agents, downsamples “Clear”, and labels `Yield / Hold / Clear`. |
| `audit_dataset.py` | Loads `metadata.jsonl`, prints class counts, and writes `audit_grid.png` for visual sanity checks. |
| `validate_dataset.py` | Logic audit + histogram for HuggingFace-style datasets. |

Usage:
```bash
python generate_data_ego_v2.py --output-dir data/ego_vla_v2 --samples 10000
python audit_dataset.py        # writes data/ego_vla_v2/audit_grid.png
```

---

## 2. CLIP-Based Classifiers

| Script | Description |
| --- | --- |
| `features.py` | DINOv2 ViT-S/14 extractor for SB3 (frozen). |
| `train_vla_encoder.py` | Wraps DINOv2 inside `BaseFeaturesExtractor`. |
| `train_vla_classifier.py` | Final 3-class CLIP ViT-B/32 finetuning on ego dataset. Uses `WeightedRandomSampler`, unfreezes the last vision block, LR=1e-5, and saves the checkpoint with the **best yield accuracy**. |

Command:
```bash
python train_vla_classifier.py \
    --data-dir data/ego_vla_v2 \
    --output-path models/multi/vla_classifier.pt
```

---

## 3. Hybrid PPO Training (Thesis Ready)

| Script | Description |
| --- | --- |
| `train_hybrid_ppo.py` | Original CLIP-probability hybrid PPO. |
| `train_hybrid_thesis.py` | Adds thesis metrics (collision rate, yield compliance, avg speed). |
| `train_hybrid_thesis_v2.py` | **Final thesis trainer**: domain randomization for ambulance color, CLIP embeddings + kinematics, aggressive reward shaping (yield bonus, left-lane penalty, anti-laziness speed penalty), TensorBoard logging, and checkpointing. |
| `record_videos.py` | “Director’s cut” evaluation: forces an ambulance behind the ego, paints it yellow, and records MP4s for thesis demos. |

Example:
```bash
python train_hybrid_thesis_v2.py \
    --classifier-path models/multi/vla_classifier.pt \
    --timesteps 200000 --n-envs 2 \
    --output-path models/hybrid_thesis_agent.zip
tensorboard --logdir ppo_vla_thesis_v2
python record_videos.py
```

`ThesisHybridObservationWrapper` appends CLIP probabilities to SB3’s kinematic observations, logs yield events, and enforces speed penalties for “lazy” driving. `ThesisMetricsCallback` writes `thesis/collision_rate`, `thesis/yield_compliance`, and `thesis/avg_speed` to TensorBoard.

---

## 4. MARL Baselines

| Script | Description |
| --- | --- |
| `train_marl_cooperative.py` | Flattens multi-agent observations/actions for 5 vehicles and uses a shared reward (ambulance bonus, shared crash penalty) with PPO. |
| `train_marl_vla.py` | Multi-agent hybrid: CLIP probabilities per agent + kinematics, cooperative reward `avg_speed − 10 * collisions + bonus when ambulance > 25 m/s`. |

These scripts convert highway-env’s native multi-agent interface into single-agent SB3 vectors (parameter sharing) to study cooperative yielding.

---

## 5. Evaluation / Utilities

| Script | Description |
| --- | --- |
| `audit_dataset.py` | Visual scrub of random Yield vs Clear samples. |
| `validate_dataset.py` | Histogram, logic error rate, and audit grid generation. |
| `record_videos.py` | Saves before/after MP4s for random vs trained agents using `VecVideoRecorder`. |
| `env_wrapper.py` | `AmbulanceYieldingEnv` that appends `[relative_x, relative_y, v_ambulance]` to each agent observation. |

---

## Quick Start

1. **Generate Ego Dataset**
   ```bash
   python generate_data_ego_v2.py --output-dir data/ego_vla_v2 --samples 10000
   python audit_dataset.py
   ```
2. **Train CLIP Classifier**
   ```bash
   python train_vla_classifier.py --data-dir data/ego_vla_v2 \
       --output-path models/multi/vla_classifier.pt
   ```
3. **Train Hybrid PPO (Thesis v2)**
   ```bash
   python train_hybrid_thesis_v2.py --classifier-path models/multi/vla_classifier.pt
   tensorboard --logdir ppo_vla_thesis_v2
   ```
4. **Record Video Evidence**
   ```bash
   python record_videos.py
   ```
5. **Explore MARL Baselines (optional)**
   ```bash
   python train_marl_cooperative.py
   python train_marl_vla.py --classifier-path models/multi/vla_classifier.pt
   ```

---

## Dependencies

Install requirements (PyTorch + CUDA build recommended):
```bash
pip install -r requirements.txt
```
- `gymnasium`, `highway-env>=1.8.2`
- `stable-baselines3`, `sb3-contrib` (MaskablePPO optional)
- `torch`, `transformers`, `Pillow`, `numpy`, `joblib`, `tqdm`, `matplotlib`

---

## Thesis Narrative (Suggested)

1. **Phase 1 – Ego Data**: Teleported ambulance + downsampled Clear frames ensure ≥20 % Yield coverage. Audit grid validates label/image alignment.
2. **Phase 2 – CLIP Classifier**: Balanced training with oversampling + partial unfreezing. Checkpoint fixed to best Yield recall.
3. **Phase 3 – Hybrid PPO**: RL policy consumes VLA embeddings + kinematics, enforces yield compliance, penalizes left-lane crowding and low speed, and logs thesis metrics.
4. **Phase 4 – MARL & Video**: Cooperative baselines (`train_marl_cooperative.py`, `train_marl_vla.py`) and “director’s cut” videos demonstrating learned yielding.

This repository documents the full journey from synthetic ego datasets to a thesis-grade VLA-assisted cooperative driving agent with reproducible evaluation artifacts. Good luck with your defense!
# avs-exp-3
