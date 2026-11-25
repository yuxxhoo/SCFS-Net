# SCFS-Net

Spatial–Channel–Frequency Collaborative Network for multispectral object re-identification.

SCFS-Net learns identity representations from aligned RGB, near-infrared (NIR), and thermal infrared (TIR) images. The architecture combines three modules:

| Module | Full name | Implementation |
| --- | --- | --- |
| FSCA | Frequency-Conditioned Spatial–Channel Collaborative Attention | `modeling/fsca.py` |
| CAFD | Complete Asymmetric Identity–Modality Discrepancy Decomposition | `modeling/cafd.py` |
| MHSR | Matrix-Valued Hypergraph Sheaf Recomposition | `modeling/mhsr.py` |

`modeling/fusion.py` coordinates these modules; `modeling/scfs_net.py` defines the complete model. This release contains only the selected SCFS architecture and supporting training/evaluation infrastructure, with unrelated experimental routes removed. Third-party components remain attributed in `THIRD_PARTY_NOTICES.md` and `licenses/`.

## Installation

Use Python 3.10 and an isolated environment. Install a matching PyTorch/torchvision pair for your CUDA runtime. Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m tools.download_backbone
```

The downloader obtains the external OpenAI CLIP ViT-B/16 initialization and checks its SHA-256. No pretrained or trained weights, datasets, outputs, or server-specific paths are distributed. Downloading CLIP requires Internet access and acceptance of its upstream license. An existing compatible CLIP file may instead be specified using `MODEL.CLIP_PATH`.

## Dataset preparation

Obtain the datasets from their respective providers and retain their original splits and filenames. Place them under `datasets/`, or set `DATASETS.ROOT_DIR` to your own root. Dataset contents are not modified by the scripts.

```text
datasets/
  RGBNT201/
    train_171/{RGB,NI,TI}/
    test/{RGB,NI,TI}/
  RGBNT100/rgbir/
    bounding_box_train/
    query/
    bounding_box_test/
  MSVR310/
    bounding_box_train/
    query3/
    bounding_box_test/
```

RGBNT201 uses aligned modality files. RGBNT100 uses the original packed-image representation expected by its loader; each image contains three horizontal 256-by-128 modality tiles. MSVR310 uses identity subfolders with `vis`, `ni`, and `th` modality directories. Consult `data/datasets/` for the exact filename and directory conventions; arbitrary renamed or differently organized datasets need an explicit conversion outside the original data.

The original same-camera exclusion is retained for RGBNT201/RGBNT100. MSVR310 retains its camera/scene-aware evaluation protocol and `query3` split.

## Training

Run from the repository root. The released entry point supports a single visible GPU. Batch 64 may require reducing the batch on smaller devices; this changes the experimental setting and should be recorded.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/MSVR310.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/RGBNT201.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/RGBNT100.yml
```

Override settings after the config path, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/MSVR310.yml DATASETS.ROOT_DIR /path/to/datasets OUTPUT_DIR outputs/my_run
```

The configuration, training log, per-validation metrics, and newly trained checkpoints are written to `OUTPUT_DIR`. Best checkpoints are selected using pattern-3 validation mAP. Never use test-image appearance to select a checkpoint for a reported quantitative experiment.

## Evaluation

No trained checkpoint is included. Evaluate your own checkpoint using:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/MSVR310.yml MODEL.INIT_WEIGHTS False TEST.WEIGHT outputs/MSVR310/SCFSNet_best.pth
```

A full trained checkpoint contains its backbone parameters, so `MODEL.INIT_WEIGHTS False` avoids needing a separate CLIP initialization during evaluation. Checkpoints produced by older private experiments use different key names and are not accepted silently.

The default descriptor concatenates the per-modality global representation and SCFS representation (3072 dimensions). Evaluation also supports global-only (pattern 1) and SCFS-only (pattern 2) descriptors through the Python API.

## Training protocol and implementation details

The YAML files retain the selected experimental settings: 50 epochs, batch 64, four samples per identity, Adam, cosine schedule with warmup, and the original augmentation and retrieval protocols. MSVR310 uses seed 2222; RGBNT201 and RGBNT100 use seed 1111. Worker count is set to four for portability. Exact retraining scores are not guaranteed across hardware, software versions, or dataset reorganizations.

The efficient implementation consolidates three scales before FSCA and writes the shared correction back to all scales. CAFD uses a complete adaptive radial partition and amplitude/phase statistics. MHSR operates on the two identity branches over modality–scale nodes through learnable restriction maps.

`MODEL.SCFS.DETACH_BACKBONE=True` detaches the SCFS input features. It does **not** freeze the shared backbone: the original global identity/triplet branches continue training the backbone. The selected protocol uses identity and triplet losses without the optional auxiliary loss (`USE_AUXILIARY_LOSS=False`). All three architectural modules remain active. Do not describe an optional loss as part of a verified training objective unless it was enabled in that experiment.

To disable modules for controlled experiments, use `MODEL.SCFS.USE_FSCA`, `MODEL.SCFS.USE_CAFD_DISENTANGLEMENT`, and `MODEL.SCFS.USE_MHSR`. Retrain each ablation under the same protocol.

## Verification

```bash
python -m tests.smoke
```

This CPU-only test requires no data or weights and checks configuration loading, full-model output contracts, finite losses, and gradient propagation through FSCA, CAFD, and MHSR. It does not replace full dataset retraining. See `VERIFICATION.md` for release verification.
