# SCFS-Net

Spatial–Channel–Frequency Collaborative Network for multispectral object re-identification.

SCFS-Net learns identity representations from aligned RGB, near-infrared (NIR), and thermal infrared (TIR) images. The architecture combines three modules:

| Module | Full name | Implementation |
| --- | --- | --- |
| FSCA | Frequency-Conditioned Spatial–Channel Collaborative Attention | `modeling/fsca.py` |
| CAFD | Complete Asymmetric Identity–Modality Discrepancy Decomposition | `modeling/cafd.py` |
| MHSR | Matrix-Valued Hypergraph Sheaf Recomposition | `modeling/mhsr.py` |

`modeling/scfs_net.py` defines the model; `modeling/fusion.py` combines the three modules.

## Installation

Use Python 3.10 and an isolated environment. Install a matching PyTorch/torchvision pair for your CUDA runtime. Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m tools.download_backbone
```

The downloader fetches CLIP ViT-B/16 and verifies its SHA-256. To use an existing CLIP file, set `MODEL.CLIP_PATH`.

## Dataset preparation

| Dataset | Download | Extraction code |
| --- | --- | --- |
| RGBNT201 | [Google Drive](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH) | — |
| RGBNT100 | [Baidu Netdisk](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug) · [Google Drive](https://drive.google.com/file/d/1R4XtbfnwTYyTvaTwrEx-pRCK2tApDWjj/view) | `rjin` (Baidu) |
| MSVR310 | [Google Drive](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view) | — |

Download links: [ICPL-ReID](https://github.com/lsh-ahu/ICPL-ReID#3-prepare-datasets), [HAMNet](https://github.com/ttaalle/multi-modal-vehicle-Re-ID#dataset).

Extract the datasets into `datasets/`, or set `DATASETS.ROOT_DIR`. Keep the original filenames and the following directory structure:

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

RGBNT201 uses aligned modality files. RGBNT100 uses the original packed-image representation expected by its loader; each image contains three horizontal 256-by-128 modality tiles. MSVR310 uses identity subfolders with `vis`, `ni`, and `th` modality directories. See `data/datasets/` for the loader implementations.

Evaluation uses same-camera exclusion for RGBNT201/RGBNT100 and camera/scene filtering with the `query3` split for MSVR310.

## Training

Run from the repository root on a single GPU. The default batch size is 64.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/MSVR310.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/RGBNT201.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/RGBNT100.yml
```

Override settings after the config path, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/MSVR310.yml DATASETS.ROOT_DIR /path/to/datasets OUTPUT_DIR outputs/my_run
```

The configuration, training log, per-validation metrics, and newly trained checkpoints are written to `OUTPUT_DIR`. Best checkpoints are selected using pattern-3 validation mAP.

## Evaluation

Run evaluation with:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/MSVR310.yml MODEL.INIT_WEIGHTS False TEST.WEIGHT outputs/MSVR310/SCFSNet_best.pth
```

A full trained checkpoint contains its backbone parameters, so `MODEL.INIT_WEIGHTS False` avoids needing a separate CLIP initialization during evaluation.

The default descriptor concatenates the per-modality global representation and SCFS representation (3072 dimensions). Evaluation also supports global-only (pattern 1) and SCFS-only (pattern 2) descriptors through the Python API.

## Training protocol and implementation details

Default settings: 50 epochs, batch 64, four samples per identity, Adam, cosine schedule with warmup, and data augmentation. MSVR310 uses seed 2222; RGBNT201 and RGBNT100 use seed 1111. The number of data-loading workers is four.

The model consolidates three scales before FSCA and writes the shared correction back to all scales. CAFD uses a complete adaptive radial partition and amplitude/phase statistics. MHSR operates on the two identity branches over modality–scale nodes through learnable restriction maps.

`MODEL.SCFS.DETACH_BACKBONE=True` detaches the SCFS input features. It does **not** freeze the shared backbone: the global identity/triplet branches continue training the backbone. Training uses identity and triplet losses without the optional auxiliary loss (`USE_AUXILIARY_LOSS=False`). All three architectural modules remain active.

To disable modules for controlled experiments, use `MODEL.SCFS.USE_FSCA`, `MODEL.SCFS.USE_CAFD_DISENTANGLEMENT`, and `MODEL.SCFS.USE_MHSR`.

## Verification

```bash
python -m tests.smoke
```

This CPU-only test requires no data or weights and checks configuration loading, full-model output contracts, finite losses, and gradient propagation through FSCA, CAFD, and MHSR.

## Acknowledgements

Third-party components and licenses are listed in `THIRD_PARTY_NOTICES.md` and `licenses/`.
