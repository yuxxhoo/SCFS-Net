# Third-party notices

The SCFS-Net modules and integration are distributed with adapted supporting components. Renaming project entry points does not change the authorship of third-party code.

- The training/data/evaluation infrastructure and rectangular CLIP integration were adapted from the DeMo-based experimental code, including its inherited multispectral ReID infrastructure. The original MIT notice (Copyright (c) 2024 Yuhao Wang) is retained in `licenses/REID_BASE_MIT.txt`.
- The visual transformer in `modeling/clip_visual.py` derives from OpenAI CLIP (MIT, Copyright (c) 2021 OpenAI), adapted to rectangular token grids and camera embeddings. See `licenses/CLIP_MIT.txt`. The external pretrained initialization is not redistributed.
- The cosine scheduler and random-erasing implementation include code attributed to Ross Wightman/timm. Original inline notices are retained. Random erasing additionally credits Zhun Zhong and Liang Zheng. The Apache 2.0 license is included in `licenses/Apache-2.0.txt`.
- Standard ReID loss, sampling, and reranking utilities retain their original source references where present. They are supporting infrastructure rather than claims of new SCFS modules.

Upstream projects: https://github.com/openai/CLIP and https://github.com/huggingface/pytorch-image-models. Dataset rights remain with their providers. No dataset images or trained parameters are included.
