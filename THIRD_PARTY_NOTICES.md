# Third-party source notices

- `networks/MedSAM/segment_anything/`: SAM/MedSAM source copied from the existing project; source headers retained. License: `networks/MedSAM/LICENSE` (Apache-2.0). Upstream: https://github.com/bowang-lab/MedSAM . The inherited convenience builder contains project-specific device defaults; PC-VLG constructs encoder modules directly and does not call this builder.
- `networks/umambabot_2d.py`: existing-project U-Mamba Bot adapter. Upstream architecture: https://github.com/bowang-lab/U-Mamba (Apache-2.0). See `LICENSES/U-Mamba-Apache-2.0.txt`. Existing adapter includes `create_umamba_bot_2d` factory and local configuration choices; it is not claimed to be byte-identical upstream code.
- Mamba and dynamic-network-architectures are dependencies, not vendored here; their licenses remain with their distributions.
- MedSAM and PubMed BERT model weights and FLARE21 data are downloaded separately and are not distributed in this Git repository. Model/data provenance is recorded by download scripts.
