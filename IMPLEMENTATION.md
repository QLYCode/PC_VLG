# PC-VLG reproduction

Source: Paper-6229.pdf, PC-VLG: Position-Correlated Vision-Language Graph Alignment for Semi-Supervised Medical Image Segmentation.



## Paper mapping

- Coarse network: copied `networks/umambabot_2d.py` U-Mamba Bot. Inference uses this network ONLY.
- Eq. 1-3: actual MedSAM ViT-B image/prompt encoder states loaded strictly; both encoders frozen and always eval. Input scaled from normalized [0,1] to SAM pixel normalization and 1024 square. Direct module construction avoids the inherited builder's hardcoded GPU 2. Redesigned decoder is trained from scratch.
- Eq. 4-5: PubMed-abstract-pretrained Microsoft BiomedBERT, adapted using concept/definition/relation pairs; then frozen concept vectors exported. No generic BERT fallback. Segmentation trains the six-block query decoder, not BERT.
- Eq. 6-7: spatial-softmax query-guided pooling, attention centroids, sinusoidal midline-relative coordinates, learned visual/position/query node projections.
- Eq. 8-9: fully connected graph without self edges; relative dx/dy/distance embedding, 256-wide relation lookup; uncertainty gate multiplies edge softmax AFTER normalization, with no renormalization that would cancel uncertainty attenuation.
- Eq. 10-11: node FFN/message passing and node-softmax spatial projection. Four graph stages progressively feed refined maps to subsequent decoder stages.
- Eq. 12-14: detached hard pseudo labels on unlabeled samples; trainable CE+soft Dice on labeled samples; Gaussian ramp to mu=0.1.


## Files and commands

```
# Data preparation: bootstrap ACDC now; wait for verified FLARE archives then prepare it
python bootstrap_pc_vlg_data.py acdc --raw-root /path/to/ACDC/raw --acdc-test-list /path/to/test.list
python bootstrap_pc_vlg_data.py flare21 --raw-root data/raw/FLARE21
# Explicit NIfTI split JSON is also accepted
python prepare_pc_vlg_data.py --dataset acdc --split-json patient_split.json --out data/acdc
# Download assets (system CA bundle is used; TLS verification remains enabled)
python download_pc_vlg_assets.py
# Equation, gradient, leakage and four-stage integration tests
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p test_pc_vlg.py -v
# Queue: GPU3, four jobs (two datasets x10%/20%), no simultaneous jobs
bash launch_pc_vlg_queue.sh
```

Queue can skip a dataset still downloading and start another ready dataset. Each dataset's first job performs text adaptation and two full-model training steps before formal training. It checks GPU free memory before starting, locks the queue against duplicates, records states in `runs/queue_gpu3/status.json`, and continues the next job when a job completes. A failed job is recorded and is not silently retried. `waiting_inputs` is NOT reported as running training. Queue and bootstrap processes must remain alive; all are launched detached on the server.

Outputs: `runs/<dataset>_label10/` or `label20/`: `train.log`, `train.jsonl`, `val.jsonl`, `test.json`, `last.pt`, `best.pt`, `best_coarse.pt`, `run_config.json`. Only `best_coarse.pt` is needed for inference. Preflight artifacts are in separate `<dataset>_preflight/` folders.

Sources: https://github.com/bowang-lab/MedSAM ; https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract ; official FLARE21 https://zenodo.org/records/5903672 . MedSAM original-format public mirror SHA256 34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74; URL and PubMed model revision saved in `assets/asset_sources.json`.
