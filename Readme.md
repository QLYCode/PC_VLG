
## PC-VLG: Position-Correlated Vision–Language Graph Alignment for Semi-Supervised Medical Image Segmentation


## Executable implementation

The repository now includes training, inference, data preparation and tests for both FLARE21 and ACDC. Original paper settings and optional ACDC stability experiments use separate entries. Implementation decisions for details absent from the paper are documented in [IMPLEMENTATION.md](IMPLEMENTATION.md); reported paper scores are not claimed as reproduced results.

### Environment

Tested training environment: Python 3.8, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, Mamba 2.2.2. Install CUDA PyTorch first, then:

```bash
pip install -r requirements.txt
pip install causal-conv1d==1.2.0.post2 mamba-ssm==2.2.2 --no-build-isolation
```

Mamba compilation requires a compatible CUDA toolkit/compiler. CPU unit tests need only PyTorch and NumPy; actual training requires CUDA. Source for the U-Mamba backbone and MedSAM encoders is vendored with third-party notices.

### Pretrained visual/textual priors

```bash
python download_pc_vlg_assets.py
python prepare_pc_vlg_text.py --knowledge configs/pc_vlg/acdc_knowledge.json \
  --pubmed-bert assets/pubmed_bert --out assets/text_acdc --gpu 3 --local-only
python prepare_pc_vlg_text.py --knowledge configs/pc_vlg/flare21_knowledge.json \
  --pubmed-bert assets/pubmed_bert --out assets/text_flare21 --gpu 3 --local-only
```

MedSAM is checked by SHA256; the PubMed model revision is recorded. Weight files are not committed. The supplied anatomy corpus is a curated starter corpus, not the unpublished original author text data. Text adaptation is followed by frozen BERT embeddings and a trainable six-block semantic decoder.

### FLARE21: download, validate, prepare and train

```bash
# Official labeled data only; resumable downloads and MD5 checks.
python download_flare21.py --root data/raw/FLARE21
# Optional unlabeled official validation images (never used for local scores):
# python download_flare21.py --root data/raw/FLARE21 --include-unlabeled-val
python bootstrap_pc_vlg_data.py flare21 --raw-root data/raw/FLARE21
python train_pc_vlg.py --config configs/pc_vlg/flare21.json --gpu 3 --ratio 0.1 --out runs/flare21_label10
python train_pc_vlg.py --config configs/pc_vlg/flare21.json --gpu 3 --ratio 0.2 --out runs/flare21_label20
```

Source: [Zenodo record 5903672](https://zenodo.org/records/5903672). The 361 labeled training cases are split 260/26/75 with seed1337. Published archive MD5 values are verified before extraction. Image/mask matching handles image channel suffix `_0000`. Labels: 0 background, 1 liver, 2 kidney, 3 spleen, 4 pancreas. Split membership is saved; no case is shared between train/val/test. Official unlabeled validation/test cases cannot provide local ground-truth metrics.



### ACDC stability experiments 

```bash
# Uses the matching baseline's best.pt to initialize both branches, with new optimizers.
python train_pc_vlg_stable.py --config configs/pc_vlg/acdc_stable10.json --gpu 3 --ratio 0.1 --out runs/acdc_stable_label10
python train_pc_vlg_stable.py --config configs/pc_vlg/acdc_stable20.json --gpu 3 --ratio 0.2 --out runs/acdc_stable_label20
```

Changes: validation-based early stopping (8 checks, interval500), EMA coarse teacher/evaluation, foreground Dice, reduced background CE weight, confidence-filtered class-balanced consistency, background-only pseudo-label rejection, weaker consistency weight (max0.02), smaller learning rates (SGD0.003/AdamW1e-5), warmup, cosine decay, plateau LR reduction and mild photometric perturbations. Teacher predictions are detached. Both optimizers skip together on AMP overflow. Baseline validation is saved before optimization, so a later degraded checkpoint cannot overwrite the best model. Validation-only tuning is the default; final test evaluation is separate. A validation DSC target above85% is tracked, not guaranteed. Set `init_checkpoint` to null for a from-scratch stability run.

### GPU queue

```bash
# Both datasets, 10% and20%, max90k iterations each, sequential GPU3 jobs.
PY=python bash launch_pc_vlg_queue.sh
# Optional: prioritize the two ACDC stability experiments, then FLARE21 paper runs.
PY=python bash launch_pc_vlg_queue.sh --stable-acdc
```

Start downloads/preprocessing separately. The queue waits for missing files and free GPU memory, prepares text banks if needed, performs two actual-model training steps, then starts formal training. Status: `runs/queue_gpu3/status.json`; per-job output: `runs/<job>/train.log`. Only one queue instance runs at once.

### Inference / metrics / tests

```bash
# Coarse network only: no MedSAM or BERT needed at inference.
python evaluate_pc_vlg.py --checkpoint runs/flare21_label10/best_coarse.pt \
  --manifest data/flare21/manifest.json --split test --gpu 3 --out flare21_test.json
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test*.py' -v
```

Validation selects checkpoints; tests never select hyperparameters. Foreground DSC/Jaccard are volume-level; HD95/ASD use original physical spacing. Empty-mask conventions are documented in IMPLEMENTATION.md. Full metrics are stored in JSON, including per-class and per-case values.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and retained licenses.
