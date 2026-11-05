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

## Authorized choices for missing/ambiguous details

1. CAPG is referenced rather than defined in the paper. Implemented class-wise GT/prediction IoU agreement (0.5) with GT fallback, 5% box padding; unlabeled prompts use predictions >=0.6 and never access hidden labels. Not claimed to be the unavailable reference CAPG source.
2. Dual decoder attention is local visual self-attention (8x8 windows) plus prompt/query cross-attention, 8 heads, 256 channels. Four spatial sizes: 16,32,64,128, then interpolate logits to 256. Initial SAM 64-grid embedding is resized at the first stage. This resolves unspecified scaling and avoids quadratic global attention at 128-grid.
3. Six query blocks map to stages at block outputs 1,2,4,6; query cross-attention memory is 16-grid pooled encoder features. Decoder FFN width 1024, no dropout.
4. Paper Eq. 13 prints Dice on argmax outputs, which has no useful gradient. Default uses probability Dice; `literal_eq13=true` reproduces that literal expression. Both preserve detached cross-branch pseudo-label targets. Dice includes background, uses squared denominators and epsilon 1e-5.
5. Paper Eq. 5 excludes positive pairs from its contrastive denominator, giving a potentially negative/unbounded loss. Default includes positive pairs (symmetric InfoNCE); `--literal-eq5` is provided for the printed formula. Embeddings L2-normalized, tau=0.07. Text adaptation: 1000 steps, AdamW 2e-5, gradient clip 1. One sampled pair per concept per step reduces duplicate-concept negatives.
6. Starter anatomical definitions/relations are curated, not falsely labeled as the authors' unavailable PubMed corpus. PubMed BERT weights are genuine. Replace knowledge JSON to match an author-specific corpus if supplied. Default mean token pooling, 128-token limit.
7. Midline is estimated from the normalized-image body extent (intensity >0.05) unless explicitly provided. Augmentation's inverse affine maps graph coordinates back to canonical anatomy, preserving laterality after flips/rotations/zoom. No unlabeled GT is used to estimate coordinates.
8. Patient split membership is not given by the paper. Seed 1337 is recorded. FLARE21: shuffle 361 labeled cases into 260/26/75. ACDC: preserve the existing 20-patient test list and split remaining 80 into 75/5. Multiple ED/ES volumes of one patient share one split and one labeled designation. Label percentages use floor(patient_count*ratio): FLARE26/52; ACDC7/15. Official unlabeled FLARE validation images are NOT used for metrics.
9. Training: 90k iterations, batch4 (2 labeled +2 unlabeled), SGD0.03/momentum0.9/weight decay1e-4 for coarse; AdamW1e-4/weight decay0.01 for VLGR. Poly decay power0.9, Gaussian warmup10k iterations, grad clip1. AMP with numerically sensitive graph operations in FP32. Resume restores network, optimizer and RNG state; multiworker stream order after resume is not guaranteed bitwise identical.
10. Input minmax per volume to [0,1], axial slices resized256. Augmentations: rotations +/-20 degrees, flips, scale0.8-1.2. Masks use nearest interpolation. Original labels and physical spacing retained for volume evaluation. Validation every1000 iterations selects best coarse checkpoint; test is run once after training. Test results never select a checkpoint.
11. Metrics: foreground DSC/Jaccard averaged across case/phase-class pairs, physical-space HD95 and directed ASD from MedPy. Predictions restored to original slice geometry. If both masks empty: DSC/JAC1, distances0; if only one empty: DSC/JAC0, distances undefined and excluded with valid-count reporting. Alternative author aggregation/empty conventions may change reported scores.

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
