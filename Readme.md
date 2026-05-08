
# PC-VLG: Position-Correlated Vision–Language Graph Alignment for Semi-Supervised Medical Image Segmentation

👉 Early Accepted (9%) at **MICCAI 2026**: This repository is the official implementation of the paper PC-VLG.

## Datasets

### FLARE21
(a) FLARE21 comprises 361 abdominal CT scans annotated for liver, kidney, spleen, and pancreas, split into 260 scans for training, 26 for validation, and 75 for testing. 


### ACDC
(b) ACDC~\cite{bernard2018deep} includes 100 cardiac MR volumes annotated for right ventricle (RV), left ventricle (LV), and myocardium (MYO). Following~\cite{wang2021tripled}, 75 cases are used for training, 5 for validation, and 20 for testing. 


## Requirements

Some important required packages include:
* Python 3.8
* CUDA 11.8
* causal-conv1d 1.2.0.post2
* einops 0.8.0
* imageio 2.35.1
* mamba-ssm 2.2.2
* matplotlib 3.7.5
* medpy 0.5.2
* nibabel 5.2.1
* opencv-python 4.11.0.86
* pillow 10.2.0
* scikit-image 0.21.0
* scipy 1.10.1
* simpleitk 2.4.1
* six 1.17.0
* tensorboardx 2.6.2.2 
* torch 2.0.0+cu118
* torchaudio 2.0.1+cu118
* torchvision 0.15.1+cu118
* tqdm 4.67.1 

## Training

To train the model, run this command:

```python train.py```



