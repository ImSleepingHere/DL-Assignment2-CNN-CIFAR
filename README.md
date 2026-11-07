# ARTI 502 – Assignment 2: Training-Testing CNN with CIFAR-10 (PyTorch) – Group 1

Everything is in Jupyter notebooks. The full report with all screenshots is in [`report/ARTI502_Assignment2_Group1.pdf`](report/ARTI502_Assignment2_Group1.pdf), and the run-by-run log (what changed and why) is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Results (CIFAR-10 test set, 10,000 images)

| Notebook | Model | Test accuracy |
|---|---|---|
| `assignment_2.ipynb` | **Tasks 1-5**: CNN (6 conv + BN + MaxPool, global-avg-pool head), 40 epochs | 92.94% |
| `assignment_2_resnet_run3.ipynb` / `_run4.ipynb` | ResNet-18 from scratch (+ AutoAugment + CutMix) | 95.74% / 96.83% |
| `assignment_2_vit_run5.ipynb` | ViT-B/16 pretrained, full fine-tune | 97.80% |
| `assignment_2_vitl_linear_run6.ipynb` | ViT-L/16 pretrained, frozen + linear head | 98.05% |
| `assignment_2_vitl_finetune_run7.ipynb` | ViT-L/16, linear probe + fine-tune (3 epochs) | 99.28% |
| `assignment_2_vitl_finetune_run8.ipynb` | **ViT-L/16, linear probe + fine-tune (5 epochs) – best** | **99.31%** |
| `assignment_2_regnet_cnn_run9.ipynb` | RegNetY-32GF pretrained CNN, LP-FT, epoch chosen on a validation split | 98.62% |

`assignment_2.ipynb` covers every task of the assignment: downloading CIFAR-10 into `DATA_PATH`, the CNN and its train/test functions, tensor sizes and image visualisation, all eight suggested augmentations (ToPILImage, RandomHorizontalFlip, RandomResizedCrop, ColorJitter, RandomRotation, Normalize, Gaussian / salt-and-pepper noise, ToTensor) plus an experiment on whether augmentation helps or hurts, and plots of train loss, test loss and test accuracy.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Open a notebook in VS Code / Jupyter, select the `.venv` kernel and Run All. The dataset is downloaded to `../dataset` (outside the repo) on the first run. An NVIDIA GPU with 16 GB is needed for the pretrained-model notebooks (runs 5-9). Trained weights (`*.pth`) are not committed because of their size; the notebooks recreate them.

## Group 1

| # | Name | ID |
|---|---|---|
| 1 | Khalid Waleed Alhilal (Group Leader) | 2230000788 |
| 2 | Ali Khalid Albinali | 2230003813 |
| 3 | Mishary Waleed AlAwadh | 2230006217 |
| 4 | Nawaf Alsabr | 2230004159 |
| 5 | Yasir Saleh Alghamdi | 2230004157 |
| 6 | Ibrahim Alkuwaiz | 2230002878 |
| 7 | Meshal Alrubaie | 2230005991 |
