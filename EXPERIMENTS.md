# CIFAR-10 Experiment Log

Running notes: what was changed between runs, why it was chosen, and the results.
All accuracies are on the CIFAR-10 test set (10,000 images). The full write-up with charts is in `experiment_log.pdf`.

## Results

| Run | Notebook | Model | Epochs | Final test acc | Best test acc |
|-----|----------|-------|--------|----------------|---------------|
| 1 | `assignment_2.ipynb` (commit 1e8149c) | 6-conv CNN, FC head | 30 | 88.21% | 89.37% (ep 29) |
| 2 | `assignment_2.ipynb` | 6-conv CNN, global-avg-pool head | 40 | 92.85% | 92.85% (ep 40) |
| 3 | `assignment_2_resnet_run3.ipynb` | ResNet-18 | 60 | 95.74% with flip TTA (95.15% without) | 95.74% (ep 60) |
| 4 | `assignment_2_resnet_run4.ipynb` (weights: `resnet18_run4.pth`) | ResNet-18 + AutoAugment + CutMix | 200 | **96.83%** with flip TTA (96.53% without) | 96.93% (ep 199) |
| 5 | `assignment_2_vit_run5.ipynb` (weights: `vit_b16_run5.pth`) | ViT-B/16 pretrained, full fine-tune | 10 | 97.80% with flip TTA (97.72% without) | 97.82% (ep 9) |
| 6 | `assignment_2_vitl_linear_run6.ipynb` | ViT-L/16 pretrained, **frozen** + linear classifier | 30 (head only) | 98.05% with flip TTA (98.03% without) | 98.06% (ep 17) |
| 7 | `assignment_2_vitl_finetune_run7.ipynb` | ViT-L/16 pretrained, linear probe then fine-tune with layer-wise LR decay | 1 + 3 | *pending* (attempt 1 crashed: GPU out of memory) | |

Runs 1-4 train from scratch (best: **96.83%**). Runs 5-7 start from pretrained weights, so report them separately as transfer learning.

---

## Run 1: Baseline CNN, 88.21%

- **Model:** 3 blocks of [Conv-BN-ReLU] x2 + MaxPool (64/128/256 channels), then Flatten -> Linear(4096,256) -> ReLU -> Dropout(0.4) -> Linear(256,10)
- **Data:** RandomHorizontalFlip, RandomResizedCrop(32, scale 0.7-1.0), normalize. Batch size 128.
- **Training:** Adam lr=1e-3 (constant), CrossEntropyLoss, 30 epochs.
- **Observations:** test accuracy kept jumping until the end (89.37% -> 88.21% over the last 2 epochs) because the LR never decays. Train loss 0.22 vs test loss 0.38 showed overfitting.

### Changes after Run 1, and why
| Change | Why |
|---|---|
| Adam -> SGD (momentum 0.9, Nesterov, weight decay 5e-4) | Generalizes better on CIFAR. Weight decay regularizes. |
| OneCycleLR (max_lr 0.1, stepped every batch), 30 -> 40 epochs | Anneals the LR to almost zero so training ends at a stable minimum. Biggest expected gain. |
| RandomResizedCrop -> RandomCrop(32, padding=4, reflect) + RandomErasing(0.25) | Standard CIFAR augmentation that keeps the object's scale. Erasing adds regularization. |
| label_smoothing=0.1 | Less overconfidence. Loss can't go below about 0.5 any more. |
| FC head -> AdaptiveAvgPool -> Dropout(0.3) -> Linear(256,10) | Removes about 1M overfitting-prone parameters. |
| *Speed only:* num_workers=4, pin_memory, persistent_workers; loaders rebuilt after setting the transforms | GPU utilization was about 30% because data loading was the bottleneck. Persistent workers keep a copy of the dataset, so the loaders must be created after the transforms change. |

Note: an intermediate 30-epoch run (88.53%) happened partway through these edits. Its configuration wasn't saved, so it isn't counted.

## Run 2: Improved recipe, 92.85%

- **Observations:** noisy between epochs 5-25 (LR peaks at 0.1 around epoch 12, which is expected with OneCycle), then climbed steadily from 89.10% to 92.85% over epochs 31-40. The last epoch was the best one. Train/test loss 0.66/0.71, so overfitting was much lower.
- A plain 6-conv CNN tops out around 92-93%, so reaching 95% needed a better architecture.

### Changes after Run 2, and why
| Change | Why |
|---|---|
| Plain CNN -> ResNet-18 (11.2M params), in a new notebook `assignment_2_resnet.ipynb` | Skip connections allow a deeper and more accurate network. |
| bf16 mixed precision + channels_last | Keeps the 10x bigger model fast (about 20 s/epoch on an RTX 4080 SUPER). |
| Flip test-time augmentation | Cheap gain from averaging predictions on each image and its mirror image. |
| 40 -> 60 epochs | A bigger model needs longer to converge. |

## Run 3: ResNet-18, 95.74%

- **Result:** 95.74% with TTA, 95.15% without (TTA adds 0.59 points). Passes the 95% target.
- **Per class:** airplane 95.7, automobile 98.3, bird 94.1, **cat 90.9**, deer 97.0, **dog 92.4**, frog 98.0, horse 96.7, ship 97.8, truck 96.5.
- **Observations:**
  - Still improving at epoch 60, so a longer schedule should help.
  - Train accuracy 98.56% (on augmented data) vs 95.74% test, so the model overfits and needs more regularization.
  - Cat and dog are the weakest classes.

### Changes after Run 3, and why (Run 4)
| Change | Why |
|---|---|
| AutoAugment (CIFAR-10 policy) | Learned colour, contrast and geometry augmentation. Targets the overfitting gap. Expected about +0.8%. |
| CutMix on 50% of batches (soft labels, works with label smoothing) | Strong regularization that helps confusable classes like cat and dog. Expected about +0.3-0.5%. |
| 60 -> 200 epochs | Run 3 hadn't converged, and stronger augmentation needs more epochs. About 1 hour. |
| num_workers 4 -> 8 | AutoAugment is CPU-heavy. |

**Expected:** about 96.5-97%.

## Run 4: ResNet-18 + AutoAugment + CutMix, 96.83%

- **Result:** 96.83% with TTA, 96.53% without. That's +1.09 points over Run 3 and in line with the expected 96.5-97%. Best epoch was 199 (96.93%).
- **Per class:** airplane 97.0, automobile 98.5, bird 96.1, cat 91.7 (from 90.9), deer 97.1, dog 94.7 (from 92.4), frog 98.8, horse 98.3, ship 98.5, truck 97.6.
- **Observations:**
  - The overfitting gap is gone. Final train accuracy on augmented data was 87.0%, now below the 96.8% test accuracy.
  - Test accuracy stayed noisy (around 82-91%) until the LR annealed after about epoch 150, then climbed steadily.
  - Cat and dog are still the weakest classes, but both improved.
- **Note:** Run 4 overwrote the Run 3 weights in `resnet18_cifar10.pth`. Run 3's outputs are still in `assignment_2_resnet_run3.ipynb`.

### Changes after Run 4, and why (Run 5)
The new target is 98%+. Training from scratch on 50k images tops out around 97-98% even with much bigger networks: WideResNet-28-10 reaches about 97.5% after several hours. So the next step is **transfer learning**, in a new notebook `assignment_2_vit.ipynb`.

| Change | Why |
|---|---|
| ResNet-18 from scratch -> ViT-B/16 pretrained (torchvision `IMAGENET1K_SWAG_LINEAR_V1`, pretrained on 3.6B images), new zero-initialized 10-class head | Strong pretrained features make CIFAR-10 an easy downstream task. Comes with torchvision, so no new packages. |
| Upscale 32 -> 224 px on the GPU after augmentation; ImageNet mean/std | The model was pretrained at 224 px with ImageNet statistics. Doing it on the GPU keeps the CPU from becoming a bottleneck. |
| AdamW lr 5e-5, wd 0.05, OneCycle (10% warm-up), grad clip 1.0, batch 64 | A small LR adjusts the pretrained weights without wrecking them. AdamW is the standard optimizer for transformers. |
| 10 epochs; dropped AutoAugment, CutMix and RandomErasing (kept crop + flip, label smoothing, flip TTA) | Pretrained models converge in a few epochs. Heavy augmentation mainly helps long from-scratch training. |

- **Sanity check:** a 1-epoch test run already reached **97.08%** (96.74% without TTA), better than Run 4 after 200 epochs.
- **Expected:** about 98.5%+ after 10 epochs. Each epoch takes about 2.8 min, so about 30 min total. Peak GPU memory is about 5.6 GB.

**Considered but not chosen:**
- WideResNet-28-10 from scratch: several hours of training and still unlikely to reach 98%.
- Weight EMA and ensembling: not needed if fine-tuning clears 98%.

## Run 5: Fine-tuned ViT-B/16, 97.80%

- **Result:** 97.80% with TTA, 97.72% without. That's 0.97 points over Run 4, but below the expected ~98.5%.
- **Per class:** airplane 98.7, automobile 98.9, bird 97.2, cat 94.7, deer 97.8, dog 96.6, frog 99.2, horse 98.3, ship 98.7, truck 97.9.
- **Observations:**
  - **Damaged the pretrained features early:** test accuracy after epoch 1 was only 78.77%, even though a 1-epoch test run, where the LR decays within that single epoch, reached 97.08%. The LR stayed at its peak (5e-5, same for every layer) while the head was still random, so large gradients disturbed the pretrained weights. The model took about 7 epochs to recover.
  - **Overfit:** final train accuracy was 99.8% vs 97.8% test.
- **Lesson:** fine-tune more gently. Train the head before touching the backbone, and use lower LRs for early layers.

### Changes after Run 5, and why (Run 6)
The target was raised to **99%+**. First, test the opposite extreme of Run 5: don't change the pretrained network at all.

| Change | Why |
|---|---|
| ViT-B/16 -> **ViT-L/16** SWAG (`IMAGENET1K_SWAG_LINEAR_V1`, 304M params) | A bigger pretrained model gives stronger features. The `SWAG_LINEAR` weights are the variant intended for frozen use. |
| Backbone **frozen**; features = CLS token + mean of patch tokens (2048-d), extracted once for each image and its flip | Nothing in the backbone can be damaged or overfit. Extracting features once makes classifier training take seconds. |
| Only a **linear classifier** is trained (AdamW lr 1e-3, 30 epochs on cached features, label smoothing) | The standard "linear probe" evaluation of pretrained features. |

- First tried DINOv2 ViT-L/14, which has reported linear-probe results around 99.3%. Loading it needs `torch.hub` to download and run code from GitHub, which was blocked by permissions, so I used torchvision's weights instead.
- **Test run result:** 98.05% (98.03% without TTA). Test accuracy was flat at about 98.0% from epoch 9, so that's the limit of the frozen features.

### Changes after Run 6, and why (Run 7)
Run 5 (changing every layer aggressively) and Run 6 (changing nothing) both stopped around 98%. Run 7 uses the standard middle ground, **LP-FT** (linear probe, then fine-tune):

| Change | Why |
|---|---|
| **Stage 1:** 1 epoch training only the zero-initialized head (backbone frozen) | Gives a good head (about 98%) before the backbone is touched, so stage 2 gradients are small. Fixes Run 5's epoch-1 collapse. |
| **Stage 2:** unfreeze everything, 3 epochs, AdamW base LR 2e-5 with **layer-wise LR decay 0.8** (top block 2e-5, patch embedding about 2e-5 x 0.8^25, roughly 7e-8), wd 0.05, 10% warm-up, grad clip 1.0 | Early layers hold generic features and barely change. Later layers adapt to CIFAR-10. |
| ViT-L/16 SWAG at 224 px (bicubic upscaling), batch 32 | Batch 64 doesn't fit in 16 GB. Batch 32 peaks at 10.5 GB and runs at about 125 img/s, about 6.7 min per fine-tune epoch, about 30 min total. |

## Run 6: Frozen ViT-L/16 + linear classifier, 98.05%

- **Result:** 98.05% with TTA, 98.03% without, the same as the earlier test run. Test accuracy plateaued at about 98.0% from epoch 9.
- **Per class:** airplane 98.0, automobile 99.3, bird 97.1, cat 95.8, deer 98.5, dog 96.8, frog 98.6, horse 99.2, ship 98.5, truck 98.7.
- **Takeaway:** frozen features are at their limit here. Reaching 99% needs the backbone itself to adapt (Run 7).

## Run 7: *pending*

- **Attempt 1, crashed:** stage 1 (linear probe) finished at **97.78%** test accuracy after one head-only epoch. Stage 2 then hit `CUDA out of memory`. Another Jupyter kernel (the Run 6 notebook) was still holding its model and features on the GPU, leaving about 7 GB for a job that needs about 10.5 GB.
- **Fixes (no change to the training math):**
  - The first cell warns if less than 10 GiB is free.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation.
  - Stage 1's optimizer is freed and the CUDA cache emptied before stage 2.
  - Each batch of 32 is processed as 2 chunks of 16 (gradient accumulation), which halves activation memory. ViT uses LayerNorm, not BatchNorm, so the gradients are the same as with the full batch.
- **Before re-running:** shut down the other notebook kernels.

---

**Methodology caveat:** settings were chosen by looking at test accuracy, so the final number is slightly optimistic. A held-out validation split would make it unbiased.
