# Training notebooks (separate from live app)

These notebooks train the **2 video models**. They are **not** part of the live CCTV inference server.

| Notebook | Dataset | Output weights |
|---|---|---|
| `violence/train_violence.ipynb` | Kaggle violence dataset | `violence_best.pt` |
| `accident/train_accident.ipynb` | Kaggle CCTV accident vs non-accident (`ckay16/accident-detection-from-cctv-footage`) | `accident_best.pt` |

The Hugging Face `ud-smart-city/car-accident-video` set is only a **10-clip paid preview** with no labels, so it is not used for training.

## Model choice

Both use the same simple recipe:

**EfficientNet-B0 (pretrained) + average over 8 frames**

Why:
- good accuracy for the effort
- fits free Kaggle GPUs
- no ViT / Mamba / research complexity

## How to use

1. Upload / open each notebook on **Kaggle**
2. Turn on **GPU** + **Internet**
3. Run all cells
4. Download the `.pt` file from `/kaggle/working`
5. Put it in this project's `models/` folder
6. Set paths in `.env`, then plug into the adapters

Train **violence first**, then accident.
