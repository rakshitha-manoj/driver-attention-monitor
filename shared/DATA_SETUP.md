# Data Setup

All three datasets are hosted on Kaggle. Each person should download the ones relevant to their module, but ideally everyone has all three locally so evaluation scripts run for anybody.

## Datasets

| Dataset | Purpose | Kaggle link |
|---|---|---|
| CEW (Closed Eyes in the Wild) | Validate EAR threshold in isolation | https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset |
| YawDD | Validate MAR threshold and yawn detection in isolation | https://www.kaggle.com/datasets/enider/yawdd-dataset |
| NTHU Drowsy Driver Detection | Primary end-to-end evaluation, train/test split, ablation, cross-subject | https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class |

## Folder structure

Download and extract each dataset into `data/` so paths match across all three machines:

```
data/
  cew/
  yawdd/
  nthu/
```

`data/` is gitignored, nothing here gets committed. Everyone sets this up locally.

## Option A: manual download

Click each Kaggle link above -> Download -> unzip into the matching folder under `data/`.

## Option B: Kaggle API (recommended, faster and repeatable)

1. Install the Kaggle CLI:
   ```
   pip install kaggle
   ```
2. Get your API token: Kaggle account settings -> Create New Token -> downloads `kaggle.json`.
3. Place `kaggle.json` in:
   - Mac/Linux: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Download each dataset:
   ```
   kaggle datasets download -d ahamedfarouk/cew-dataset -p data/cew --unzip
   kaggle datasets download -d enider/yawdd-dataset -p data/yawdd --unzip
   kaggle datasets download -d samymesbah/nthu-dataset-ddd-multi-class -p data/nthu --unzip
   ```

## After downloading

Once extracted, check the actual internal folder/file structure of each dataset (Kaggle uploads aren't always organized the same way as the original academic release) and note it here so nobody re-does this investigation:

- CEW internal structure: (fill in once inspected)
- YawDD internal structure: (fill in once inspected)
- NTHU internal structure: (fill in once inspected)

## Self-recorded data

Separate from the above. Each team member records 15-20 min at 0, 15, 25 degree camera angles, doing normal and exaggerated drowsy behaviour. Store under:

```
data/self_recorded/<name>/<angle>/
```

Also gitignored. Used for the camera-angle sensitivity experiment.