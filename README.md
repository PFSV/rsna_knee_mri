# RSNA Knee MRI

Research workspace for the [RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge).

## Local HPC setup

Keep the Git working tree and the large competition dataset on separate storage. The setup script creates an ignored local `data` symlink, so no dataset or machine-specific path is committed.

Create and validate the link after cloning:

```sh
./scripts/setup_data_link.sh /path/to/rsna/data
```

To use another mounted copy without editing tracked files:

```sh
RSNA_DATA_ROOT=/path/to/rsna/data ./scripts/setup_data_link.sh
```

The notebook already searches `data/`, so it works locally through this link while retaining its Kaggle paths. Dataset files, caches, outputs, and W&B runs are ignored by Git.

Quick verification:

```sh
test -f data/train.csv
test -d data/train_series
test -f data/test.csv
test -d data/test_series
git status --short
```
