# DVC exp run with Hydra

## Requirements

1. Python 3.9 or higher.

## How to run

Install dependencies:
```
pip install -r. /requirements.txt
```

For development:
```
pip install -r ./requirements.txt -r ./requirements.dev.txt
```

```
dvc exp run --queue -S model=log_reg -S model.C=1.0e-2,100  train
dvc exp run --queue -S model=dec_tree -S model.max_depth=6,12  train
dvc queue start -j <num_workers>
```
