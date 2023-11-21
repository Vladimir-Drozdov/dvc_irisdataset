import pathlib

import hydra
import numpy as np
from sklearn.datasets import make_classification


@hydra.main(version_base="1.3", config_path="configs", config_name="prepare_data")
def main(config):
    features, target = make_classification(
        n_samples=config.dataset.n_samples,
        n_features=config.dataset.n_features,
        n_classes=config.dataset.n_classes,
        flip_y=config.dataset.flip_y,
    )

    out_dir = pathlib.Path(config.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "target.npy", target)


if __name__ == "__main__":
    main()
