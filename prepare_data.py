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

    data_dir = pathlib.Path(config.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    np.save(data_dir / "features.npy", features)
    np.save(data_dir / "target.npy", target)


if __name__ == "__main__":
    main()
