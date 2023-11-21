import pathlib

import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config):
    data_dir = pathlib.Path(config.data_dir)
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    features = np.load(data_dir / "features.npy", mmap_mode="r")
    target = np.load(data_dir / "target.npy", mmap_mode="r")

    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=config.train_test_split.test_size
    )

    model = hydra.utils.instantiate(config.model)
    model.fit(train_features, train_target)
    predicted_traget = model.predict(test_features)

    accuracy = accuracy_score(test_target, predicted_traget)
    OmegaConf.save(
        OmegaConf.create({"accuracy": float(accuracy)}), exp_dir / "metrics.yaml"
    )


if __name__ == "__main__":
    main()
