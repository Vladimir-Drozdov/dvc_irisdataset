import pathlib
import pandas as pd
import seaborn as sns
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
sns.set()

@hydra.main(version_base="1.3", config_path=".", config_name="params")
def main(config):
    data_dir = pathlib.Path(config.data_dir)
    exp_dir = pathlib.Path(config.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    #features = np.load(data_dir / "features.npy", mmap_mode="r")
    #target = np.load(data_dir / "target.npy", mmap_mode="r")

    #train_features, test_features, train_target, test_target = train_test_split(
    #    features, target, test_size=config.train_test_split.test_size
    #)

    iris = pd.read_csv('./iris.csv')
    iris = iris.drop('Id', axis=1)
    pd.options.mode.chained_assignment = None  # предотвращаем ошибку
    iris['Species'] = pd.factorize(iris['Species'])[0]
    X_train = iris.drop('Species', axis=1)
    y_train = iris['Species']

    #print(iris)
    arrX = X_train.to_numpy()
    arrY = np.transpose(y_train.to_numpy())
    arrY = np.array(arrY, dtype='f')
    for i in range(0, 150):
        arrY[i] = np.float64(arrY[i]) / 10

    for i in range(0, 150):
        for j in range(0, 4):
            arrX[i][j] = float(arrX[i][j]) / 10

    def sigmoid(x):  # функция активации
        return 1 / (1 + np.exp(-x))

    training_inputs = np.array(arrX)
    training_outputs = np.array([arrY]).T
    np.random.seed(1)
    synaptic_weights = 2 * np.random.random((4, 1)) - 1  # генерируем случайные инициаизирующие веса:
    # Используем Perceptron
    # Метод обратного распространения
    for epochs in range(2000):
        input_layer = training_inputs
        outputs = sigmoid(np.dot(input_layer, synaptic_weights))  # np.dot-скалярное произведение

        err = training_outputs - outputs
        adjustment = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
        synaptic_weights += adjustment
        # print(outputs)
        correct = 0
        for j in range(0, 150):
            rew = round(outputs[j][0], 1)
            arrYj = np.float64(arrY[j])
            arrYj = round(arrYj, 1)
            if rew == arrYj:
                correct = correct + 1

        accuracyP = correct / 150
        with open("logs.csv", "a") as myfile:  # a-append-добавляет текст к существующему тексту
            myfile.write(f"\n{epochs},{accuracyP}")




    #model = hydra.utils.instantiate(config.model)
    #model.fit(train_features, train_target)
    #predicted_traget = model.predict(test_features)

    #accuracy = accuracy_score(test_target, predicted_traget)
    accuracy=accuracyP
    OmegaConf.save(
        OmegaConf.create({"accuracy": float(accuracy)}), exp_dir / "metrics.yaml"
    )


if __name__ == "__main__":
    main()
