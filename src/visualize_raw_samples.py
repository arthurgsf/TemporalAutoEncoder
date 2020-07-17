import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = pd.read_csv("../dataset/iris.data")
    setosa = dataset[dataset['class'] == 'Iris-setosa']
    virginica = dataset[dataset['class'] == 'Iris-virginica']
    versicolor = dataset[dataset['class'] == 'Iris-versicolor']


    fig, ax = plt.subplots()
    setosa_samples = setosa.head(2).values[:, :-1]
    for s in setosa_samples:
        ax.plot(range(4), s, "b")

    virginica_samples = virginica.head(2).values[:, :-1]
    for s in virginica_samples:
            ax.plot(range(4), s, "m")

    versicolor_samples = versicolor.head(2).values[:, :-1]
    for s in versicolor_samples:
            ax.plot(range(4), s, "c")

    ax.set(xlabel='features', ylabel='feature_value',
        title='Iris Clustering')
    ax.grid()
    plt.show()