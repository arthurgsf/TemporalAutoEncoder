from TAE import temporal_autoencoder, temporal_autoencoder_v2
from datasets import load_data
import matplotlib.pyplot as plt
from hierarchical import n_clusters, get_labels
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

if __name__ == "__main__":
    dataset = load_data("../dataset/sjr_residencial.csv")

    # separando os hashes de cada conta, da série de cada conta
    hashes = dataset[:,0].reshape(-1,1)
    dataset = dataset[:,1:].astype('float64')
    
    autoencoder, encoder, _ = temporal_autoencoder(dataset.shape[1], 1, pool_size=7)
    autoencoder.compile('SGD', loss='MSLE', metrics=['MeanAbsoluteError', 'MeanSquaredError'])
    autoencoder.fit(x=dataset, y=dataset, epochs=20)

    new_x = autoencoder.predict(dataset)
    encoded_x = encoder.predict(dataset)

    decoded_features = []
    encoded_features = []
    for i, _ in enumerate(encoded_x):
        decoded_features.append(new_x[i].ravel().tolist())
        encoded_features.append(encoded_x[i].ravel().tolist())

    decoded_dataset = pd.DataFrame(decoded_features)
    decoded_dataset.to_csv("decoded_features.csv", index=False)

    tae_encoded_dataset = pd.DataFrame(encoded_features)
    tae_encoded_dataset.to_csv("encoded_features.csv", index=False)

    n1 = n_clusters(decoded_dataset.values[:,1:], plot=True, title='Dendogram')

    n2 = n_clusters(dataset, plot=True, title='Dendogram no-TAE')

    #fiquei com preguiça de por os hashes
    n3 = n_clusters(tae_encoded_dataset.values, plot=True, title='Dendogram PCA-TAE')

    print("TAE = " + str(n1))
    print("no-TAE = " + str(n2))
    print("PCA-TAE = " + str(n3))


    # clusters por cada tipo de teste
    cmap = plt.get_cmap('jet_r')

    #TAE
    tae_labels = get_labels(n_clusters=n1, data=decoded_dataset.values)
    # no_tae_labels = get_labels(n_clusters=n1, data=dataset)
    # pca_tae_labels = get_labels(n_clusters=n1, data=tae_encoded_dataset.values)


    unique_labels = np.unique(tae_labels)
    fig, ax = plt.subplots(len(unique_labels), 1)
    for i in unique_labels:
        two_samples = dataset[tae_labels == i][:2]
        ax[i].plot(two_samples[0], c=cmap(float(i)/len(unique_labels)))
        ax[i].plot(two_samples[1], c=cmap(float(i)/len(unique_labels)))
    silhouette = silhouette_score(dataset, tae_labels)
    fig.suptitle("TAE - FEATURES DECODIFICADAS\nsilhouette score = "+ str(silhouette))
    plt.savefig("../img/tae_labels")
    plt.clf()

    # NO TAE
    
    no_tae_labels = get_labels(n_clusters=n2, data=dataset)
    # pca_tae_labels = get_labels(n_clusters=n1, data=tae_encoded_dataset.values)
    unique_labels = np.unique(no_tae_labels)
    fig, ax = plt.subplots(len(unique_labels), 1)
    for i in unique_labels:
        two_samples = dataset[no_tae_labels == i][:2]
        ax[i].plot(two_samples[0], c=cmap(float(i)/len(unique_labels)))
        ax[i].plot(two_samples[1], c=cmap(float(i)/len(unique_labels)))
    silhouette = silhouette_score(dataset, no_tae_labels)
    fig.suptitle("FEATURES SEM TAE\nsilhouette score = "+ str(silhouette))
    plt.savefig("../img/no_tae_labels")
    plt.clf()

    # NO TAE
    
    pca_tae_labels = get_labels(n_clusters=n3, data=tae_encoded_dataset.values)

    unique_labels = np.unique(pca_tae_labels)
    fig, ax = plt.subplots(len(unique_labels), 1)
    for i in unique_labels:
        two_samples = dataset[pca_tae_labels == i][:2]
        ax[i].plot(two_samples[0], c=cmap(float(i)/len(unique_labels)))
        ax[i].plot(two_samples[1], c=cmap(float(i)/len(unique_labels)))
    silhouette = silhouette_score(dataset, pca_tae_labels)
    fig.suptitle("TAE - FEATURES CODIFICADAS\nsilhouette score = "+ str(silhouette))
    plt.savefig("../img/pca_tae_labels")
    plt.clf()