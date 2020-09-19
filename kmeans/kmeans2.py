from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
        主函数
    """
    digits = load_digits()
    dig_data = digits.data

    kmeans = KMeans(n_clusters=10)
    cluster_codes = kmeans.fit_predict(dig_data)
    # cluster_codes_ser = pd.Series(cluster_codes).value_counts()
    # cluster_codes_ser.plot(kind='bar')
    # plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(10, 8, 8)

    for ax, center in zip(axes.flat, centers):
        ax.set(xticks=[], yticks=[])
        ax.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

    plt.show()


if __name__ == '__main__':
    main()