
from sklearn.datasets import make_moons , make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

X1, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X2, _ = make_moons(n_samples=300, noise=0.5, random_state=42)

kmeans_1 = KMeans(n_clusters=3).fit(X1)
kmeans_2 = KMeans(n_clusters=2,).fit(X2)



fig ,axe = plt.subplots(1,2, figsize=(10, 4))

axe[0].scatter(X1[:,0],X1[:,1],c=kmeans_1.labels_, cmap='viridis')
axe[0].set_title("This is perfect isotopic graphs")

axe[1].scatter(X2[:,1],X2[:,1], c=kmeans_2.labels_,cmap="viridis")

plt.show()






