import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

np.random.seed(42)

def load_data():
    x = glob("Training/*/")
    img_list = []
    i = 0
    for sub_dir in x:
        for filename in glob(sub_dir + '/*.jpg'):
            img = cv2.imread(filename, 0)
            img_flat = img.flatten()
            img_list.append(img_flat)
            
    return img_list

data = load_data()
print(len(data))

# shape of each image after flatten (coordinates in a n-dim space)
data[0].shape
data[0]

# Applying PCA it reduces the 10000 pixel features to just 2
from sklearn.decomposition import PCA
R_data = PCA(n_components = 2).fit_transform(data)

R_data[0].shape
R_data[0]

#Applying k-means
kmeans_pca = KMeans(init='k-means++', n_clusters=10, n_init=3)
kmeans_pca.fit(R_data)

kmeans_pca.labels_

h = 10 

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = R_data[:, 0].min() - 1, R_data[:, 0].max() + 1
y_min, y_max = R_data[:, 1].min() - 1, R_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation = 'nearest',
           extent = (xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Paired,
           aspect = 'auto', origin = 'lower')

plt.plot(R_data[:, 0], R_data[:, 1], 'k.', markersize = 1)
# Plot the centroids as a white X
centroids = kmeans_pca.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker = 'x', s = 150, linewidths = 3,
            color = 'w', zorder = 10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.figure(figsize=(100,100))
plt.show()