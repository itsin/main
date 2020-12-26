from glob import glob
import mahotas as mh
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial import distance

images = glob('SimpleImageDataset/*.jpg')
images[0][21:-len('00.jpg')]
im = mh.imread(images[0])
im = mh.colors.rgb2gray(im, dtype=np.uint8)
mh.features.haralick(im)

features = []
labels = []

start = time.time()
for im in images:
    labels.append(im[21:-len('00.jpg')])
    im = mh.imread(im)
    im = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(mh.features.haralick(im).ravel())

features = np.array(features)
labels = np.array(labels)

clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])
scores = cross_val_score(clf, features, labels)

sc = StandardScaler()
features = sc.fit_transform(features)
dists = distance.squareform(distance.pdist(features))

def selectImage(n, m, dists, images):

    image_position = dists[n].argsort()[m]
    image = mh.imread(images[image_position])
    return image


def plotImages(n):
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))


    for i in range(4):
        ax[i].imshow(selectImage(n, i, dists, images))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()
