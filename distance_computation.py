from pathlib import Path

import numpy as np
import cv2 as cv

correct_match = 0.7  # Value between 0 and 1 to determine which similarity is considered a correct match


class Image:
    def __init__(self, name, similarity):
        self.name = name
        self.similarity = similarity


def nearest_neighbors(data, predict):
    distances = []
    final_distances = []
    for des1 in data:

        # Euclidean distance for each descriptor
        [distances.append(np.sqrt(np.sum((des1 - des2) ** 2))) for des2 in predict]
        distances = sorted(distances)

        # Detect correct match
        if distances[0] / distances[1] <= correct_match:
            final_distances.append(distances[0])
        distances = []

    # Number of correct matches correspond to image similarity
    vote_result = len(final_distances)
    print("final: ", len(final_distances), " / ", len(data))
    return vote_result

#===============================================================
import cv2
from typing import Dict, List, Tuple
import numpy.typing as npt

Descriptor = npt.ArrayLike
Weight = float


def euclidean_distance(v1: Descriptor, v2: Descriptor) -> float:
    return np.sqrt(np.sum((v1 - v2) ** 2))


def similarity_f(r1: Descriptor, r2: Descriptor) -> float:
    return np.e ** (
        -30 * euclidean_distance(r1, r2) ** 2
    )  # alt: 1. / (1. + euclidean_distance(r1, r2))


def computeSQFD(
    data: List[Tuple[Descriptor, Weight]],
    predict: List[Tuple[Descriptor, Weight]],
):
    """
    features signatures are array of (centroid, weight of cluster)
    """

    n = len(data)
    m = len(predict)
    print(data)
    all_features = data + predict
    a = np.ndarray(shape=(n + m, n + m))
    for i, c_i in enumerate(all_features):
        for j, c_j in enumerate(all_features):
            a[i][j] = similarity_f(c_i[0], c_j[0])

    weights = np.array(
        [float(weight) for _, weight in data]
        + [-float(weight) for _, weight in predict],
    )

    return np.sqrt(np.dot(np.dot(weights, a), weights.T))

def compute_signatures(
    centroids_of_features, original_features_len: int, cluster_sizes: Dict[int, int]
):
    """
    For each centroid compute its clusters size / original size of features. The result value is cluster's weight
    """
    return [
        (centroid, cluster_sizes[i] / original_features_len)
        for i, centroid in enumerate(centroids_of_features)
    ]

def make_clusters(features):
    """

    """
    _, labels, feature_centroids = cv2.kmeans(
        features,
        50,
        None,
        (cv2.TERM_CRITERIA_EPS, None, .000000001),
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    cluster_sizes = {}
    for [label] in labels:
        cluster_sizes.setdefault(label, 0)
        cluster_sizes[label] += 1
    return cluster_sizes, feature_centroids

#===============================================================

# Compute distance between uploaded image and all images in the database with the given method and number of
# descriptors. Return top 10 images with the highest similarity.
def compute(img_path, desc_num, method):
    sift = cv.SIFT_create(int(desc_num))
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

    # Compute keyPoints and descriptors
    keyPoints, descriptors = sift.detectAndCompute(img, None)
    myDict = {}
    # Euclidean distance
    if method == 0:
        # Iterate over all image descriptors in images/descriptors and return closest matches
        for descriptor_path in sorted(Path("static/images/descriptors").glob("*.npy")):
            # Load descriptor
            distance = 0
            descriptor = np.load(descriptor_path)

            print("Distance between " + str(img_path) + " and " + str(descriptor_path) + ": " + str(distance))
            # Save distance with image name
            myDict[descriptor_path.stem] = distance
            distance = nearest_neighbors(descriptor, descriptors)
    # SQFD
    elif method == 1:
        features = np.float32(np.array([kp.pt for kp in keyPoints]))
        original_features_len = len(features)
        cluster_sizes, clustered_features = make_clusters(features)
        descriptor = compute_signatures(
            clustered_features, original_features_len, cluster_sizes
        )
        result = {}
        for descriptor_path in sorted(Path("static/images/descriptors").glob("*.npy")):
            # descriptor = np.load(descriptor_path)
            result[descriptor_path.stem] = computeSQFD(descriptor, descriptors)

        # makes array of [key, value]
        results_arr = result.items()
        # sorts array by second element (value), and gets only n_results number of pictures
        n_results = 10 # maximum number of the best match images
        results_sorted = sorted(results_arr, key=lambda item: item[1])[:n_results]
        # creates map (key, value)
        myDict = dict(results_sorted)

        # distance = cv.xfeatures2d.PCTSignaturesSQFD.computeQuadraticFormDistance(descriptor, descriptors)

    imageList = []
    for x in sorted(myDict, key=myDict.get, reverse=True):
        print(x, myDict[x])
        imageList.append(Image(x, myDict[x]))
    # Return top 10 images with the highest similarity
    return imageList[:10]


if __name__ == '__main__':
    compute('static/images/uploaded.jpg', desc_num=100, method=0)
