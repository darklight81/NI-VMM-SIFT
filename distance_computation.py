from pathlib import Path

import numpy as np
import cv2 as cv

correct_match = 0.7  # Value between 0 and 1 to determine which similarity is considered a correct match


class Image:
    def __init__(self, name, similarity):
        self.name = name
        self.similarity = similarity


def nearest_neighbors(data, predict, similarity):
    distances = []
    final_distances = []

    for des1 in data:
        # Euclidean distance for each descriptor
        [distances.append(np.sqrt(np.sum((des1 - des2) ** 2))) for des2 in predict]
        distances = sorted(distances)

        # Detect correct match
        if distances[0] / distances[1] <= int(similarity)/10:
            final_distances.append(distances[0])
        distances = []

    # Number of correct matches correspond to image similarity
    vote_result = len(final_distances)
    print("final: ", len(final_distances), " / ", len(data))
    return vote_result


# Compute distance between uploaded image and all images in the database with the given method and number of
# descriptors. Return top 10 images with the highest similarity.
def compute(img_path, desc_num, method, similarity):
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
            distance = nearest_neighbors(descriptor, descriptors, similarity)
            myDict[descriptor_path.stem] = distance
    # SQFD
    else:
        pass

    imageList = []
    for x in sorted(myDict, key=myDict.get, reverse=True):
        print(x, myDict[x])
        imageList.append(Image(x, myDict[x]))

    # Return top 10 images with the highest similarity
    return imageList[:10]


if __name__ == '__main__':
    compute('static/images/uploaded.jpg', desc_num=100, method=0)
