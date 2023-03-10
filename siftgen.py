from pathlib import Path
import numpy as np
import cv2 as cv


# Todo: Possibly experiment with the number of descriptors that generates and how the similarity changes then

# Generate SIFT descriptors from all images stored in './images/img' and save them to './images/descriptors'
def generate_descriptors(desc_num):
    sift = cv.SIFT_create(int(desc_num))
    for img_path in sorted(Path("static/images/img").glob("*.jpg")):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

        print("Computing descriptors for image " + str(img_path) + "...")

        # Compute keyPoints and descriptors
        keyPoints, descriptors = sift.detectAndCompute(img, None)

        # Save descriptors
        np.save("./static/images/descriptors/" + img_path.stem + ".npy", descriptors)


if __name__ == '__main__':
    generate_descriptors()
