import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from inception_resnet_v1 import *


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    img = np.multiply(np.subtract(img, mean), 1 / std_adj)

    img = np.expand_dims(img, axis=0)
    return img


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


metric = "cosine"  # euclidean or cosine

threshold = 0
if metric == "euclidean":
    threshold = 0.35
elif metric == "cosine":
    threshold = 0.07


def verifyFace(img1, img2, model):
    # produce 128-dimensional representation
    img1_representation = model.predict(preprocess_image(img1))[0, :]
    img2_representation = model.predict(preprocess_image(img2))[0, :]

    if metric == "euclidean":
        img1_representation = l2_normalize(img1_representation)
        img2_representation = l2_normalize(img2_representation)

        return findEuclideanDistance(img1_representation, img2_representation)

    elif metric == "cosine":
        img1_representation = l2_normalize(img1_representation)
        img2_representation = l2_normalize(img2_representation)
        return findCosineSimilarity(img1_representation, img2_representation)


def load_model():
    model = InceptionResNetV1()

    # model = model_from_json(open("facenet_model.json", "r").read())
    model.load_weights('facenet_weights.h5')
    # model.summary()
    return model


def analysis_similar_face(folder=''):
    model = load_model()
    list_images = os.listdir(folder)
    images = []
    for img in list_images:
        img = cv2.imread(f'{folder}/{img}')
        img = cv2.resize(img, (160, 160))
        mean = np.mean(img)
        std = np.std(img)
        std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
        img = np.multiply(np.subtract(img, mean), 1 / std_adj)
        # img = preprocess_input(img)
        images.append(img)

    images = np.asarray(images)
    embeddings = model.predict(images)
    # np.save('a', embeddings[0])
    results = np.round(1. - cosine_similarity(embeddings, embeddings), 2)
    # results = np.round(1. - euclidean_distances(embeddings, embeddings), 2)
    df = pd.DataFrame(results, columns=list_images)
    print(df.head(5))
    # print(df.describe())
    plt.figure(figsize=(16, 8))
    sns.histplot(data=df.stack())
    plt.xlabel('threshold', fontsize=16)
    plt.ylabel('frequency', fontsize=16)
    plt.title(f"Histogram of Similar face {len(list_images)}", fontsize=18)
    # df.stack().plot.hist()
    # name_image = folder.split('/')[-1]
    # plt.savefig(f'reports/{name_image}.png')
    plt.show()

model = load_model()
r = verifyFace("faces/user1/1.jpg", "faces/user1/3.jpg", model)
print(r)
# analysis_similar_face('faces/user1')
