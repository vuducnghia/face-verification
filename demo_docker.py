import numpy as np
import os
import cv2
import dlib
from inception_resnet_v1 import *


def preprocess_image(img):
    img = cv2.resize(img, (160, 160))
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    img = np.multiply(np.subtract(img, mean), 1 / std_adj)

    img = np.expand_dims(img, axis=0)
    return img


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def verifyFace(img1, img2, model):
    # produce 128-dimensional representation
    img1_representation = model.predict(preprocess_image(img1))[0, :]
    img2_representation = model.predict(preprocess_image(img2))[0, :]

    if distance == "euclidean":
        img1_representation = l2_normalize(img1_representation)
        img2_representation = l2_normalize(img2_representation)

        return findEuclideanDistance(img1_representation, img2_representation)

    elif distance == "cosine":
        return findCosineSimilarity(img1_representation, img2_representation)


def load_model():
    model = InceptionResNetV1()

    # model = model_from_json(open("facenet_model.json", "r").read())
    model.load_weights('facenet_weights.h5')
    model.summary()
    return model


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]


def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom


def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)


def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6


def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)


def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)


def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))


def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M


def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        print('mat bi thieu')
        return False
    return image[top:bottom, left:right]


def alignment_face(path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(path)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width, height))

    dets = detector(img, 1)
    if len(dets) == 0:
        return False

    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)

        cropped = crop_image(rotated, det)

        return cropped


model = load_model()
distance = os.environ["distance"]
thresh = float(os.environ["thresh"])
if not distance:
    distance = "cosine"

if not thresh:
    thresh = 0.2
# distance = "cosine"
# thresh = 0.2
align_img1 = alignment_face('0.jpg')
align_img2 = alignment_face('1.jpg')
if align_img1 is False:
    print('hinh anh 0 khong du chat luong')
if align_img2 is False:
    print('hinh anh 1 khong du chat luong')
else:
    r = verifyFace(align_img1, align_img2, model)
    if r > thresh:
        print('khong cung 1 nguoi', r)
    else:
        print('cung 1 nguoi', r)
