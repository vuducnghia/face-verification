import cv2
import os
import dlib
from face_recognition import face_locations

import numpy as np
import cv2

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
        return False
    return image[top:bottom, left:right]


def extract_frame(video='videos/MOVI0033.avi'):
    name = video.split('/')[-1].split('.')[0]
    if not os.path.isdir(f'images/{name}'):
        os.mkdir(f'images/{name}')
    print(name)
    vid = cv2.VideoCapture(video)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    print(totalFrames)
    index = 1
    while True:
        vid.set(cv2.CAP_PROP_POS_MSEC, index * 5 * fps)
        ret, frame = vid.read()

        if index * 5 * fps < totalFrames:
            cv2.imwrite(f'images/{name}/{index}.jpg', frame)

            index += 1
        else:
            break
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def extract_faces(folder='images', face_scale_thres=(20, 20), subfolder=None):
    if subfolder:
        if not os.path.isdir(f'faces/{subfolder}'):
            os.mkdir(f'faces/{subfolder}')
        for path in os.listdir(f'{folder}/{subfolder}'):
            image = cv2.imread(f'{folder}/{subfolder}/{path}')
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = alignment_face(image)
            print(path)
            if face is False:
                continue
            cv2.imwrite(f'faces/{subfolder}/{path}', face)
            # bboxs = face_locations(image)
            # for box in bboxs:
            #     (startY, startX, endY, endX) = box
            #     minX, maxX = min(startX, endX), max(startX, endX)
            #     minY, maxY = min(startY, endY), max(startY, endY)
            #     face = image[minY:maxY, minX:maxX].copy()
            #
            #     (fH, fW) = face.shape[:2]
            #
            #     # ensure the face width and height are sufficiently large
            #     if fW > face_scale_thres[0] and fH > face_scale_thres[1]:
            #         cv2.imwrite(f'faces/{subfolder}/_{path}', face)
    else:
        for subfolder in os.listdir(folder):
            if not os.path.isdir(f'faces/{subfolder}'):
                os.mkdir(f'faces/{subfolder}')
            for path in os.listdir(f'{folder}/{subfolder}'):
                image = cv2.imread(f'{folder}/{subfolder}/{path}')
                # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # face = alignment_face(image)
                # cv2.imwrite(f'faces/{subfolder}/{path}', face)
                bboxs = face_locations(image)
                for box in bboxs:
                    (startY, startX, endY, endX) = box
                    minX, maxX = min(startX, endX), max(startX, endX)
                    minY, maxY = min(startY, endY), max(startY, endY)
                    face = image[minY:maxY, minX:maxX].copy()

                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW > face_scale_thres[0] and fH > face_scale_thres[1]:
                        cv2.imwrite(f'faces/{subfolder}/_{path}', face)


scale = 4


def alignment_face(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    height, width = img.shape[:2]
    img = cv2.resize(img, (width, height))
    # cv2.imshow('img', img)
    # cv2.waitKey()
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
        # cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
        # if cropped is False:
        #     return False
        # cv2.imshow('', cropped)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imwrite('faces/user1/1_align.jpg', cropped)
        return cropped


# extract_frame(video='videos/MOVI0037.avi')
extract_faces(subfolder='IPS_2021_05_10_09_30_23_3400')
# img = cv2.imread('images/IPS_2021_05_10_09_21_04_8840/img013.jpg')
# alignment_face(img)
