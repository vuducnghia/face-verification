import os
import cv2
from face_recognition import face_locations
import matplotlib.pyplot as plt
from imutils import paths
import pickle

IMAGE_TEST = "images/nghia/1.jpg"
DATASET_PATH = "images/nghia"


def _blobImage(image, out_size=(300, 300), scaleFactor=1.0, mean=(104.0, 177.0, 123.0)):
    """
    input:
      image: ma trận RGB của ảnh input
      out_size: kích thước ảnh blob
    return:
      imageBlob: ảnh blob
    """
    # Chuyển sang blobImage để tránh ảnh bị nhiễu sáng
    imageBlob = cv2.dnn.blobFromImage(image,
                                      scalefactor=scaleFactor,  # Scale image
                                      size=out_size,  # Output shape
                                      mean=mean,  # Trung bình kênh theo RGB
                                      swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                      crop=False)
    return imageBlob


def _image_read(image_path):
    """
    input:
      image_path: link file ảnh
    return:
      image: numpy array của ảnh
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


image = _image_read(IMAGE_TEST)


def _extract_bbox(image, single=True):
    """
    Trích xuất ra tọa độ của face từ ảnh input
    input:
      image: ảnh input theo kênh RGB.
      single: Lấy ra 1 face trên 1 bức ảnh nếu True hoặc nhiều faces nếu False. Mặc định True.
    return:
      bbox: Tọa độ của bbox: <start_Y>, <start_X>, <end_Y>, <end_X>
    """
    bboxs = face_locations(image)
    if len(bboxs) == 0:
        return None
    if single:
        bbox = bboxs[0]
        return bbox
    else:
        return bboxs


def _extract_face(image, bbox, face_scale_thres=(20, 20)):
    """
    input:
      image: ma trận RGB ảnh đầu vào
      bbox: tọa độ của ảnh input
      face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face
    return:
      face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.
    """
    h, w = image.shape[:2]
    try:
        (startY, startX, endY, endX) = bbox
    except:
        return None
    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    # extract the face ROI and grab the ROI dimensions
    (fH, fW) = face.shape[:2]

    # ensure the face width and height are sufficiently large
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None
    else:
        return face


def _model_processing(face_scale_thres=(20, 20)):
    """
    face_scale_thres: Ngưỡng (W, H) để chấp nhận một khuôn mặt.
    """
    image_links = list(paths.list_images(DATASET_PATH))
    images_file = []
    y_labels = []
    faces = []
    total = 0
    for image_link in image_links:
        split_img_links = image_link.split("/")
        # Lấy nhãn của ảnh
        name = split_img_links[-2]
        # Đọc ảnh
        image = _image_read(image_link)
        (h, w) = image.shape[:2]
        # Detect vị trí các khuôn mặt trên ảnh. Gỉa định rằng mỗi bức ảnh chỉ có duy nhất 1 khuôn mặt của chủ nhân classes.
        bbox = _extract_bbox(image, single=True)
        # print(bbox_ratio)
        if bbox is not None:
            # Lấy ra face
            face = _extract_face(image, bbox, face_scale_thres=(20, 20))
            if face is not None:
                faces.append(face)
                y_labels.append(name)
                images_file.append(image_links)
                total += 1
            else:
                next
    print("Total bbox face extracted: {}".format(total))
    return faces, y_labels, images_file


def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def _embedding_faces(encoder, faces):
    emb_vecs = []
    for face in faces:
        faceBlob = _blobImage(face, out_size=(96, 96), scaleFactor=1 / 255.0, mean=(0, 0, 0))
        # Embedding face
        encoder.setInput(faceBlob)
        vec = encoder.forward()
        emb_vecs.append(vec)
    return emb_vecs


faces, y_labels, images_file = _model_processing()
_save_pickle(faces, "./faces.pkl")
_save_pickle(y_labels, "./y_labels.pkl")
_save_pickle(images_file, "./images_file.pkl")
embed_faces = _embedding_faces(encoder, faces)
_save_pickle(embed_faces, "./embed_blob_faces.pkl")
# bbox = _extract_bbox(image)
# face = _extract_face(image, bbox)
# plt.axis("off")
# plt.imshow(face)
# plt.show()
