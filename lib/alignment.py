import dlib
import cv2
import os
import numpy as np
from detection import detect_face

def extract_features(img, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    b, g, r = cv2.split(img)
    img_2 = cv2.merge([r, g, b])

    dets = detector(img, 1)

    img_c = img.copy()
    for index, face in enumerate(dets):

        shape = predictor(img, face)

        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(img_c, pt_pos, 2, (255, 0, 0), 1)
    cv2.imshow('img',img_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 普氏变换闭式解
def transformation_from_points(points_1, points_2):
    """
    校准 缩放
    """
    # 整型转为浮点型
    points_1 = points_1.astype(np.float64)
    points_2 = points_2.astype(np.float64)
    # 计算关键点的中心点
    c1 = np.mean(points_1, axis = 0)
    c2 = np.mean(points_2, axis = 0)

    points_1 -= c1
    points_2 -= c2
    # 计算方差
    std1 = np.std(points_1)
    std2 = np.std(points_2)

    points_1 /= std1
    points_2 /= std2

    U, S, Vt = np.linalg.svd(np.matmul(points_1.T, points_2))
    R = (U * Vt).T
    return np.vstack([
        np.hstack(((std2 / std1) * R, c2.T - (std2 / std1) * R * c1.T)),
        np.matrix([0., 0., 1.])
        ]
    )

# def umeyama()
if __name__ == '__main__':
    #file_path = 'img\\raw\\000076.jpg'
    #extract_features(detect_face(file_path), 
    #    'dat\\shape_predictor_68_face_landmarks.dat')
    points_1 = np.matrix([   
        [2, 3],
        [4, 5],
        [7, 8]
    ])
    points_2 = np.matrix([   
        [2, 3],
        [4, 5],
        [7, 8],
    ])
    a = transformation_from_points(points_1, points_2)
    print(a)
    print((points_1.T*a).T)