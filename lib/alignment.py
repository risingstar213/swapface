import dlib
import cv2
import numpy as np
# from lib.detection import detect_face

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

# 相似度变换
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
# 寻找相似度变换，umeyama算法

def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis = 0)
    dst_mean = dst.mean(axis = 0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.dot(dst_demean.T, src_demean) / num

    d = np.ones((dim,), dtype = np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    
    T = np.eye(dim + 1, dtype = np.double)

    U, S, Vt = np.linalg.svd(A)

    rank = np.linalg.matrix_rank(A)

    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = np.dot(U, Vt)
        else: 
            temp_d = d.copy()
            temp_d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(temp_d), Vt))
    else: 
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vt.T))
    
    if estimate_scale: 
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else: 
        scale = 1.0
    
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

            


if __name__ == '__main__':
    #file_path = 'img\\raw\\000076.jpg'
    #extract_features(detect_face(file_path), 
    #    'dat\\shape_predictor_68_face_landmarks.dat')
    points_1 = np.array([   
        [2, 3],
        [4, 5],
        [7, 8],
        [8, 9]
    ])
    points_2 = np.array([   
        [20, 30],
        [40, 50],
        [70, 80],
        [80, 90]
    ])
    a = umeyama(points_1, points_2, True)
    print(a)
    
    b = a[0:2, :]
    ans = np.dot(points_1, b[:, :2]) + b[:, 2]
    print(ans)
    
    
    