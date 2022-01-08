import cv2
import os
import dlib
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    if ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= ix + ih:
        return True
    else:
        return False  
def detect_people(file_name):
    img = cv2.imread(file_name)

    hog = cv2.HOGDescriptor() #定义描述子分类器
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector() 
    
    hog.setSVMDetector(detector)
     #多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
    found,w = hog.detectMultiScale(img,winStride=(5,5),padding=(4,4),scale=1.4,useMeanshiftGrouping=False)
    print('found',type(found),found.shape)
    
    #过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
    found_filtered = []
    for ri,r in enumerate(found):
        for qi,q in enumerate(found):
            #r在q内？
            if ri != qi and is_inside(r,q):
                break
        else:
            found_filtered.append(r)
            
    for person in found_filtered:
        x,y,w,h = person
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

        
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_face(file_name):
    img = cv2.imread(file_name)
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    #print(len(dets))
    for index, face in enumerate(dets):
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        extract_img = img[top:bottom + 1, left:right + 1, :]
        cv2.imwrite(os.path.join('img\\extract', str(index) + '.jpg'), extract_img)
if __name__ == '__main__':
    detect_face('img\\test\\000076.jpg')