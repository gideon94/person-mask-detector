from mtcnn import MTCNN
detector = MTCNN()
from masknet.core.mask_net import get_pred_mask
# from mask_net import get_pred_mask
import cv2

def get_person(image_file):
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    if len(faces) ==0:
        return False
    result = list()
    for f in faces:
        if f['confidence'] > 0.8:
            r = dict()
            r['confidence'] = f['confidence']
            r['box'] = f['box']
            r['label'] = "Person found"
            x, y, width, height = f['box']
            roi = image[y:y+height, x:x+width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_a = cv2.resize(roi, (256, 256)) 
            roi_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2RGB)
            roi_a = cv2.resize(roi_a, dsize=(224,224) , interpolation=cv2.INTER_AREA)
            r['mask'] = get_pred_mask(roi_a)
            result.append(r)
    return result


# print(get_person("/home/madhan/Downloads/xx.jpg"))