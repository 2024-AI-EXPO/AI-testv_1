from keras.models import load_model
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
size = (64, 64)
model = load_model('/home/modeep3/바탕화면/AI-testv_1/model64.keras')

labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
result_dict = {labels_dict[i]:i for i in labels_dict}

def preprocessing(frame):
    frame = cv2.resize(frame, size)
    print(frame.shape)
    image = np.array([frame])
    print(image.shape)
    image = image.astype('float32')/255.0
    print(image.shape)
    return image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('카메라 오류 발생')
        break
    print(frame.shape)
    cv2.imshow('frame', frame)
    pre_image = preprocessing(frame)
    result = model.predict(pre_image).squeeze()
    idx = int(np.argmax(result))
    text = result_dict[idx]
    frame = cv2.resize(frame, dsize=(720,1080), interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, text, org=(540, 320), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('ASL_Transform', frame)
    if cv2.waitKey(100) == ord('q'):
        print('프로그램을 종료합니다.')
        break

cap.release()
cv2.destroyAllWindows()