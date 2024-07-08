import cv2

# 얼굴 인식 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 로드
image = cv2.imread('images.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 얼굴 주위에 사각형 그리기
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 결과 이미지 보기
cv2.imshow('Face Detected', image)
cv2.waitKey()