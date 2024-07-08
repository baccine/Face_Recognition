import cv2

# 얼굴 인식 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 얼굴 주위에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 결과 프레임 보기
    cv2.imshow('Face Detected', frame)
    
    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
