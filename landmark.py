# 에러 확인중i
mport cv2
import dlib

# 얼굴 인식 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
    
    # 결과 프레임 보기
    cv2.imshow('Face Landmarks', frame)
    
    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
