import torch
from ultralytics import YOLO
import cv2
import time
import os

# 캡처 저장 디렉터리
capture_dir = "captures"
os.makedirs(capture_dir, exist_ok=True)  # 디렉터리 생성

# 양자화된 모델 로드
quantized_model = YOLO("detect_model/yolov8n.pt")
quantized_model.model.load_state_dict(torch.load("detect_model/quantized_yolov8n.pth"))

# 모델을 평가 모드로 설정
quantized_model.model.eval()

# 카메라 열기 (기본 카메라)
cap = cv2.VideoCapture(0)

# 카메라가 열리지 않으면 종료
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

frame_width = 640
frame_height = 480

# FPS 계산을 위한 변수
prev_time = time.time()

# 캡처 타이머
last_capture_time = 0  # 마지막 캡처 시간
capture_interval = 5   # 초 단위 최소 캡처 간격

# 스킵 프레임 설정
frame_skip = 9
frame_count = 0
pred_frame = 0

# FPS 평균 계산 변수
fps_samples = []
while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 프레임 리사이즈
    frame = cv2.resize(frame, (frame_width, frame_height))

    # frame_skip마다 pred_frame만큼 예측을 진행
    if frame_count % frame_skip == 0:
        pred_frame = 3  # 예측 프레임 수를 설정

    # pred_frame이 0보다 큰 동안 예측 진행
    if pred_frame > 0:
        pred_frame -= 1  # 예측 후 pred_frame 감소
        # YOLO 예측
        results = quantized_model.predict(frame, conf=0.8, save=False)

        # 탐지된 오브젝트 가져오기
        for result in results:
            boxes = result.boxes  # 탐지된 박스
            for box in boxes:
                bb = box.xyxy.numpy()[0]  # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
                conf = box.conf.numpy()[0]  # 신뢰도 점수
                clsID = int(box.cls.numpy()[0])  # 클래스 ID
                class_name = quantized_model.names[clsID]  # 클래스 이름

                # 'bird' 클래스인 경우
                if 'bird' in class_name.lower():
                    # 초록색 바운딩 박스 그리기
                    cv2.rectangle(
                        frame,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        (0, 255, 0),
                        3,
                    )
                    # 라벨 추가
                    cv2.putText(
                        frame,
                        f"{class_name} {conf:.2f}",
                        (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                    # 현재 시간과 마지막 캡처 시간 비교
                    current_time = time.time()
                    if current_time - last_capture_time > capture_interval:
                        last_capture_time = current_time

                        # 바운딩 박스 좌표를 정수로 변환
                        xmin, ymin, xmax, ymax = map(int, bb)
                        # 바운딩 박스 영역만 잘라내기
                        bird_crop = frame[ymin:ymax, xmin:xmax]
                        # 크기 리사이즈
                        bird_resized = cv2.resize(bird_crop,(224,224))
                        # 잘라낸 이미지 저장
                        capture_path = os.path.join(capture_dir, f"bird_{int(current_time)}.jpg")
                        cv2.imwrite(capture_path, bird_resized)
                        print(f"'bird' 바운딩 박스 캡처 저장: {capture_path}")
    frame_count += 1

    # FPS 계산
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # FPS 평균을 계산하여 더 부드럽게 출력
    fps_samples.append(fps)
    if len(fps_samples) > 10:
        fps_samples.pop(0)  # 마지막 10개 FPS 샘플만 유지
    avg_fps = sum(fps_samples) / len(fps_samples)

    # FPS 텍스트 추가
    cv2.putText(
        frame,
        f"FPS: {avg_fps:.2f}",
        (int(frame_width - 120), int(frame_height - 20)),  # 오른쪽 아래 위치
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 프레임 출력
    cv2.imshow("Object Detection", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
