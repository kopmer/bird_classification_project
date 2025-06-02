import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2, Preview
import libcamera
from libcamera import Transform
import torch
from ultralytics import YOLO
import cv2
import os
import concurrent.futures
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datetime import datetime
import requests#서버 통신용
import os
import websocket
import numpy as np
import shutil

# GPIO 설정 (PIR 모션 센서)
GPIO.setmode(GPIO.BCM)
pirPin = 7  # 모션 센서 GPIO 핀
servo_pin = 18 #서보모터 GPIO 핀
GPIO.setup(pirPin, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(servo_pin, GPIO.OUT)

# Picamera2 설정
picam2 = Picamera2()
transform = Transform(vflip=1)
picam2.start_preview()
preview_config = picam2.create_preview_configuration()
preview_config["transform"] = libcamera.Transform(hflip=1,vflip=1)
picam2.configure(preview_config)
#transform=Transform(vflip = True)
#picam2.Transform(vflip = True)
# 이미지 및 비디오 저장 디렉토리
capture_dir = "captures"
video_dir = "videos"
os.makedirs(capture_dir, exist_ok=True)  # 디렉토리 생성
os.makedirs(video_dir, exist_ok=True)

# YOLO 모델 초기화
yolo_model = YOLO("detect_model/yolov8n.pt")
yolo_model.eval()

# MobileNet 초기화 (새로운 모델을 사용하여 새로 구성)
mobilenet_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_classes = 6
mobilenet_model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(mobilenet_model.last_channel, num_classes)
)
mobilenet_model.load_state_dict(torch.load('mobile_bird_model.pth', map_location=torch.device('cpu')))
mobilenet_model.eval()

# MobileNet 모델의 클래스 이름들
class_names = ['Motacilla grandis', 'Parus varius', 'Corvus corone', 'Pica pica', 'Parus major', 'Passer montanus']

# 장치 설정 (GPU가 있으면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mobilenet_model.to(device)

# MobileNet을 위한 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# FPS 계산 변수
frame_width = 640
frame_height = 480
frame_skip = 12
frame_count = 0
fps_samples = []
prev_time = time.time()

# 비디오 녹화 변수
is_recording = False
video_writer = None
record_start_time = 0


# 기존 변수에 추가
last_detection_time = time.time()  # 마지막 새가 감지된 시간
def move_files(source_folder, destination_folder):
    # 소스 폴더와 대상 폴더 확인
    if not os.path.exists(source_folder):
        print(f"소스 폴더가 존재하지 않습니다: {source_folder}")
        return
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder) # 대상 폴더가 없으면 생성

    # 소스 폴더 내 파일 이동
    for file_name in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        
        # 파일인지 확인 후 이동
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_file)
        else:
            print(f"파일이 아닙니다 (건너뜀): {source_file}")
            
def generate_random_txt_files(folder_path, file_prefix, num_files=5, shape=(5, 5), value_range=(0, 100)):
    """
    랜덤한 NumPy 배열을 생성하고 여러 .txt 파일로 저장합니다.

    :param folder_path: 저장할 폴더 경로
    :param file_prefix: 파일 이름의 접두사 (예: "random_array_")
    :param num_files: 생성할 파일 개수 (기본값: 5)
    :param shape: 생성할 배열의 모양 (기본값: (5, 5))
    :param value_range: 배열 값의 범위 (기본값: (0, 100))
    """
    # 폴더가 없으면 생성
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_files):
        # 랜덤 배열 생성
        random_array = np.random.randint(value_range[0], value_range[1], size=shape)

        # 고유한 파일 이름 생성
        file_name = f"{file_prefix}{i + 1}.txt"

        # 파일 경로 생성
        file_path = os.path.join(folder_path, file_name)

        # 배열을 .txt 파일로 저장
        np.savetxt(file_path, random_array, fmt='%d')

# 서버와 연결이 되었을 때 이미지 전송을 시작하는 함수
def on_open(ws):
    image_files = sorted(os.listdir(image_folder))
    
    # 이미지를 순차적으로 전송
    for idx, image_file in enumerate(image_files[:5]):  # 5개의 이미지만 전송
        image_path = os.path.join(image_folder, image_file)
        
        with open(image_path, "rb") as img:
            image_data = img.read()
            # WebSocket을 통해 이미지 전송
            ws.send({'image': image_data, 'image_num': idx+1})
            print(f"Sent {image_file}")
        
        time.sleep(1)  # 전송 간에 잠시 대기 (서버의 처리 능력에 맞춰)
        
def on_message(ws, message):
    print(f"Received message: {message}")
    if message.get('status') == 'success':
        print(f"Image {message['image_num']} received and saved successfully")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")
        
def process_frame(frame, pred_frame):
    global is_recording, record_start_time, video_writer, last_detection_time

    # BGR 프레임을 MobileNet RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # YOLO 모델을 위한 리사이즈
    frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height))

    # 녹화 중이 아닐 때만 예측
    if not is_recording:
        if frame_count % frame_skip == 0:
            pred_frame = 3  # 예측 프레임 설정

        if pred_frame > 0:
            pred_frame -= 1
            # YOLO 예측
            results = yolo_model.predict(frame_resized, conf=0.8, save=False)

            for result in results:
                boxes = result.boxes  # 탐지된 박스
                for box in boxes:
                    bb = box.xyxy.numpy()[0]  # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)
                    conf = box.conf.numpy()[0]  # 신뢰도 점수
                    clsID = int(box.cls.numpy()[0])  # 클래스 ID
                    class_name = yolo_model.names[clsID]  # 클래스 이름

                    # 'bird' 클래스가 탐지되면
                    if 'bird' in class_name.lower():
                        cv2.rectangle(
                            frame_rgb,
                            (int(bb[0]), int(bb[1])),
                            (int(bb[2]), int(bb[3])),
                            (0, 255, 0),
                            3,
                        )
                        # 라벨 추가
                        cv2.putText(
                            frame_rgb,
                            f"{class_name} {conf:.2f}",
                            (int(bb[0]), int(bb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )
                        # 새 영역을 자른 후 리사이즈
                        bird_crop = frame[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                        bird_resized = cv2.resize(bird_crop, (224, 224))

                        # MobileNet을 위한 PIL 변환
                        bird_pil = Image.fromarray(cv2.cvtColor(bird_resized, cv2.COLOR_BGR2RGB))

                        # MobileNet 모델을 위한 전처리
                        bird_tensor = transform(bird_pil).unsqueeze(0).to(device)

                        # MobileNet 예측
                        with torch.no_grad():
                            output = mobilenet_model(bird_tensor)
                            _, predicted_class = torch.max(output, 1)
                            predicted_class = predicted_class.item()
                            predicted_class_name = class_names[predicted_class]

                        # 클래스 예측에 따라 이미지 저장
                        species_dir = os.path.join(capture_dir, predicted_class_name)
                        os.makedirs(species_dir, exist_ok=True)

                        # 새 이미지 캡처 경로
                        current_time = time.time()
                        capture_path = os.path.join(species_dir, f"bird_{int(current_time)}.jpg")
                        cv2.imwrite(capture_path, frame_rgb)
                        print(f"'{predicted_class_name}' 이미지 저장: {capture_path}")

                        # 비디오 녹화 시작
                        is_recording = True
                        record_start_time = current_time

                        # 비디오 기록을 위한 VideoWriter 초기화
                        video_species_dir = os.path.join(video_dir, predicted_class_name)
                        os.makedirs(video_species_dir, exist_ok=True)
                        video_path = os.path.join(video_species_dir, f"bird_{int(current_time)}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
                        print(f"'{predicted_class_name}' 비디오 녹화 시작: {video_path}")

                        # 새 감지 시간을 업데이트
                        last_detection_time = time.time()

    # 녹화 중이면 프레임을 비디오에 기록
    if is_recording:
        current_time = time.time()
        video_writer.write(frame_rgb)  # 프레임 기록

        # 5초 후 녹화 종료
        if current_time - record_start_time > 5:
            is_recording = False
            video_writer.release()  # VideoWriter 해제
            if 'predicted_class_name' in locals():
                print(f"'{predicted_class_name}' 비디오 녹화 완료")

    return frame_rgb, pred_frame
camera_on = False

# 메인 루프: 프레임 캡처 및 처리
with concurrent.futures.ThreadPoolExecutor() as executor:

    picam2.start()
    pwm = GPIO.PWM(servo_pin, 30)  # 50Hz (서보모터 PWM 동작을 위한 주파수)
    pwm.start(4.0) 
    time.sleep(3.0)
    pwm.stop() #서보모터 정리
    print("프로그램 시작")
    try:
        while True:
            
            pir_state = GPIO.input(pirPin)
            # 현재 시간
            current_hour = datetime.now().hour
            # 오전 6시 ~ 오후 7시에 작동(오전 동작)
                        
            # 오후 7시 ~ 오전 6시에 작동(저녁 동작)            
            if 0 <= current_hour < 24:
                if current_hour == 6:
                    pwm = GPIO.PWM(servo_pin, 30)  # 50Hz (서보모터 PWM 동작을 위한 주파수)
                    pwm.start(4.0) 
                    time.sleep(3.0)
                    pwm.stop() #서보모터 정리
                if current_hour == 18:
                    pwm = GPIO.PWM(servo_pin, 30)  # 50Hz (서보모터 PWM 동작을 위한 주파수)
                    pwm.start(4.0) 
                    time.sleep(3.0)
                    pwm.stop() #서보모터 정리
                # PIR 모션 센서가 트리거된 경우
                if pir_state == True:
                    camera_on = True
                    motion_Detection_time=time.time()
                if camera_on == True:
                    frame = picam2.capture_array()
                    frame_count += 1

                    # 프레임 처리
                    future = executor.submit(process_frame, frame, pred_frame=0)
                    frame, pred_frame = future.result()

                    # 프레임 화면에 표시
                    cv2.imshow("camera", frame)
                    if last_detection_time and time.time() - last_detection_time > 30 and time.time() - motion_Detection_time > 30:
                        print("30초 동안 새가 감지되지 않았습니다. 카메라 off")
                        camera_on = False
                        continue
            else:
                # 동작하지 않는 시간에는 카메라를 끔
                if camera_on:
                    print("작동 시간이 아닙니다. 카메라 stop")
                    camera_on = False          
                    continue
            # 'q'를 눌러 루프 종료
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # 자원 해제
        if video_writer is not None:
            video_writer.release()  # 비디오 writer 해제
        picam2.stop()  # 카메라 정리
        pwm.stop() #서보모터 정리
        cv2.destroyAllWindows()  # OpenCV 창 종료
        GPIO.cleanup()  # GPIO 해제
