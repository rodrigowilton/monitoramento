import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for
from flask_socketio import SocketIO
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
import time

app = Flask(__name__)
socketio = SocketIO(app)


# Constantes geral
USERNAME = 'admin'
PASSWORD = '102030Aa'
USERNAME1 = 'tattica'
PASSWORD1 = '102030Aa@'
PORT = '554'
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# Configurações das câmeras com área específica
cameras = [
    {'ip': '10.0.62.11', 'name': 'Paróquea', 'channels': ["1"], 'area': [200, 200, 1100, 500], 'USERNAME': USERNAME1,
     'PASSWORD': PASSWORD1},
    {'ip': '10.0.10.11', 'name': 'Floreça', 'channels': ["3"], 'area': [200, 250, 1100, 500], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.26.11', 'name': 'Angelo Rizzo', 'channels': ["5"], 'area': [330, 90, 350, 250], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.95.11', 'name': 'Porto Oceanico', 'channels': ["8"], 'area': [350, 10, 500, 300], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.168.11', 'name': 'Blue Tower', 'channels': ["9"], 'area': [600, 130, 430, 150], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.51.11', 'name': 'Paradíse', 'channels': ["8"], 'area': [600, 130, 430, 150], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.9.11', 'name': 'Rio Branco', 'channels': ["2"], 'area': [200, 200, 1100, 500], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.9.11', 'name': 'Rio Branco', 'channels': ["10"], 'area': [200, 200, 1100, 500], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
    {'ip': '10.0.9.11', 'name': 'Rio Branco', 'channels': ["15"], 'area': [200, 200, 1100, 500], 'USERNAME': USERNAME,
     'PASSWORD': PASSWORD},
]

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# Inicializa a captura de vídeo
videos = []


def initialize_videos():
    global videos
    videos = []
    for cam in cameras:
        for channel in cam['channels']:
            url = f'rtsp://{cam["USERNAME"]}:{cam["PASSWORD"]}@{cam["ip"]}:{PORT}/cam/realmonitor?channel={channel}&subtype=0'
            video = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not video.isOpened():
                print(f"Falha ao conectar na câmera {cam['ip']} canal {channel}")
            videos.append(video)


initialize_videos()

modelo = YOLO('yolov8n.pt')

screen_res = (1920, 1080)

capture_running = True

detected_ips = {}


def save_image(img, ip):
    # Cria uma pasta específica para cada IP
    ip_folder = os.path.join('detected_images', ip)
    if not os.path.exists(ip_folder):
        os.makedirs(ip_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(ip_folder, f"{timestamp}.jpg")
    cv2.imwrite(filename, img)
    return filename


def reconnect_camera(index):
    global videos
    cam = cameras[index]
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Tentando reconectar na câmera {cam['ip']} (tentativa {attempt + 1}/{MAX_RETRIES})")
            for channel in cam['channels']:
                url = f'rtsp://{cam["USERNAME"]}:{cam["PASSWORD"]}@{cam["ip"]}:{PORT}/cam/realmonitor?channel={channel}&subtype=0'
                video = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if video.isOpened():
                    videos[index] = video
                    print(f"Reconexão bem-sucedida na câmera {cam['ip']} canal {channel}")
                    return
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Erro ao reconectar na câmera {cam['ip']}: {e}")
    print(f"Falha ao reconectar na câmera {cam['ip']} após {MAX_RETRIES} tentativas")


def gen_frames():
    global capture_running
    while capture_running:
        for i, video in enumerate(videos):
            check, img = video.read()
            if not check:
                print(f"Falha ao ler frame da câmera {i + 1}")
                reconnect_camera(i)
                continue

            img = cv2.resize(img, (screen_res[0] // 2, screen_res[1] // 2))
            overlay = np.zeros_like(img, dtype=np.uint8)
            alpha = 0.4

            resultado = modelo(img)
            cam_area = cameras[i]['area']
            cam_ip = cameras[i]['ip']
            cam_name = cameras[i]['name']

            area_detected = False

            for objeto in resultado:
                for dados in objeto.boxes:
                    x1, y1, x2, y2 = map(int, dados.xyxy[0])
                    cls = int(dados.cls[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if cam_area[0] <= cx <= cam_area[0] + cam_area[2] and cam_area[1] <= cy <= cam_area[1] + cam_area[
                        3]:
                        if cls == 0:
                            area_detected = True
                            cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), 2)
                            cv2.putText(img, "INVASOR DETECTADO", (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                        4)

                    if cls == 0:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if area_detected:
                cv2.rectangle(overlay, (cam_area[0], cam_area[1]),
                              (cam_area[0] + cam_area[2], cam_area[1] + cam_area[3]),
                              (0, 0, 0), -1)
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                filename = save_image(img, cam_name)
                if cam_name not in detected_ips:
                    detected_ips[cam_name] = []
                detected_ips[cam_name].append(filename)
                socketio.emit('invasion_detected', {'ip': cam_name, 'image_url': filename})

                _, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


@app.route('/images/<name>')
def list_images(name):
    image_folder = os.path.join('detected_images', name)
    if not os.path.exists(image_folder):
        return jsonify([])  # Retorna uma lista vazia se a pasta não existir
    images = [img for img in os.listdir(image_folder)]
    images_with_dates = [{'filename': img, 'date': img.split('_')[1].replace('.jpg', '')} for img in images]
    images_with_dates.sort(key=lambda x: x['date'], reverse=True)
    image_urls = [{'url': os.path.join('/detected_images', name, img['filename']), 'date': img['date']} for img in
                  images_with_dates]
    return jsonify(image_urls)


@app.route('/detected_images/<path:filename>')
def serve_image(filename):
    return send_from_directory('detected_images', filename)


@app.route('/view_images/<name>')
def view_images(name):
    image_folder = os.path.join('detected_images', name)
    if not os.path.exists(image_folder):
        return render_template('view_images.html', images=[], ip=name)  # Retorna uma lista vazia se a pasta não existir
    images = [img for img in os.listdir(image_folder)]
    images_with_dates = [{'filename': img, 'date': img.split('_')[1].replace('.jpg', '')} for img in images]
    images_with_dates.sort(key=lambda x: x['date'], reverse=True)
    image_urls = [{'url': os.path.join('/detected_images', name, img['filename']), 'date': img['date']} for img in
                  images_with_dates]
    return render_template('view_images.html', images=image_urls, ip=name)


@app.route('/delete_folder/<name>', methods=['POST'])
def delete_folder(name):
    folder_path = os.path.join('detected_images', name)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(folder_path)
    return redirect(url_for('index'))


@app.route('/')
def index():
    detected_ips = {name: [] for name in os.listdir('detected_images') if
                    os.path.isdir(os.path.join('detected_images', name))}
    return render_template('index.html', detected_ips=detected_ips)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capture_running
    capture_running = False
    return '', 200


@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_running
    capture_running = True
    thread = Thread(target=gen_frames)
    thread.start()
    return '', 200


if __name__ == '__main__':
    #socketio.run(app, debug=True)
	socketio.run(app, host='0.0.0.0', port=5000)

