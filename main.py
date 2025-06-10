# main.py

import cv2
import torch
import numpy as np
from flask import Flask, Response, request, render_template, redirect, url_for, send_file
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True).to(device)
midas.eval()
transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform


def estimate_depth(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def depth_to_image(depth_map):
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype('uint8'), cv2.COLORMAP_MAGMA)
    return depth_colored


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/estimate_image', methods=['POST'])
def estimate_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(img_bgr, (320, 240))
    depth_map = estimate_depth(resized_img)
    depth_image = depth_to_image(depth_map)

    combined = cv2.hconcat([resized_img, depth_image])

    _, buffer = cv2.imencode('.jpg', combined)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')

def generate_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (320, 240))
        depth_map = estimate_depth(resized)
        depth_img = depth_to_image(depth_map)
        combined = cv2.hconcat([resized, depth_img])

        _, jpeg = cv2.imencode('.jpg', combined)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return "No frame uploaded", 400
    
    file = request.files['frame']
    in_memory_file = BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is None:
        return "Invalid image", 400

    resized = cv2.resize(img, (320, 240))
    depth_map = estimate_depth(resized)
    depth_img = depth_to_image(depth_map)

    _, buffer = cv2.imencode('.jpg', depth_img)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')


@app.route('/live_feed')
def live_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
