from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import torch
import pytesseract
from PIL import Image
import numpy as np
import re
import os

# PATH 설정
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
capture = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
desired_width = 1920
desired_height = 1080
capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

def GenerateFrames():
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        else:
            ref, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()  
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/stream')
def Stream():
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET'])
def Capture():
    ret, frame = capture.read()
    file_name = f'live_stream\\images\\screenshot.jpg'
    cv2.imwrite(file_name, frame)
    
    # Return a success response
    return 'Screenshot captured and saved ', 200

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_path = data['image_path']
    img = Image.open(image_path)

    results = model(img)
    
    if results.ims[0] is not None:
        save_dir = pathlib.Path("live_stream\\saved_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(results.xyxy[0]):  
            original_image = Image.fromarray(results.ims[0])
            xmin, ymin, xmax, ymax = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            crop_img = original_image.crop((xmin, ymin, xmax, ymax))
            crop_img.save(save_dir / f"crop_{i}.jpg")

    results.render()
    print(results)
    
    if results.ims[0] is not None:
        final_image = Image.fromarray(results.ims[0])
        final_image = final_image.resize((640, 480), Image.LANCZOS)
        final_image.save("live_stream\\results\\detected.jpg", format="JPEG")

    return send_file("results\\detected.jpg", mimetype='image/jpeg')

@app.route('/ocr', methods=['POST'])
def ocr():
    directory_path = "live_stream/saved_images"
    texts = []
    
    files = os.listdir(directory_path)
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    for filename in files:
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory_path, filename)
            image = np.array(Image.open(img_path))
            image = cv2.resize(image, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # tesseract-ocr
            config = r'-c tessedit_char_whitelist=0123456789+-=()., -l kor+eng --oem 3 --psm 11'
            text = pytesseract.image_to_string(binary, config=config)
            cleaned_text = re.sub(r'\s+', ' ', text)
            match = re.search(r'(\d+\.\d+)\s+(\d{4}-\d+)', cleaned_text)
            if match:
                final_text = f"{match.group(1)} {match.group(2)}"
                print(f'{filename} : {final_text}')
                texts.append(f'{filename} : {final_text}')
            else:
                print(f'{filename} : No valid data found')
                texts.append(f'{filename} : No valid data found')
                
    with open('live_stream/saved_texts.txt', 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text + '\n')
            
    return jsonify({"texts": texts})

def extract_year(volume_str):
    """
    볼륨 번호 문자열에서 연도 부분을 추출합니다.
    """
    return int(volume_str.split('-')[0])

def extract_index(volume_str):
    """
    볼륨 번호 문자열에서 인덱스 번호 부분을 추출합니다.
    """
    return int(volume_str.split('-')[1])

def is_sorted(call_number_objects):
    """
    호출 번호, 볼륨 번호 연도, 볼륨 번호 인덱스를 기준으로 정렬합니다.
    """
    # 키 함수 정의
    def sort_key(obj):
        call_number, volume_number = obj
        return (call_number, extract_year(volume_number), extract_index(volume_number))

    sorted_objects = sorted(call_number_objects, key=sort_key)
    print(sorted_objects)
    return call_number_objects == sorted_objects

@app.route('/check_placement', methods=['POST'])
def check_placement():
    print('.')
    with open('live_stream/saved_texts.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = [line.strip() for line in lines if line.strip()]
    book_data = [text for text in texts if "No valid data found" not in text]
        
    book_tuples = []
    for data in book_data:
        parts = data.split(':')
        classification_author = parts[1].strip()
        classification, author_code = classification_author.split()
        book_tuples.append((classification, author_code))
    
    is_correct = is_sorted(book_tuples)
    print(is_correct)
    return jsonify({"is_correct": is_correct})

if __name__ == "__main__":
    model_path = 'C:\\vscode-workspace\\live_stream\\weight\\best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # 모델 로드
    app.run(host="127.0.0.1", port="8080")