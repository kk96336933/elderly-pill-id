import os
import sys
import json
import time
import queue
import threading
import math
import cv2
import numpy as np
import csv
import base64
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv
import google.generativeai as genai
from pyzbar import pyzbar
import smtplib
from email.mime.text import MIMEText
from email.header import Header

load_dotenv()
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        genai.configure(api_key=api_key)
        # 依照使用者要求，使用 Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        HAS_GEMINI = True
        print("[INFO] Gemini API 設定成功。")
    else:
        HAS_GEMINI = False
        print("[WARNING] 尚未設定有效的 GEMINI_API_KEY。")
except Exception as e:
    HAS_GEMINI = False
    print(f"[ERROR] Gemini API 設定失敗: {e}")

# Email notification configuration (defaults loaded from .env)
EMAIL_CONFIG = {
    "sender_email": os.environ.get("SENDER_EMAIL", ""),
    "sender_password": os.environ.get("SENDER_PASSWORD", ""),
    "receiver_email": os.environ.get("RECEIVER_EMAIL", ""),
    "auto_send": os.environ.get("AUTO_SEND_EMAIL", "False").lower() == "true",
    "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.environ.get("SMTP_PORT", "465"))
}

def send_notification_email(product_info, receiver=None):
    sender = EMAIL_CONFIG["sender_email"]
    password = EMAIL_CONFIG["sender_password"]
    if not receiver:
        receiver = EMAIL_CONFIG["receiver_email"]
    
    if not sender or not password or not receiver:
        print("[WARNING] 郵件伺服器、帳密或收件者信箱未設定，略過郵件發送。")
        return False
        
    try:
        name = product_info.get("name", "未知藥品")
        brand = product_info.get("brand", "未知品牌")
        category = product_info.get("category", "未指定分類")
        dosage = product_info.get("dosage", "依包裝說明")
        time_str = product_info.get("time", "無特定時間")
        
        # Format warnings array to string list
        raw_warnings = product_info.get("warnings", ["無特殊警告"])
        if isinstance(raw_warnings, list):
            warnings_text = "\n".join([f"  * {w}" for w in raw_warnings])
        else:
            warnings_text = f"  * {raw_warnings}"
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        subject = f"【長輩用藥通知】已於 {timestamp} 服用 {name}"
        body = f"""您好：

您的家人已於 {timestamp} 掃描並確認服用以下藥品/保健食品：

* 藥品名稱：{brand} {name}
* 藥物分類：{category}
* 服用劑量：{dosage}
* 服用時間：{time_str}
* 用藥警告與提醒：
{warnings_text}

---
本郵件由「銀髮族藥袋藥品安全監控系統」自動發送。請關心家人的健康！
"""
        message = MIMEText(body, 'plain', 'utf-8')
        message['From'] = Header("銀髮族用藥監控系統", 'utf-8')
        message['To'] = Header(receiver, 'utf-8')
        message['Subject'] = Header(subject, 'utf-8')
        
        # Connect to SMTP server (SSL)
        server = smtplib.SMTP_SSL(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"], timeout=10)
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
        server.quit()
        print(f"[EMAIL] 用藥通知已發送至 {receiver}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] 郵件發送失敗: {e}")
        return False

app = Flask(__name__)

# Global state
clients = []
clients_lock = threading.Lock()
CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), 'history.csv')

IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'static', 'history_images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# Push event to all connected SSE clients
def broadcast_result(info, fps=0.0, inference_time=0.0):
    info['fps'] = round(fps, 1)
    info['inference_time_ms'] = round(inference_time, 1)
    
    image_filename = info.get('image_filename', 'N/A')
    
    # Save to CSV
    try:
        file_exists = os.path.isfile(CSV_LOG_PATH)
        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Timestamp', 'Image File', 'Product Name', 'Category', 'FPS', 'Inference Time (ms)'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, image_filename, info.get('name', ''), info.get('category', ''), info['fps'], info['inference_time_ms']])
    except Exception as e:
        print(f"[ERROR] 無法寫入 CSV: {e}")

    # 移除自動發送郵件通知家人 (改由前端調用發送，防信箱衝突)
    pass

    with clients_lock:
        for q in clients:
            q.put(info)
    print(f"[BROADCAST] 辨識完成並廣播: {info.get('name')} - 耗時: {info['inference_time_ms']}ms")

# Camera class managing background frame reading
class VideoCamera:
    def __init__(self):
        camera_url = os.environ.get("CAMERA_URL")
        self.is_mock = False
        
        if camera_url:
            print(f"[INFO] 嘗試連接網路攝影機: {camera_url}")
            self.cap = cv2.VideoCapture(camera_url)
        else:
            # Try index 0, fallback to 1 if needed
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        
        if not self.cap.isOpened():
            print("[WARNING] 無法開啟攝影機鏡頭。將開啟「模擬虛擬鏡頭畫面」模式。")
            self.is_mock = True
        else:
            print("[INFO] 攝影機鏡頭開啟成功。")
            # 針對本機 USB 鏡頭設定解析度 (網路串流通常由發送端決定)
            if not camera_url:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        self.frame = None
        self.running = True
        
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
        self.last_inference_time_ms = 0.0
        
        self.scan_interval = 5.0
        self.last_scan_time = 0
        
        # QR Code / 一維條碼掃描變數
        self.raw_frame = None
        self.last_barcode_data = None
        self.barcode_cooldown = 4.0
        
        # Start background loop thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            if self.is_mock:
                # Generate a premium look synthetic frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Draw grid background
                for x in range(0, 640, 40):
                    cv2.line(frame, (x, 0), (x, 480), (15, 15, 30), 1)
                for y in range(0, 480, 40):
                    cv2.line(frame, (0, y), (640, y), (15, 15, 30), 1)
                    
                # Animated circular element representing scanner focal point
                t = time.time()
                cx = int(320 + 100 * math.cos(t * 1.5))
                cy = int(240 + 60 * math.sin(t * 2))
                
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), 2)
                cv2.circle(frame, (320, 240), 180, (40, 40, 80), 1)
                
                # Draw text details
                cv2.putText(frame, "VIRTUAL SCREEN ACTIVE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (99, 102, 241), 2)
                cv2.putText(frame, "No camera detected. App is running in sandbox mode.", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                # Dynamic status pulse
                pulse = int(128 + 127 * math.sin(t * 4))
                cv2.circle(frame, (600, 40), 8, (0, pulse, 0), -1)
                
                self.frame = frame
                time.sleep(0.033)  # ~30 FPS
                continue
                
            # 讀取畫面
            success, img = self.cap.read()
            if not success:
                print("[ERROR] 讀取鏡頭畫面失敗，嘗試重新連線...")
                time.sleep(1.0)
                continue
                
            # 儲存未翻轉的原始影像（用於 AI 辨識與條碼掃描，避免左右相反無法讀取）
            self.raw_frame = img.copy()
            
            # 本地端即時 QR Code / 一維條碼掃描
            try:
                gray = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
                barcodes = pyzbar.decode(gray)
                for barcode in barcodes:
                    barcode_data = barcode.data.decode("utf-8")
                    barcode_type = barcode.type
                    
                    now = time.time()
                    # 避免在冷卻時間內重複觸發同一個條碼
                    if (barcode_data != self.last_barcode_data) or (now - self.last_scan_time > self.barcode_cooldown):
                        self.last_barcode_data = barcode_data
                        self.last_scan_time = now
                        
                        info = {
                            "not_found": False,
                            "is_qrcode": True,
                            "qrcode_data": barcode_data,
                            "qrcode_type": barcode_type,
                            "name": f"已掃描 {barcode_type}",
                            "category": "QR Code / 條碼快速掃描",
                            "brand": "本地解碼",
                            "dosage": barcode_data,
                            "time": "即時讀取",
                            "storage": "無特定"
                        }
                        
                        # 若掃描內容為 JSON 格式，則嘗試解析以填充詳細欄位
                        if barcode_data.strip().startswith('{') and barcode_data.strip().endswith('}'):
                            try:
                                parsed = json.loads(barcode_data)
                                info.update(parsed)
                                info["is_qrcode"] = True
                                info["qrcode_data"] = barcode_data
                            except Exception as json_e:
                                print(f"[JSON Decode Error] QR Code JSON 解析失敗: {json_e}")
                                
                        # 廣播辨識結果給前端
                        broadcast_result(info, self.fps, 0.0)
            except Exception as scan_e:
                pass
                
            # 水平翻轉影像以提供使用者直覺的自拍鏡像流
            img = cv2.flip(img, 1)
            
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
                
            # Draw FPS and Inference Time on frame
            cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if self.last_inference_time_ms > 0:
                cv2.putText(img, f"Last Infer: {self.last_inference_time_ms:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            self.frame = img
            time.sleep(0.01)



    def get_jpeg_frame(self):
        if self.frame is None:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', black_frame)
            return jpeg.tobytes()
            
        ret, jpeg = cv2.imencode('.jpg', self.frame)
        if not ret:
            return b''
        return jpeg.tobytes()

    def close(self):
        self.running = False
        if not self.is_mock:
            self.cap.release()

# Global camera instance
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(cam):
    while True:
        frame_bytes = cam.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    try:
        cam = get_camera()
        return Response(gen_frames(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error serving video feed: {e}")
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, jpeg = cv2.imencode('.jpg', black_frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/api/events')
def sse_events():
    def event_generator():
        q = queue.Queue()
        with clients_lock:
            clients.append(q)
        print(f"[SSE] 新增客戶端連線，目前連線數: {len(clients)}")
        try:
            while True:
                try:
                    data = q.get(timeout=20.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            with clients_lock:
                if q in clients:
                    clients.remove(q)
            print(f"[SSE] 客戶端中斷連線，目前連線數: {len(clients)}")
            
    return Response(event_generator(), mimetype='text/event-stream')

@app.route('/api/settings/camera', methods=['POST'])
def set_camera():
    data = request.json
    url = data.get('url', '')
    
    global camera
    if camera:
        camera.close()
        
    if url.strip() == '':
        os.environ.pop("CAMERA_URL", None)
    else:
        os.environ["CAMERA_URL"] = url.strip()
        
    camera = VideoCamera()
    return jsonify({"status": "success"})

@app.route('/api/gemini_scan', methods=['POST'])
def gemini_scan():
    if not HAS_GEMINI:
        return jsonify({"error": "Gemini API 尚未設定。請檢查 .env 檔案中的 GEMINI_API_KEY。"}), 400
        
    cam = get_camera()
    
    # 決定送給 Gemini 的影像：
    # 如果是模擬虛擬鏡頭，直接用 cam.get_jpeg_frame()。如果是實體鏡頭，使用未水平翻轉的 cam.raw_frame 確保文字方向正常。
    if cam.is_mock:
        frame_bytes = cam.get_jpeg_frame()
    else:
        if cam.raw_frame is None:
            return jsonify({"error": "無法擷取鏡頭影像"}), 500
        ret, jpeg = cv2.imencode('.jpg', cam.raw_frame)
        if not ret:
            return jsonify({"error": "影像編碼失敗"}), 500
        frame_bytes = jpeg.tobytes()
        
    if not frame_bytes:
        return jsonify({"error": "無法擷取鏡頭影像"}), 500

    t0 = time.time()
    
    # 建立圖片檔名與路徑
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_filename = f"IMG_{timestamp_str}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    # 存入本地小型資料庫資料夾
    try:
        with open(image_path, 'wb') as f:
            f.write(frame_bytes)
    except Exception as e:
        print(f"[ERROR] 無法儲存圖片到 {image_path}: {e}")

    image_url = f"/static/history_images/{image_filename}"
    
    try:
        image_parts = [{"mime_type": "image/jpeg", "data": frame_bytes}]
        
        prompt = """
        你是一個專門協助銀髮族辨識藥品、醫院藥袋、保健食品與用藥安全的 AI 助理。
        請仔細分析這張圖片中的物件，尤其是「醫院藥袋上的列印文字」、「藥盒外包裝」、「藥瓶標籤」、「成分列表」與「所有印刷字體」，並盡力萃取出用藥指引。
        只要畫面中有類似藥袋、藥品紙盒、藥瓶、藥包或保健品，就請嘗試辨識上面的文字，不要輕易判斷為找不到。
        只有在畫面完全空白、全黑或完全沒有任何相關物品時，才將 not_found 設為 true。
        
        請務必且只能回傳一個 JSON 格式的字串，包含以下欄位：
        {
            "not_found": false,
            "barcode": "N/A",
            "name": "商品名稱或藥品名稱 (例如: 普拿疼 Acetaminophen、深海魚油、大正微粒，若無具體名稱請依包裝大字進行分析)",
            "brand": "品牌名稱或醫療機構名稱 (例如: 台北榮總、健康優選，若無則填 未知)",
            "dosage": "建議劑量或用法 (例如: 每次1包，每日三次，若無則填 依包裝說明或醫囑)",
            "time": "建議服用時間 (例如: 飯後服用、睡前服用，若無則填 無特定時間)",
            "warnings": ["警告事項1", "警告事項2"], // 針對該成分給予銀髮族的通用用藥安全提示（如：可能會嗜睡請勿開車、請勿併用阿斯匹靈等）
            "category": "分類 (例如: 感冒退燒藥、心血管保健、關節保健)",
            "storage": "保存方式 (例如: 室溫避光保存、置於陰涼乾燥處)"
        }
        """
        
        response = model.generate_content([prompt, image_parts[0]])
        text_resp = response.text
        
        # 嘗試從回應中萃取 JSON
        start_idx = text_resp.find('{')
        end_idx = text_resp.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text_resp[start_idx:end_idx+1]
            info = json.loads(json_str)
            info['image_url'] = image_url
            info['image_filename'] = image_filename
            info['time_str'] = datetime.now().strftime('%H:%M:%S')
            
            # 將完整的辨識結果存入小型資料庫 (JSON 格式)
            json_filename = f"DATA_{timestamp_str}.json"
            json_path = os.path.join(IMAGE_DIR, json_filename)
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"[ERROR] 無法儲存 JSON 到 {json_path}: {e}")
        else:
            raise ValueError("回傳格式非 JSON")
            
        inference_time_ms = (time.time() - t0) * 1000
        cam.last_inference_time_ms = inference_time_ms
        
        # 廣播給所有前端更新介面
        broadcast_result(info, cam.fps, inference_time_ms)
        
        return jsonify({"status": "success", "data": info})
        
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return jsonify({"error": f"辨識失敗: {str(e)}"}), 500

def decode_base64_image(base64_data):
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img, img_data
    except Exception as e:
        print(f"[ERROR] Base64 解碼失敗: {e}")
        return None, None

@app.route('/api/scan_barcode_frame', methods=['POST'])
def scan_barcode_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "缺少影像資料"}), 400
        
    base64_image = data['image']
    img, _ = decode_base64_image(base64_image)
    if img is None:
        return jsonify({"error": "影像解碼失敗"}), 400
        
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            
            info = {
                "not_found": False,
                "is_qrcode": True,
                "qrcode_data": barcode_data,
                "qrcode_type": barcode_type,
                "name": f"已掃描 {barcode_type}",
                "category": "QR Code / 條碼快速掃描",
                "brand": "本地解碼",
                "dosage": barcode_data,
                "time": "即時讀取",
                "storage": "無特定"
            }
            
            if barcode_data.strip().startswith('{') and barcode_data.strip().endswith('}'):
                try:
                    parsed = json.loads(barcode_data)
                    info.update(parsed)
                    info["is_qrcode"] = True
                    info["qrcode_data"] = barcode_data
                except Exception as json_e:
                    print(f"[JSON Decode Error] QR Code JSON 解析失敗: {json_e}")
            
            # 廣播辨識結果給前端 (SSE)
            broadcast_result(info, 0.0, 0.0)
            return jsonify({"status": "success", "barcode_found": True, "data": info})
            
        return jsonify({"status": "success", "barcode_found": False})
    except Exception as e:
        print(f"[Scan Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/gemini_scan_frame', methods=['POST'])
def gemini_scan_frame():
    if not HAS_GEMINI:
        return jsonify({"error": "Gemini API 尚未設定。請檢查 .env 檔案中的 GEMINI_API_KEY。"}), 400
        
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "缺少影像資料"}), 400
        
    base64_image = data['image']
    img, frame_bytes = decode_base64_image(base64_image)
    if img is None or frame_bytes is None:
        return jsonify({"error": "影像解碼失敗"}), 400
        
    t0 = time.time()
    
    # 建立圖片檔名與路徑
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_filename = f"IMG_{timestamp_str}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    
    # 存入本地小型資料庫資料夾
    try:
        with open(image_path, 'wb') as f:
            f.write(frame_bytes)
    except Exception as e:
        print(f"[ERROR] 無法儲存圖片到 {image_path}: {e}")

    image_url = f"/static/history_images/{image_filename}"
    
    try:
        image_parts = [{"mime_type": "image/jpeg", "data": frame_bytes}]
        
        prompt = """
        你是一個專門協助銀髮族辨識藥品、醫院藥袋、保健食品與用藥安全的 AI 助理。
        請仔長分析這張圖片中的物件，尤其是「醫院藥袋上的列印文字」、「藥盒外包裝」、「藥瓶標籤」、「成分列表」與「所有印刷字體」，並盡力萃取出用藥指引。
        只要畫面中有類似藥袋、藥品紙盒、藥瓶、藥包或保健品，就請嘗試辨識上面的文字，不要輕易判斷為找不到。
        只有在畫面完全空白、全黑或完全沒有任何相關物品時，才將 not_found 設為 true。
        
        請務必且只能回傳一個 JSON 格式的字串，包含以下欄位：
        {
            "not_found": false,
            "barcode": "N/A",
            "name": "商品名稱或藥品名稱 (例如: 普拿疼 Acetaminophen、深海魚油、大正微粒，若無具體名稱請依包裝大字進行分析)",
            "brand": "品牌名稱或醫療機構名稱 (例如: 台北榮總、健康優選，若無則填 未知)",
            "dosage": "建議劑量或用法 (例如: 每次1包，每日三次，若無則填 依包裝說明或醫囑)",
            "time": "建議服用時間 (例如: 飯後服用、睡前服用，若無則填 無特定時間)",
            "warnings": ["警告事項1", "警告事項2"], // 針對該成分給予銀髮族的通用用藥安全提示（如：可能會嗜睡請勿開車、請勿併用阿斯匹靈等）
            "category": "分類 (例如: 感冒退燒藥、心血管保健、關節保健)",
            "storage": "保存方式 (例如: 室溫避光保存、置於陰涼乾燥處)"
        }
        """
        
        response = model.generate_content([prompt, image_parts[0]])
        text_resp = response.text
        
        # 嘗試從回應中萃取 JSON
        start_idx = text_resp.find('{')
        end_idx = text_resp.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text_resp[start_idx:end_idx+1]
            info = json.loads(json_str)
            info['image_url'] = image_url
            info['image_filename'] = image_filename
            info['time_str'] = datetime.now().strftime('%H:%M:%S')
            
            # 將完整的辨識結果存入小型資料庫 (JSON 格式)
            json_filename = f"DATA_{timestamp_str}.json"
            json_path = os.path.join(IMAGE_DIR, json_filename)
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"[ERROR] 無法儲存 JSON 到 {json_path}: {e}")
        else:
            raise ValueError("回傳格式非 JSON")
            
        inference_time_ms = (time.time() - t0) * 1000
        
        # 廣播給所有前端更新介面
        broadcast_result(info, 0.0, inference_time_ms)
        
        return jsonify({"status": "success", "data": info})
        
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return jsonify({"error": f"辨識失敗: {str(e)}"}), 500

@app.route('/api/history')
def get_history():
    history_data = []
    try:
        files = os.listdir(IMAGE_DIR)
        # 讀取 DATA_ 開頭的 json 檔案，依照時間排序 (最新的在前面)
        json_files = sorted([f for f in files if f.endswith('.json') and f.startswith('DATA_')], reverse=True)[:10]
        # 反轉順序，讓最舊的先被前端讀取，最新的最後插入 (這樣前端表格最上面才是最新的)
        json_files.reverse()
        for jf in json_files:
            with open(os.path.join(IMAGE_DIR, jf), 'r', encoding='utf-8') as f:
                history_data.append(json.load(f))
    except Exception as e:
        print(f"[ERROR] 讀取歷史資料庫失敗: {e}")
    return jsonify(history_data)

@app.route('/api/settings/email', methods=['GET', 'POST'])
def email_settings():
    if request.method == 'GET':
        return jsonify({
            "receiver_email": EMAIL_CONFIG["receiver_email"],
            "auto_send": EMAIL_CONFIG["auto_send"],
            "has_credentials": bool(EMAIL_CONFIG["sender_email"] and EMAIL_CONFIG["sender_password"])
        })
    elif request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({"error": "無效的請求"}), 400
        
        EMAIL_CONFIG["receiver_email"] = data.get("receiver_email", "").strip()
        EMAIL_CONFIG["auto_send"] = bool(data.get("auto_send", False))
        
        print(f"[SETTINGS] 郵件收件人更新為: {EMAIL_CONFIG['receiver_email']} - 自動發送: {EMAIL_CONFIG['auto_send']}")
        return jsonify({"status": "success"})

@app.route('/api/send_email', methods=['POST'])
def trigger_email():
    data = request.json
    if not data:
        return jsonify({"error": "請提供藥物辨識資訊"}), 400
        
    # Check if SMTP configuration exists
    if not EMAIL_CONFIG["sender_email"] or not EMAIL_CONFIG["sender_password"]:
        return jsonify({"error": "郵件伺服器未設定。請在 Windows 端的 .env 檔案中配置 SENDER_EMAIL 與 SENDER_PASSWORD。"}), 400
        
    # 優先使用前端傳入的特定收信人
    receiver = data.get("receiver_email") or EMAIL_CONFIG["receiver_email"]
    if not receiver:
        return jsonify({"error": "請先設定家人的收件信箱！"}), 400
        
    success = send_notification_email(data, receiver)
    if success:
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "信件發送失敗，請確認 SMTP 設定、網路連線或是否使用了正確的 App 密碼。"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
