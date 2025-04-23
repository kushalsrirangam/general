# Smart Hat Backend Server with ngrok Integration
# Updated to support modular JS/CSS and static file serving

from flask import Flask, request, jsonify, redirect, render_template_string, Response, send_from_directory
import subprocess, os, json, threading, cv2, numpy as np, time, lgpio, psutil, shutil, requests, socket
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from datetime import datetime
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from flask_socketio import SocketIO
import firebase_admin
from firebase_admin import credentials, firestore, storage
import zipfile
from flask import send_file
import getpass
from collections import deque
print("🧠 Running as user:", getpass.getuser())

print("[SERVER] Starting as user:", getpass.getuser())

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('/home/ada/de/smartaid-6c5c0-firebase-adminsdk-fbsvc-cee03b08da.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smartaid-6c5c0-default-rtdb.firebaseio.com/',
        'storageBucket': 'smartaid-6c5c0.appspot.com'
    })

# Flask app setup
app = Flask(__name__, static_folder="/home/ada/de/app_server/web_app")
app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["DEBUG"] = True
socketio = SocketIO(app, cors_allowed_origins="*")

db = firestore.client()

frame_lock = threading.Lock()

# Global config
health_status = "OK"
detection_active = True
config_data = {"indoor_mode": False}
LABEL_PATH = "/home/ada/de/coco_labels.txt"
MODEL_PATH = "/home/ada/de/mobilenet_v2.tflite"
CONFIG_FILE = "/home/ada/de/detection/config.json"
voice_alert_enabled = True
normalSize = (2028, 1520)
lowresSize = (300, 300)
latest_frame = None
indoor_mode = False
logging_paused = False  # ✅ Define it once here, no need for global outside
ultrasonic_voice_enabled = True  # ✅ Prevent undefined var

# --- Static file and UI routes ---
@app.route('/')
def serve_index():
    return redirect('/control_panel')

@app.route('/control_panel')
def serve_control_panel():
    return send_from_directory(app.static_folder, 'control_panel.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# 🔧 PATCH: Unified config endpoint
@app.route("/get_config", methods=["GET"])
def get_config():
    mode_labels = config_data.get("filter_classes", {})
    is_custom = isinstance(mode_labels, list) or (
        isinstance(mode_labels, dict) and "indoor" in mode_labels and "outdoor" in mode_labels and
        config_data.get("indoor_mode", False) is False and
        config_data.get("filter_classes").get("indoor") != config_data.get("filter_classes")
    )

    return jsonify({
        **config_data,
        "custom_mode": is_custom
    })

# 🔧 PATCH: Quiet mode added to status
@app.route("/status")
def get_status():
    battery = psutil.sensors_battery()
    return jsonify({
        "battery": battery.percent if battery else -1,
        "health": health_status,
        "detection_active": detection_active,
        "quiet_mode_enabled": config_data.get("quiet_mode_enabled", False),
        "mode": "indoor" if config_data.get("indoor_mode") else "outdoor"
    })

# Create Dash app under /analytics
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/analytics/',
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY]
)
dash_app.title = "Smart Hat Analytics"

# Helper: Downsample and clean timestamps
def clean_and_downsample(df, value_cols=None, interval='1min'):
    if df.empty or 'timestamp' not in df.columns:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    if value_cols:
        return df.groupby(pd.Grouper(key='timestamp', freq=interval))[value_cols].mean(numeric_only=True).reset_index()
    else:
        return df.groupby(pd.Grouper(key='timestamp', freq=interval)).mean(numeric_only=True).reset_index()

# Firebase fetch helpers
def fetch_data(collection_name):
    try:
        docs = [doc.to_dict() for doc in db.collection(collection_name).stream()]
        return pd.DataFrame(docs)
    except:
        return pd.DataFrame()

def fetch_system_health():
    return fetch_data("system_health_logs")

def fetch_battery():
    return fetch_data("battery_logs")

def fetch_detection(full=False):
    df = fetch_data("detection_logs")
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['detection_count'] = 1
    return df if full else df.groupby(pd.Grouper(key='timestamp', freq='1min')).sum(numeric_only=True).reset_index()

def fetch_voice_logs():
    return fetch_data("voice_logs")

def fetch_video_logs():
    return fetch_data("video_logs")

def fetch_location_logs():
    return fetch_data("location_logs")

def fetch_motion_logs():
    return fetch_data("motion_logs")

def fetch_ultrasonic_logs():
    return fetch_data("ultrasonic_logs")

# Graph generator callback builder
def generate_graph_callback(fetch_func, graph_id, y_columns=None, title="", kind="line", interval_id='fast-refresh'):
    @dash_app.callback(Output(graph_id, 'figure'), Input(interval_id, 'n_intervals'))
    def update_graph(n):
        df = fetch_func()
        df = clean_and_downsample(df, value_cols=y_columns)
        if not df.empty and 'timestamp' in df.columns:
            cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=20)
            df = df[df['timestamp'] >= cutoff]

        if df.empty:
            return px.line(title=f"{title} (No Data)")

        if kind == "bar":
            fig = px.bar(df, x='timestamp', y=y_columns, title=title)
        else:
            fig = px.line(df, x='timestamp', y=y_columns, title=title)
        fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40), xaxis_title="Time")
        return fig
    return update_graph

# Dash Layout with Categories
dash_app.layout = dbc.Container([
    html.H2("Smart Hat Analytics", className="text-center my-4"),
    dcc.Interval(id='fast-refresh', interval=10*1000, n_intervals=0),
    dcc.Interval(id='slow-refresh', interval=30*1000, n_intervals=0),

    html.H4("🔬 System Performance", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='battery-graph'), lg=6),
        dbc.Col(dcc.Graph(id='cpu-graph'), lg=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='fps-graph'), lg=6),
        dbc.Col(dcc.Graph(id='latency-graph'), lg=6),
    ]),

    html.H4("🎯 Detection Accuracy & Behavior", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='detection-log-graph'), lg=6),
        dbc.Col(dcc.Graph(id='label-frequency-graph'), lg=6),
    ]),

    html.H4("🤖 Smart System Behavior", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='voice-command-graph'), lg=6),
        dbc.Col(dcc.Graph(id='video-trigger-graph'), lg=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='motion-graph'), lg=12),
    ]),

    html.H4("📡 Ultrasonic Sensor Insights", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='ultra-graph'), lg=12),
    ]),

    html.H4("👨‍🦯 User-Centric Insights", className="my-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='distance-walked-graph'), lg=12)
    ])
])


# Register callbacks
# Graph registration
generate_graph_callback(fetch_battery, 'battery-graph', ['battery_percentage'], "Battery Over Time", interval_id='slow-refresh')()
generate_graph_callback(fetch_system_health, 'cpu-graph', ['cpu'], "CPU Usage Over Time", interval_id='slow-refresh')()
generate_graph_callback(lambda: fetch_detection(full=True), 'fps-graph', ['frame_id'], "FPS Over Time", interval_id='fast-refresh')()
generate_graph_callback(lambda: fetch_detection(full=True), 'latency-graph', ['duration_ms'], "Latency (ms) Over Time", interval_id='slow-refresh')()
generate_graph_callback(lambda: fetch_detection(full=False), 'detection-log-graph', ['detection_count'], "Detections Over Time", kind="bar", interval_id='slow-refresh')()
generate_graph_callback(lambda: fetch_detection(full=True), 'label-frequency-graph', ['label'], "Detected Labels", kind="bar", interval_id='slow-refresh')()
generate_graph_callback(fetch_voice_logs, 'voice-command-graph', ['command'], "Voice Commands Over Time", kind="bar", interval_id='slow-refresh')()
generate_graph_callback(fetch_video_logs, 'video-trigger-graph', ['video_url'], "Smart Video Recordings", kind="bar", interval_id='slow-refresh')()
generate_graph_callback(fetch_location_logs, 'distance-walked-graph', ['distance_m'], "Distance Walked (meters)", kind="line", interval_id='slow-refresh')()
generate_graph_callback(fetch_motion_logs, 'motion-graph', ['motion_value'], "Motion Status Over Time", kind="line", interval_id='fast-refresh')()
generate_graph_callback(fetch_ultrasonic_logs, 'ultra-graph', ['readings.Left Front', 'readings.Right Front'], "Ultrasonic Readings (Front)", kind="line", interval_id='fast-refresh')()


# Function to convert logs to CSV
def convert_logs_to_csv(logs):
    filename = "/tmp/logs.csv"  # Path to temporary CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    return filename

# Route to serve the CSV file
@app.route('/download_logs', methods=['GET'])
def download_all_logs():
    try:
        # Collections to export
        collections = [
            'battery_logs',
            'ultrasonic_logs',
            'motion_logs',
            'detection_logs',
            'location_logs',
            'system_health_logs',
            'video_logs'
        ]

        zip_filename = "/tmp/all_logs.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for col in collections:
                docs = [doc.to_dict() for doc in db.collection(col).stream()]
                if not docs:
                    continue
                df = pd.DataFrame(docs)
                csv_path = f"/tmp/{col}.csv"
                df.to_csv(csv_path, index=False)
                zipf.write(csv_path, arcname=f"{col}.csv")

        return send_file(zip_filename, as_attachment=True, download_name="all_logs.zip")

    except Exception as e:
        print("[ERROR] Failed to download all logs:", e)
        return f"Error creating ZIP: {e}", 500
    
@app.route("/set_analytics_section", methods=["POST"])
def set_analytics_section():
    data = request.get_json()
    section = data.get("section", "performance")
    # Optional: validate section
    valid_sections = {"performance", "detection", "interaction", "videos"}
    if section not in valid_sections:
        return jsonify({"error": "Invalid section"}), 400
    # Save to a global, file, or database if needed
    return jsonify({"status": "success", "section": section})
 
    
# Motion status route
@app.route("/motion-status")
def get_motion_status():
    return jsonify({
        "motion_active": motion_active
    })


# --- Voice command to simulate motion ---
@app.route("/voice_command", methods=["POST"])
def voice_command():
    command = request.json.get("command", "").lower()
    if "i am walking" in command or "start walking" in command:
        requests.post("http://localhost:5000/motion", json={"moving": True})
        return jsonify({"status": "triggered", "motion": True})
    elif "i stopped" in command or "stop walking" in command:
        requests.post("http://localhost:5000/motion", json={"moving": False})
        return jsonify({"status": "triggered", "motion": False})
    return jsonify({"status": "ignored"})



# Default config
default_config = {
    "filter_classes": {
        "indoor": [
            "person", "chair", "couch", "tv", "laptop", "remote", "cell phone",
            "microwave", "refrigerator", "bed", "dining table", "book", "toaster", "sink"
        ],
        "outdoor": [
            "person", "car", "bus", "bicycle", "motorcycle", "truck", "traffic light", "stop sign",
            "dog", "cat", "bench", "parking meter", "train", "sheep", "cow", "horse"
        ]
    },
    "logging": True,
    "quiet_mode_enabled": False,
    "indoor_mode": False,
    "ultrasonic_thresholds": {
        "Left Front": 70,
        "Left Middle": 70,
        "Left Rear": 70,
        "Right Front": 70,
        "Right Middle": 70,
        "Right Rear": 70
    }
}




SENSORS = {
    "Left Front":  {"trigger": 4,  "echo": 17},
    "Left Middle": {"trigger": 27, "echo": 22},
    "Left Rear":   {"trigger": 23, "echo": 24},
    "Right Front": {"trigger": 5,  "echo": 6},
    "Right Middle": {"trigger": 12, "echo": 13},
    "Right Rear":   {"trigger": 19, "echo": 26}
}

CHIP = 4
ultrasonic_readings = {}
motion_active = False  # Track motion status
last_ultra_speak_time = {}

# --- Utility Functions ---
def read_label_file(path):
    with open(path, 'r') as f:
        return {int(line.split()[0]): line.strip().split(maxsplit=1)[1] for line in f}

def push_message_to_clients(message):
    socketio.emit('speak', {'message': message})

def measure_distance(h, trig, echo, timeout=0.02):
    lgpio.gpio_write(h, trig, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, trig, 0)
    start = time.time()
    timeout_start = time.time()
    while lgpio.gpio_read(h, echo) == 0:
        start = time.time()
        if time.time() - timeout_start > timeout:
            return "No Echo"
    timeout_start = time.time()
    while lgpio.gpio_read(h, echo) == 1:
        stop = time.time()
        if time.time() - timeout_start > timeout:
            return "Echo Timeout"
    elapsed = stop - start
    distance = (elapsed * 34300) / 2
    return round(distance, 2) if 2 < distance < 400 else "Out of Range"

ultra_history = {name: deque(maxlen=3) for name in SENSORS}
detection_timers = {}
baseline_distances = {}  # Track average distance per sensor

def ultrasonic_loop():
    global logging_paused, health_status, ultrasonic_readings
    h = None
    try:
        h = lgpio.gpiochip_open(4)
        for s in SENSORS.values():
            try:
                lgpio.gpio_free(h, s["trigger"])
                lgpio.gpio_free(h, s["echo"])
            except:
                pass
            lgpio.gpio_claim_output(h, s["trigger"])
            lgpio.gpio_claim_input(h, s["echo"])

        while True:
            if logging_paused:
                print("[ULTRASONIC] Skipping log due to paused flag")
                time.sleep(1)
                continue

            now = time.time()
            failed = []
            readings = {}
            successful_readings = 0

            for name, pin in SENSORS.items():
                dist = measure_distance(h, pin["trigger"], pin["echo"])
                threshold = config_data.get("ultrasonic_thresholds", {}).get(name, 100)

                if isinstance(dist, (int, float)):
                    ultra_history[name].append(dist)

                    if len(ultra_history[name]) == 3:
                        diffs = [abs(ultra_history[name][i] - ultra_history[name][i - 1]) for i in range(1, 3)]
                        if max(diffs) < 5:
                            smoothed = sum(ultra_history[name]) / 3
                            readings[name] = round(smoothed, 2)
                            successful_readings += 1

                            # Save baseline (if no alert)
                            if name not in baseline_distances:
                                baseline_distances[name] = smoothed
                            else:
                                baseline_distances[name] = 0.9 * baseline_distances[name] + 0.1 * smoothed

                        else:
                            readings[name] = None
                            detection_timers.pop(name, None)
                    else:
                        readings[name] = None
                        detection_timers.pop(name, None)
                else:
                    readings[name] = None
                    failed.append(name)
                    detection_timers.pop(name, None)

            # HALLWAY LOGIC
            left_close = any(name.startswith("Left") and dist and dist < 80 for name, dist in readings.items())
            right_close = any(name.startswith("Right") and dist and dist < 80 for name, dist in readings.items())

            # Speak only if one side is unusually close (new obstacle)
            if ultrasonic_voice_enabled and voice_alert_enabled and not config_data.get("indoor_mode", False):
                for name, dist in readings.items():
                    if dist and dist < threshold:
                        baseline = baseline_distances.get(name, 100)
                        side = "left" if "Left" in name else "right"

                        if left_close and right_close:
                            continue  # hallway — suppress speaking

                        if dist < baseline * 0.6:  # sudden close object
                            if name not in detection_timers:
                                detection_timers[name] = now
                            elif now - detection_timers[name] > 2 and now - last_ultra_speak_time.get(name, 0) > 4:
                                push_message_to_clients(f"Obstacle on {side} at {int(dist)} cm")
                                last_ultra_speak_time[name] = now
                        else:
                            detection_timers.pop(name, None)
                    else:
                        detection_timers.pop(name, None)

            if successful_readings == 0:
                print("[SKIP] All ultrasonic sensors failed — not logging this cycle.")
                health_status = "All sensors unresponsive"
                if ultrasonic_voice_enabled and voice_alert_enabled and not config_data.get("indoor_mode", False) and now - last_ultra_speak_time.get("all_failed", 0) > 10:
                    push_message_to_clients("All ultrasonic sensors are offline. Please check connections.")
                    last_ultra_speak_time["all_failed"] = now
                time.sleep(1)
                continue

            ultrasonic_readings = readings
            db.collection('ultrasonic_logs').add({
                'timestamp': int(time.time() * 1000),
                'readings': readings,
                'faults': failed
            })
            health_status = "OK" if not failed else f"Sensor fault: {', '.join(failed)}"
            time.sleep(1)

    except Exception as e:
        print("[Ultrasonic Error]", e)
    finally:
        if h is not None:
            try:
                lgpio.gpiochip_close(h)
            except Exception as e:
                print("[ULTRASONIC] Failed to close gpiochip:", e)



# --- Example for logging with standardized timestamps ---
def battery_monitor():
    warned = False
    while True:
        battery = psutil.sensors_battery()
        percent = battery.percent if battery else 100

        # 🔋 Log to Firestore with standardized timestamp
        db.collection('battery_logs').add({
            'timestamp': int(time.time() * 1000),
            'readable_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'battery_percentage': percent
})
        if percent <= 20 and not warned:
            push_message_to_clients("Battery low. Please charge Smart Hat.")
            warned = True
        if percent > 30:
            warned = False

        time.sleep(60)  # Log every minute

        
def system_metrics_monitor():
    while True:
        # Safely read CPU temperature
        try:
            temps = psutil.sensors_temperatures()
            cpu_temp = 0
            for group in temps.values():
                if group and isinstance(group, list) and hasattr(group[0], 'current'):
                    cpu_temp = group[0].current
                    break

            if cpu_temp == 0:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    cpu_temp = int(f.read()) / 1000.0
        except:
            cpu_temp = 0

        usage = {
            "timestamp": int(time.time() * 1000),
            "readable_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "temperature": round(cpu_temp, 1)
        }

        db.collection("system_health_logs").add(usage)
        time.sleep(60)

def clear_all_logs(keys=None):
    global logging_paused
    logging_paused = True
    print("[LOGGING] Paused during log deletion")

    all_collections = {
        'battery_logs': 'Battery Logs',
        #'ultrasonic_logs': 'Ultrasonic Logs',
        'motion_logs': 'Motion Logs',
        'detection_logs': 'Detection Logs',
        'location_logs': 'Location Logs',
        'system_health_logs': 'System Health Logs',
        'video_logs': 'Video Logs'
    }

    if not keys or keys == ['all']:
        keys = list(all_collections.keys())

    deleted_summary = []

    for col in keys:
        if col not in all_collections:
            print(f"[SKIP] Unknown collection: {col}")
            continue

        docs = db.collection(col).stream()
        deleted = 0
        for doc in docs:
            doc.reference.delete()
            deleted += 1

        print(f"[CLEAR] Deleted {deleted} documents from {col}")
        deleted_summary.append(all_collections[col])

    logging_paused = False
    print("[LOGGING] Resumed after log deletion")
    return deleted_summary
    
    
def get_wifi_status():
    try:
        ssid = subprocess.check_output(["iwgetid", "-r"]).decode().strip()
    except Exception:
        ssid = "Not connected"

    signal_strength = "N/A"
    quality = "Unknown"

    try:
        iwconfig = subprocess.check_output(["iwconfig"]).decode()
        match = re.search(r"Signal level=(-?\d+) dBm", iwconfig)
        if match:
            dbm = int(match.group(1))
            signal_strength = f"{dbm} dBm"

            # Convert dBm to quality
            if dbm >= -55:
                quality = "Excellent"
            elif dbm >= -65:
                quality = "Good"
            elif dbm >= -75:
                quality = "Fair"
            else:
                quality = "Bad"
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    except Exception:
        ip_address = "Unavailable"

    return {
        "ssid": ssid,
        "signal": signal_strength,
        "quality": quality,
        "ip": ip_address
    }



# --- Video Recording and Upload ---
def record_video(picam2, duration_sec=2, fps=30):
    # Define the directory where videos are saved
    video_dir = "/home/ada/de/videos"

    # Ensure the directory exists, if not create it
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Generate filename with a timestamp
    filename = f"{video_dir}/alert_{int(time.time())}.avi"

    # Initialize the video writer with XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, normalSize)

    # Record the video for the specified duration with bounding boxes
    labels = read_label_file(LABEL_PATH)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for _ in range(int(duration_sec * fps)):
        lores = picam2.capture_array("lores")
        frame = picam2.capture_array("main")

        # Run inference
        resized = cv2.resize(lores, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_tensor = np.expand_dims(resized, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                class_id = int(classes[i])
                label = labels.get(class_id, f"id:{class_id}")
                x1, y1 = int(xmin * normalSize[0]), int(ymin * normalSize[1])
                x2, y2 = int(xmax * normalSize[0]), int(ymax * normalSize[1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        time.sleep(1 / fps)

    out.release()

    # Now upload the video to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f"videos/{filename.split('/')[-1]}")
    blob.upload_from_filename(filename)
    blob.make_public()  # Make the file publicly accessible
    print(f"[VIDEO] Uploaded to Firebase Storage: {blob.public_url}")

    # Save metadata in Firestore
    db.collection('video_logs').add({
    'timestamp': int(time.time() * 1000),
    'readable_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'video_url': blob.public_url  # Store public URL to access the video
})

    # Delete the video file from Raspberry Pi after upload
    if os.path.exists(filename):
        os.remove(filename)  # Delete the video file from the Pi
        print(f"[VIDEO] Deleted local video file: {filename}")
    else:
        print(f"[ERROR] Video file not found for deletion: {filename}")

def upload_to_firebase_storage(local_filename, remote_filename):
    bucket = storage.bucket()
    blob = bucket.blob(remote_filename)
    blob.upload_from_filename(local_filename)
    blob.make_public()
    print(f"File uploaded to {blob.public_url}")




def detection_loop():
    global latest_frame, detection_active
    labels = read_label_file(LABEL_PATH)
    frame_counter = 0

    from collections import defaultdict
    import uuid
    import threading

    # Smart Dictionaries
    track_memory = {}
    last_spoken = defaultdict(float)
    last_logged = defaultdict(float)
    recent_tracks = set()
    last_video_time = 0
    global last_detected_summary
    last_detected_summary = ""

    def get_center(x1, y1, x2, y2):
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def compute_danger_score(label, size, pos):
        class_score = danger_scores.get(label, 1)
        size_score = 10 if size > 0.12 else 5 if size > 0.05 else 1
        pos_score = 10 if pos == "center" else 5
        return class_score + size_score + pos_score

    def determine_direction(x1):
        if x1 < normalSize[0] // 3:
            return "to your left"
        elif x1 > 2 * normalSize[0] // 3:
            return "to your right"
        else:
            return "in front of you"

    def record_video_with_frame(frame):
        try:
            filename = f"/home/ada/de/videos/alert_{int(time.time())}.avi"
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, normalSize)
            for _ in range(40):
                out.write(frame)
                time.sleep(0.05)
            out.release()
            print(f"[VIDEO] Saved: {filename}")
        except Exception as e:
            print("[VIDEO ERROR]", e)

    INDOOR_MESSAGES = {
        "person": "Someone is in the room. Move slowly to avoid bumping.",
        "chair": "Chair ahead. Move slightly to the left or right.",
        "couch": "Couch detected. Walk gently around it.",
        "tv": "TV in front. Likely near a wall. Turn slightly.",
        "bed": "Bed nearby. Use it as a guide or move around it.",
        "sink": "Sink ahead. You may be in a bathroom or kitchen.",
        "book": "Bookshelf or table ahead. Be cautious.",
        "laptop": "Laptop detected. Likely on a desk. Avoid impact.",
        "toilet": "Toilet detected. Bathroom likely. Step gently.",
        "refrigerator": "Fridge ahead. You're near a kitchen zone."
    }

    OUTDOOR_MESSAGES = {
        "person": "There is a person in front. Stay alert.",
        "car": "Car detected ahead. Do not cross yet.",
        "bus": "Bus nearby. Move cautiously away from the road.",
        "bicycle": "Bicycle approaching. Step aside safely.",
        "motorcycle": "Motorcycle detected. Wait before proceeding.",
        "train": "Train detected. Stay clear of the tracks.",
        "traffic light": "Traffic light ahead. Await signal.",
        "stop sign": "Stop sign visible. Pause and assess surroundings.",
        "dog": "Dog nearby. Proceed slowly.",
        "bench": "Bench detected. You may rest here.",
        "umbrella": "Umbrella detected. Walk carefully around it."
    }

    danger_scores = {
        "person": 10, "car": 9, "bus": 9, "truck": 9, "motorcycle": 9, "train": 9,
        "bicycle": 8, "horse": 8, "traffic light": 8, "stop sign": 8,
        "dog": 7, "cat": 6, "cow": 6, "sheep": 6, "bench": 6, "zebra": 6,
        "elephant": 5, "giraffe": 5, "bed": 5, "chair": 5, "couch": 5,
        "toilet": 5, "dining table": 5, "tv": 4, "refrigerator": 4,
        "microwave": 4, "oven": 4, "sink": 4, "potted plant": 4, "backpack": 4,
        "umbrella": 4, "handbag": 3, "cell phone": 3, "laptop": 3, "book": 3,
        "bottle": 3, "cup": 3, "fork": 2, "knife": 2, "spoon": 2, "bowl": 2,
        "clock": 2, "remote": 2, "teddy bear": 2, "scissors": 2, "toaster": 2,
        "wine glass": 1, "banana": 1, "pizza": 1, "cake": 1, "sandwich": 1, "apple": 1
    }

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print("[ERROR] Failed to load TFLite model:", e)
        return

    try:
        picam2 = Picamera2()

        # ✅ Resolution mode toggle
        if config_data.get("high_res_mode", True):
            camera_config = picam2.create_preview_configuration(
                main={"size": normalSize, "format": "RGB888"},
                lores={"size": lowresSize, "format": "RGB888"}
            )
        else:
            camera_config = picam2.create_preview_configuration(
                lores={"size": lowresSize, "format": "RGB888"}
            )

        picam2.configure(camera_config)
        picam2.start()
        time.sleep(2)

        while True:
            if not detection_active:
                time.sleep(0.5)
                continue

            now = time.time()
            for obj_id in list(track_memory):
                if now - track_memory[obj_id][2] > 10:
                    del track_memory[obj_id]

            lores = picam2.capture_array("lores")
            frame = picam2.capture_array("main") if config_data.get("high_res_mode", True) else lores
            display_frame = frame.copy()

            resized = cv2.resize(lores, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]

            mode = "indoor" if config_data.get("indoor_mode", False) else "outdoor"
            allowed_labels = config_data.get("filter_classes", ["person"])
            summaries = []

            for i in range(len(scores)):
                if scores[i] < 0.5:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                class_id = int(classes[i])
                label = labels.get(class_id, f"id:{class_id}").lower()
                if label not in allowed_labels:
                    continue

                x1 = max(0, int(xmin * normalSize[0]))
                y1 = max(0, int(ymin * normalSize[1]))
                x2 = min(normalSize[0], int(xmax * normalSize[0]))
                y2 = min(normalSize[1], int(ymax * normalSize[1]))
                if x2 <= x1 or y2 <= y1:
                    continue

                box_area = (x2 - x1) * (y2 - y1)
                rel_size = box_area / (normalSize[0] * normalSize[1])
                if rel_size < 0.03:
                    continue

                center = get_center(x1, y1, x2, y2)
                matched_id = None

                for obj_id, (l, prev_center, last_seen, prev_size) in track_memory.items():
                    if l == label and abs(center[0] - prev_center[0]) < 50 and abs(center[1] - prev_center[1]) < 50:
                        matched_id = obj_id
                        break

                if not matched_id:
                    matched_id = str(uuid.uuid4())

                track_memory[matched_id] = (label, center, now, rel_size)
                direction = determine_direction(x1)
                base_msg = (INDOOR_MESSAGES if mode == "indoor" else OUTDOOR_MESSAGES).get(label, f"{label} detected.")
                message = f"{base_msg} It is {direction}."
                summaries.append(f"{label} {direction}")

                danger_score = compute_danger_score(label, rel_size, "center")

                if danger_score > 14 and now - last_spoken[label] > 4:
                    push_message_to_clients(message)
                    last_spoken[label] = now

                if now - last_logged[label] > 4:
                    db.collection("detection_logs").add({
                        "timestamp": int(now * 1000),
                        "readable_time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S"),
                        "label": label,
                        "confidence": float(scores[i]),
                        "bounding_box": {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        "spoken_message": message,
                        "source": "camera",
                        "frame_id": frame_counter,
                        "danger_score": danger_score
                    })
                    last_logged[label] = now

                # ✅ Smart Video Trigger
                if (
                    config_data.get("smart_video_enabled", True) and
                    danger_score > 18 and
                    rel_size > 0.12 and
                    (now - last_video_time > 10)
                ):
                    last_video_time = now
                    frame_copy = display_frame.copy()
                    threading.Thread(target=record_video_with_frame, args=(frame_copy,), daemon=True).start()

                frame_counter += 1
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} ({scores[i]*100:.1f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if summaries:
                last_detected_summary = "I see: " + ", ".join(summaries)

            ret, jpeg = cv2.imencode('.jpg', display_frame)
            if ret:
                with frame_lock:
                    latest_frame = jpeg.tobytes()

    except Exception as e:
        print("[THREAD ERROR]", e)
    finally:
        try:
            picam2.stop()
        except Exception as e:
            print("[CAMERA STOP ERROR]", e)




# --- API Routes ---
@app.route("/resolution_mode", methods=["POST"])
def update_res_mode():
    try:
        data = request.get_json()
        config_data["high_res_mode"] = data.get("high_res_mode", True)
        return jsonify({
            "status": "ok",
            "high_res_mode": config_data["high_res_mode"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/get_summary")
def get_summary():
    global last_detected_summary
    return jsonify({"summary": last_detected_summary or "Nothing detected right now."})


@app.route('/wifi_status')
def wifi_status():
    return jsonify(get_wifi_status())


@app.route("/shutdown", methods=["POST"])
def shutdown_pi():
    print("[SHUTDOWN] Shutdown route was triggered!")
    try:
        subprocess.run(["sudo", "shutdown", "now"], check=False)
        return jsonify({"message": "Shutdown command sent."})
    except Exception as e:
        print("[SHUTDOWN] ERROR:", e)
        return jsonify({"message": f"Shutdown failed: {e}"}), 500

@app.route("/save_custom_labels", methods=["POST"])
def save_custom_labels():
    try:
        labels = request.json.get("labels", [])
        db.collection("config").document("custom_labels").set({
            "labels": labels,
            "timestamp": int(time.time() * 1000)
        })
        return jsonify({"status": "success", "labels": labels})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/load_custom_labels", methods=["GET"])
def load_custom_labels():
    try:
        doc = db.collection("config").document("custom_labels").get()
        if doc.exists:
            return jsonify({"status": "success", "labels": doc.to_dict().get("labels", [])})
        return jsonify({"status": "empty", "labels": []})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/start", methods=["POST"])
def start_detection():
    global detection_active
    detection_active = True
    return jsonify({"status": "Detection started"})

@app.route("/stop", methods=["POST"])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({"status": "Detection stopped"})

@app.route("/voice_alert_toggle", methods=["POST"])
def voice_toggle():
    global voice_alert_enabled
    voice_alert_enabled = request.json.get("enabled", True)
    return jsonify({"voice_alert_enabled": voice_alert_enabled})

@app.route("/speak", methods=["POST"])
def speak():
    msg = request.json.get("message", "")
    push_message_to_clients(msg)
    return jsonify({"status": "spoken", "message": msg})

@app.route("/reset_wifi", methods=["POST"])
def reset_wifi():
    try:
        config_path = "/etc/wpa_supplicant/wpa_supplicant.conf"
        backup_path = f"{config_path}.bak"
        if os.path.exists(config_path):
            shutil.move(config_path, backup_path)
        os.system("sudo reboot")
        return jsonify({"message": "Wi-Fi reset. Rebooting..."})
    except Exception as e:
        return jsonify({"message": f"Failed: {e}"})

@app.route("/log_location", methods=["POST"])
def log_location():
    data = request.json
    lat = data.get("lat")
    lng = data.get("lng")
    speed = data.get("speed", 0)
    distance = data.get("distance", 0)
    timestamp = data.get("timestamp", int(time.time() * 1000))

    if lat is None or lng is None:
        return jsonify({"status": "error", "message": "Missing lat/lng"}), 400

    db.collection('location_logs').add({
        "lat": lat,
        "lng": lng,
        "speed_kmh": round(speed, 2),
        "distance_m": round(distance, 2),
        "timestamp": timestamp,
        "readable_time": datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
    })

    return jsonify({"status": "ok"})


@app.route("/motion", methods=["POST"])
def receive_motion():
    data = request.get_json()
    motion = data.get("moving")
    global motion_active
    motion_active = motion
    db.collection('motion_logs').add({
        'timestamp': int(time.time() * 1000),
        'readable_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'motion_status': 'active' if motion else 'inactive',
        'motion_value': 1 if motion else 0  # ✅ Added for graph support
    })
    return jsonify({"status": "received", "motion": motion})

@app.route('/latest_video')
def latest_video():
    try:
        with open("/home/ada/de/logs/video_clips/latest.txt", "r") as f:
            path = f.read().strip()
        return send_file(path, as_attachment=True)
    except Exception as e:
        return f"Error loading video: {e}", 500

        
@app.route("/delete_logs", methods=["POST"])
def delete_logs():
    try:
        collections = [
            'battery_logs',
            #'ultrasonic_logs',
            'motion_logs',
            'detection_logs',
            'location_logs',
            'system_health_logs',
            'video_logs'
        ]
        for col in collections:
            docs = db.collection(col).stream()
            for doc in docs:
                doc.reference.delete()
        return jsonify({"status": "success", "message": "All logs deleted."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
        


# --- Ngrok Tunnel ---

# Make sure `socketio` and `app` are already defined somewhere above

def start_flask():
    print("[FLASK] Starting Flask app...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

def start_ngrok():
    try:
        time.sleep(5)  # Give Flask time to bind to port 5000

        print("[NGROK] Launching tunnel to https://smartaid.ngrok.io ...")
        process = subprocess.Popen([
            "ngrok", "http", "--domain=smartaid.ngrok.io", "5000"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Optional: Wait a bit and check logs for troubleshooting
        time.sleep(5)
        # Show last 10 lines of ngrok log for debugging (optional)
        print("[NGROK] Tunnel started. Check dashboard or browser: https://smartaid.ngrok.io")

        return process
    except Exception as e:
        print(f"[NGROK] Failed to start: {e}")
        return None

if __name__ == "__main__":
    try:
        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()

        # Start ngrok after Flask binds
        ngrok_proc = start_ngrok()

        # Start background monitoring threads
        threading.Thread(target=ultrasonic_loop, daemon=True).start()
        threading.Thread(target=battery_monitor, daemon=True).start()
        threading.Thread(target=detection_loop, daemon=True).start()
        threading.Thread(target=system_metrics_monitor, daemon=True).start()

        # ✅ Announce system is ready
        time.sleep(3)  # optional delay to allow WebSocket connection
        push_message_to_clients("Smart Hat system online and ready.")

        # Keep the main thread alive
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected. Shutting down gracefully...")
        if 'ngrok_proc' in locals() and ngrok_proc:
            ngrok_proc.terminate()
            print("[NGROK] Tunnel closed")
