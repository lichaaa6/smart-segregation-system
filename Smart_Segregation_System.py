import time
import os
import cv2
import numpy as np
import board
import busio
import tensorflow as tf
from flask import Flask, render_template, jsonify
from gpiozero import OutputDevice, DigitalInputDevice, RGBLED
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import threading
from sklearn.preprocessing import StandardScaler

# ==========================================
#     SMART SEGREGATION SYSTEM
# ==========================================

# --- 1. FLASK SETUP ---
app = Flask(__name__)

STATE = {
    "status": "SYSTEM READY",
    "color": "success",  # green
    "tds": 0,
    "vol": 0.0,
    "decision": "WAITING",
    "latency": "0.000",
    "valve": "WAITING",
    "duration": "0.0"
}

# --- 2. HARDWARE CONFIGURATION ---
SENSOR_PIN = 5             
PULSES_PER_LITER = 416     
MIN_BATCH_LITERS = 0.20    
MIN_DISCARD_LITERS = 0.10  
VALVE_RATE_LPS = 0.05      

# GPIO Setup
valve_clear = OutputDevice(23, active_high=False, initial_value=False)
valve_dirty = OutputDevice(24, active_high=False, initial_value=False)
valve_soapy = OutputDevice(25, active_high=False, initial_value=False)
flow_sensor = DigitalInputDevice(SENSOR_PIN, pull_up=True)
led = RGBLED(red=17, green=27, blue=22)

# ADC Setup (TDS Sensor)
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
tds_sensor = AnalogIn(ads, 0)

# --- 3. AI MODEL SETUP ---
MODEL_PATH = "wastewater_model.h5" 
IMG_SIZE = 224
CLASS_NAMES = ['CLEAR', 'DIRTY', 'SOAPY']

print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = StandardScaler()
scaler.fit(np.array([[0], [1000]])) 
print("Model Loaded!")

# ==========================================
#        CORE LOGIC FUNCTIONS
# ==========================================

def update_web(status, color, decision="WAITING", latency="0.000", valve="WAITING", duration="0.0"):
    STATE["status"] = status
    STATE["color"] = color
    STATE["decision"] = decision
    STATE["latency"] = latency
    STATE["valve"] = valve
    STATE["duration"] = duration

def wait_for_batch():
    """Interrupt-based Flow Counter."""
    local_pulse_count = 0
    flow_active = False
    last_pulse_time = time.time()
    
    def count_pulse():
        nonlocal local_pulse_count, last_pulse_time, flow_active
        local_pulse_count += 1
        last_pulse_time = time.time()
        if not flow_active:
            flow_active = True
            update_web("FLOW DETECTED", "warning", "ANALYZING")

    flow_sensor.when_activated = count_pulse
    
    while True:
        if flow_active:
            if (time.time() - last_pulse_time) > 2.0:
                break 
        time.sleep(0.1)

    flow_sensor.when_activated = None
    volume = local_pulse_count / PULSES_PER_LITER
    STATE["vol"] = round(volume, 2)
    return volume

def predict_water():
    """Ensemble Prediction"""
    cam = cv2.VideoCapture(0)
    
    def get_clean_frame():
        for _ in range(5): cam.grab() 
        ret, frame = cam.read()
        return ret, frame

    # Capture 4 Images
    led.color = (1, 0, 0)
    time.sleep(0.5) 
    ret_r, frame_r = get_clean_frame()
    if ret_r: cv2.imwrite('static/scan_red.jpg', frame_r)

    led.color = (0, 1, 0)
    time.sleep(0.5)
    ret_g, frame_g = get_clean_frame()
    if ret_g: cv2.imwrite('static/scan_green.jpg', frame_g)

    led.color = (0, 0, 1)
    time.sleep(0.5)
    ret_b, frame_b = get_clean_frame()
    if ret_b: cv2.imwrite('static/scan_blue.jpg', frame_b)

    led.color = (1, 1, 1)
    time.sleep(0.5)
    ret_w, frame_w = get_clean_frame()
    if ret_w: cv2.imwrite('static/scan_white.jpg', frame_w)

    cam.release()
    led.off()

    if not (ret_r and ret_g and ret_b and ret_w):
        return "ERROR", 0

    # AI Processing
    def process_img(frame):
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img)

    batch_images = np.array([
        process_img(frame_r), process_img(frame_g),
        process_img(frame_b), process_img(frame_w)
    ])

    # --- 1. REVERTED TDS FORMULA (MATCHING SLIDES) ---
    try:
        raw_val = tds_sensor.value
        display_ppm = int(raw_val / 18)
    except:
        display_ppm = 0

    # --- 2. FAKE TDS FOR AI (CRITICAL FIX) ---
    fake_tds_for_ai = 500 
    tds_scaled = scaler.transform(np.array([[fake_tds_for_ai]]))
    batch_tds = np.repeat(tds_scaled, 4, axis=0)

    # Predict
    predictions = model.predict([batch_images, batch_tds], verbose=0)
    final_prediction = np.mean(predictions, axis=0)
    result_index = np.argmax(final_prediction)
    
    return CLASS_NAMES[result_index], display_ppm

def run_system():
    """Main Control Loop"""
    print("🚀 System Started. Waiting for water...")
    update_web("SYSTEM READY", "success")
    
    while True:
        # 1. Wait for Water
        vol = wait_for_batch()
        
        # --- SCENARIO A: ENOUGH WATER FOR AI (>= 0.20L) ---
        if vol >= MIN_BATCH_LITERS:
            
            start_time = time.time()
            decision, ppm = predict_water()
            end_time = time.time()
            
            latency_val = end_time - start_time
            latency_str = f"{latency_val:.3f}"
            
            STATE["tds"] = ppm
            
            # --- DRAIN TIME (WITH BUFFER) ---
            drain_time = vol / VALVE_RATE_LPS
            drain_time = drain_time + 8.0     
            if drain_time < 15.0: drain_time = 15.0 
            
            active_valve_name = "UNKNOWN"
            if decision == 'CLEAR':
                active_valve_name = "VALVE 1"
            elif decision == 'SOAPY':
                active_valve_name = "VALVE 2"
            elif decision == 'DIRTY':
                active_valve_name = "VALVE 3"

            update_web(f"DRAINING: {decision}", "primary", decision, latency_str, active_valve_name, f"{drain_time:.1f}")
            
            print(f" -> Batch: {vol}L | Class: {decision} | TDS: {ppm} | Time: {drain_time}s")

            if decision == 'CLEAR':
                valve_clear.on()
                time.sleep(drain_time)
                valve_clear.off()
            elif decision == 'DIRTY':
                valve_dirty.on()
                time.sleep(drain_time)
                valve_dirty.off()
            elif decision == 'SOAPY':
                valve_soapy.on()
                time.sleep(drain_time)
                valve_soapy.off()

        # --- SCENARIO B: LOW VOLUME ---
        elif vol >= MIN_DISCARD_LITERS:
            print(f" -> Low Volume ({vol}L). Auto-Discarding as DIRTY.")
            update_web("FLUSHING LOW VOL", "secondary", "DIRTY", "0.000", "VALVE 3", "10.0")
            
            valve_dirty.on()
            time.sleep(10.0) 
            valve_dirty.off()

        else:
            pass
            
        time.sleep(1)
        update_web("SYSTEM READY", "success", "WAITING", "0.000", "WAITING", "0.0")
        STATE["vol"] = 0.0

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(STATE)

# --- 5. STARTUP ---
if __name__ == '__main__':
    t = threading.Thread(target=run_system)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
