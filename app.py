from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import numpy as np

app = Flask(__name__)

# URL de DroidCam (ajusta con tu IP)
DROIDCAM_URL = "http://192.168.8.101:4747/video"

# Cargar modelo
with open("modelo_color.pkl", "rb") as f:
    modelo = pickle.load(f)

# Datos de mezclas
mezclas = {
    "Passion_Red": {
        "125ml": {
            "UR50 UNIVERSAL MIL": 10.8,
            "BC105 WHITE": 10.2,
            "BC201 DEEP BLACK": 0.9,
            "BC710 BRIGHT ORANGE": 37.7,
            "BC815 ORGANIC BRIGHT": 21.5,
            "BC832 RED": 36.3
        },
        "250ml": { "UR50 UNIVERSAL MIL": 21.6, "BC105 WHITE": 20.4, "BC201 DEEP BLACK": 1.9, "BC710 BRIGHT ORANGE": 75.4, "BC815 ORGANIC BRIGHT": 43.0, "BC832 RED": 72.5 },
        "500ml": { "UR50 UNIVERSAL MIL": 43.2, "BC105 WHITE": 40.8, "BC201 DEEP BLACK": 3.8, "BC710 BRIGHT ORANGE": 150.7, "BC815 ORGANIC BRIGHT": 85.9, "BC832 RED": 145.1 },
        "1000ml": { "UR50 UNIVERSAL MIL": 86.4, "BC105 WHITE": 81.7, "BC201 DEEP BLACK": 7.5, "BC710 BRIGHT ORANGE": 301.4, "BC815 ORGANIC BRIGHT": 171.8, "BC832 RED": 290.2 },
    },
    "Celeste_Pearl": {
        "125ml": { "UR50 UNIVERSAL MI": 12.1, "BC194 HI HIDING WHITE": 94.4, "BC259 LS BLACK": 2.6, "BC407 PHTHALO BLUE II": 13.1, "BC410 GREEN BLUE": 1.9, "BC805 RED OXIDE": 10.8, "BC840 RED VIOLET": 0.5 },
        "250ml": { "UR50 UNIVERSAL MI": 24.2, "BC194 HI HIDING WHITE": 188.8, "BC259 LS BLACK": 5.3, "BC407 PHTHALO BLUE II": 26.3, "BC410 GREEN BLUE": 3.7, "BC805 RED OXIDE": 21.5, "BC840 RED VIOLET": 1.1 },
        "500ml": { "UR50 UNIVERSAL MI": 48.3, "BC194 HI HIDING WHITE": 377.7, "BC259 LS BLACK": 10.5, "BC407 PHTHALO BLUE II": 52.6, "BC410 GREEN BLUE": 7.4, "BC805 RED OXIDE": 43.0, "BC840 RED VIOLET": 2.1 },
        "1000ml": { "UR50 UNIVERSAL MI": 96.7, "BC194 HI HIDING WHITE": 755.4, "BC259 LS BLACK": 21, "BC407 PHTHALO BLUE II": 105.2, "BC410 GREEN BLUE": 14.9, "BC805 RED OXIDE": 86.1, "BC840 RED VIOLET": 4.2 },
    },
    "Grey": {
        "125ml": { "UR50 UNIVERSAL MI": 17, "BC194 HI HIDING WHITE": 76.7, "BC200 BLACK": 7.1, "BC250 BLUE BLACK": 17, "BC600 GOLD": 9.2, "BC805 RED OXIDE": 4.5 },
        "250ml": { "UR50 UNIVERSAL MI": 34, "BC194 HI HIDING WHITE": 153.3, "BC200 BLACK": 14.2, "BC250 BLUE BLACK": 34, "BC600 GOLD": 18.3, "BC805 RED OXIDE": 9 },
        "500ml": { "UR50 UNIVERSAL MI": 68.1, "BC194 HI HIDING WHITE": 306.6, "BC200 BLACK": 28.4, "BC250 BLUE BLACK": 68.1, "BC600 GOLD": 36.6, "BC805 RED OXIDE": 18 },
        "1000ml": { "UR50 UNIVERSAL MI": 136.1, "BC194 HI HIDING WHITE": 613.2, "BC200 BLACK": 56.8, "BC250 BLUE BLACK": 136.1, "BC600 GOLD": 73.3, "BC805 RED OXIDE": 35.9 },
    },
    "Racing_Blue": {
        "125ml": { "UR50 UNIVERSAL MI": 11, "BC194 HI HIDING WHITE": 23.5, "BC400 RED-BLUE": 50.2, "BC410 GREEN BLUE": 35.9 },
        "250ml": { "UR50 UNIVERSAL MI": 21.9, "BC194 HI HIDING WHITE": 47, "BC400 RED-BLUE": 100.3, "BC410 GREEN BLUE": 71.9 },
        "500ml": { "UR50 UNIVERSAL MI": 43.9, "BC194 HI HIDING WHITE": 94, "BC400 RED-BLUE": 200.6, "BC410 GREEN BLUE": 143.7 },
        "1000ml": { "UR50 UNIVERSAL MI": 87.8, "BC194 HI HIDING WHITE": 188.1, "BC400 RED-BLUE": 401.2, "BC410 GREEN BLUE": 287.4 },
    },
    "SR150.50": {
        "125ml": { "UR50 UNIVERSAL MI": 10.4, "BC194 HI HIDING WHITE": 1, "BC815 ORGANIC BRIGHT": 17.9, "BC832 RED": 81.3, "BC880 VIOLET": 3.3 },
        "250ml": { "UR50 UNIVERSAL MI": 20.7, "BC194 HI HIDING WHITE": 2, "BC815 ORGANIC BRIGHT": 35.7, "BC832 RED": 162.5, "BC880 VIOLET": 6.6 },
        "500ml": { "UR50 UNIVERSAL MI": 41.4, "BC194 HI HIDING WHITE": 4.1, "BC815 ORGANIC BRIGHT": 71.5, "BC832 RED": 325.1, "BC880 VIOLET": 13.2 },
        "1000ml": { "UR50 UNIVERSAL MI": 82.9, "BC194 HI HIDING WHITE": 8.2, "BC815 ORGANIC BRIGHT": 143, "BC832 RED": 650.1, "BC880 VIOLET": 26.4 },
    },
}

color_detectado = "desconocido"

def generar_frames():
    global color_detectado
    cap = cv2.VideoCapture(DROIDCAM_URL)  # Abrimos UNA vez

    if not cap.isOpened():
        print("Error: No se pudo conectar con DroidCam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        x, y, ancho, alto = w//2 - 50, h//2 - 50, 100, 100
        roi = frame[y:y+alto, x:x+ancho]

        cv2.rectangle(frame, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
        color_promedio = roi.mean(axis=(0, 1))
        color_detectado = modelo.predict([color_promedio])[0]

        cv2.putText(frame, f"Color: {color_detectado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_color')
def get_color():
    return jsonify({"color": color_detectado})

@app.route('/get_mezcla/<tamano>')
def get_mezcla(tamano):
    color = color_detectado
    mezcla = mezclas.get(color, {}).get(tamano, {})
    return jsonify(mezcla)

if __name__ == '__main__':
    app.run(debug=True)