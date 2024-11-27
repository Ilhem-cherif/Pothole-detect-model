from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load YOLO model
model = YOLO("best.pt")
class_names = model.names

# Create directories for storing uploaded and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/detect', methods=['POST'])
def detect_potholes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Save the uploaded image
    file = request.files['image']
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Read the image using OpenCV
    img = cv2.imread(input_path)

    # Run YOLO model prediction
    results = model.predict(img)

    # Process detections and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right corners
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_id = int(box.cls)
            conf = box.conf[0]
            class_name = class_names[class_id]

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save the processed image
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    cv2.imwrite(output_path, img)

    # Return processed image as response
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
