from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
from werkzeug.utils import secure_filename

from service.evaluate import evaluate_signature_from_image_local
from utils.load import load_model_from_checkpoint

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once at startup
sess = load_model_from_checkpoint()



@app.route("/")
def home():
    return "Welcome to Flask server!"


@app.route("/predict", methods=["POST"])
def predict():
    person_id = request.form.get("person_id")
    image_file = request.files.get("image")

    if not person_id or not image_file:
        return jsonify({"error": "Missing person ID or image file"}), 400

    # Secure the filename and save the image
    filename = secure_filename(f"{person_id}_{image_file.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(file_path)

    # 🔽 (Optional) Log or return the saved path
    print(f"Saved image to {file_path}")

    # TODO: Add your image processing and prediction code here
    # For now, let's just return success with file info
    result, gen_conf, fog_conf = evaluate_signature_from_image_local(file_path, person_id)

    return jsonify({
        "message": "File received and saved successfully.",
        "filename": filename,
        "path": file_path,
        "result": result,  # Mock result
        "confidence_genuine": float(gen_conf) ,
        "confidence_forged": float(fog_conf)
}), 200

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=8000)