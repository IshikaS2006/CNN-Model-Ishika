from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import os

# Import the predict_pil function from infer.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infer import predict_pil

app = Flask(__name__, static_folder='../static')

# Route to serve index.html
@app.route('/')
def index():
	return send_from_directory(app.static_folder, 'index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
	if 'file' not in request.files:
		return jsonify({'error': 'No file part'}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({'error': 'No selected file'}), 400
	try:
		image = Image.open(file.stream).convert('RGB')
		result = predict_pil(image)
		return jsonify(result)
	except Exception as e:
		return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
	app.run(debug=True)
