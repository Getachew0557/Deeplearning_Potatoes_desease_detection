from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "api", "saved-models", "model.keras")

MODEL = keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy'

]

CONFIDENCE_THRESHOLD = 0.6  # Define a threshold for classification

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        try:
            image = Image.open(file).convert('RGB')
            processed_image = preprocess_image(image)

            # Predict using the model
            predictions = MODEL.predict(processed_image)
            # Remove softmax application
            max_prob = float(np.max(predictions[0]))  # Use raw model output

            
            # Get the highest probability and corresponding class
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # Use raw model output


            # Remove classification as "unknown"


            return render_template("index.html", prediction=predicted_class, confidence=max_prob)
        
        
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
