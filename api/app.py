from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import requests

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

# Weather API Key and URL
API_KEY = 'your_openweathermap_api_key'
WEATHER_URL = 'http://api.openweathermap.org/data/2.5/weather'

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_weather(location):
    """Get weather information from OpenWeatherMap API."""
    params = {'q': location, 'appid': API_KEY, 'units': 'metric'}
    response = requests.get(WEATHER_URL, params=params)
    data = response.json()
    if data.get("cod") != 200:
        return "Unknown"
    weather = data.get('weather')[0].get('main')
    temp = data.get('main').get('temp')
    return f"{weather} at {temp}°C"

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
            max_prob = float(np.max(predictions[0]))  # Use raw model output
            
            # Check if the prediction confidence is above the threshold
            if max_prob < CONFIDENCE_THRESHOLD:
                predicted_class = "Unknown"
            else:
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # Use raw model output

            # Get weather information
            location = "your_location_here"  # This can be dynamically set or entered by the user
            weather = get_weather(location)

            # Chatbot recommendation based on the predicted disease
            if predicted_class == "Potato___Early_blight":
                recommendation = f"Early Blight detected. The current weather is {weather}. Use copper-based fungicides. Ensure proper spacing and increase airflow."
            elif predicted_class == "Potato___Late_blight":
                recommendation = f"Late Blight detected. The current weather is {weather}. Apply fungicides immediately, especially during wet conditions."
            elif predicted_class == "Potato___healthy":
                recommendation = f"The potato plant looks healthy. Current weather: {weather}. Keep monitoring regularly for pests and disease."
            else:
                recommendation = f"Unable to detect a disease. Current weather: {weather}. Ensure proper care for healthy growth."

            return render_template("index.html", prediction=predicted_class, confidence=max_prob, recommendation=recommendation)
        
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
