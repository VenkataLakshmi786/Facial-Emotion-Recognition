import os
import uuid
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageEnhance
import numpy as np
from flask import Flask, request, render_template


app = Flask(__name__)


# Create the folder used to store uploaded images.
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Select the device for model inference.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the emotion classes used during model training.
class_names = ["Angry", "Fear", "Happy", "Sad", "Suprise"]


# Create the ResNet18 model architecture.
resnet_model = models.resnet18(weights=None)


# Get the number of input features of the final layer.
num_features = resnet_model.fc.in_features


# Replace the final layer with the emotion classes.
resnet_model.fc = nn.Linear(num_features, len(class_names))


# Load the saved model.
checkpoint = torch.load("best_emotion_model.pth", map_location=device)


# Load the trained model weights.
resnet_model.load_state_dict(checkpoint["model_state_dict"])


# Move the model to the selected device.
resnet_model = resnet_model.to(device)


# Set the model to evaluation mode.
resnet_model.eval()


print("ResNet18 emotion model loaded successfully.")
print("Device:", device)
print("Classes:", class_names)


# Apply the preprocessing
def preprocess_image(image):

    # Resize the image to the model input size.
    image = image.resize((224, 224))

    # Convert the image to grayscale.
    image = image.convert("L")

    # Convert grayscale to three channels.
    image = np.stack([np.array(image)] * 3, axis=-1)

    image = Image.fromarray(image.astype(np.uint8))

    # Apply sharpening with factor 2.0.
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    # Convert the image to NumPy float32.
    image = np.asarray(image).astype(np.float32)

    # Scale pixel values to 0 to 1.
    image = image / 255.0

    # Define ImageNet normalization values.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Apply ImageNet normalization.
    image = (image - mean) / std

    # Convert HWC format to CHW format.
    image = np.transpose(image, (2, 0, 1))

    # Convert the image to a PyTorch tensor.
    image = torch.tensor(image, dtype=torch.float32)

    # Add the batch dimension.
    image = image.unsqueeze(0)

    return image


@app.route("/")
def home():

    # Display the main page.
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded.
    if "image" not in request.files:
        return render_template("index.html", error="Please upload an image.")

    # Get the uploaded image.
    image_file = request.files["image"]

    # Check whether a file was selected.
    if image_file.filename == "":
        return render_template("index.html", error="Please select an image.")

    try:

        # Open the uploaded image as RGB.
        image = Image.open(image_file).convert("RGB")

        # Generate a unique filename.
        filename = str(uuid.uuid4()) + ".jpg"

        # Create the complete image path.
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the original uploaded image.
        image.save(image_path, format="JPEG")

        # Apply the same preprocessing used during training.
        processed_image = preprocess_image(image)

        # Move the processed image to the selected device.
        processed_image = processed_image.to(device)

        # Disable gradient calculation during prediction.
        with torch.no_grad():

            # Generate model outputs.
            outputs = resnet_model(processed_image)

            # Convert outputs into probabilities.
            probabilities = torch.softmax(outputs, dim=1)

            # Get the predicted class index.
            predicted_index = torch.argmax(probabilities, dim=1).item()

            # Get the predicted emotion.
            predicted_emotion = class_names[predicted_index]

            # Get the prediction confidence.
            confidence = probabilities[0][predicted_index].item() * 100

        # Convert Windows path separators for the browser.
        browser_image_path = image_path.replace("\\", "/")

        # Send prediction information and image path.
        return render_template("index.html", prediction=predicted_emotion, confidence=round(confidence, 2), image_path="/" + browser_image_path)

    except Exception as error:

        return render_template("index.html", error=f"Error processing image: {error}")


if __name__ == "__main__":

    # Start the Flask application.
    app.run(debug=True)