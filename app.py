import flask
import base64
import io
import torch
import numpy as np
from PIL import Image
from model import Generator

# Create Flask App
app = flask.Flask(__name__)

# Load Generator
device = torch.device('cpu')
model = Generator()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

@app.route("/")
def index():
    return flask.render_template("index.html")

@app.route("/generate")
def generate():
    noise = torch.randn(1, 100, 1, 1)
    with torch.no_grad():
        # Create tensor from noise
        image = model(noise)
        # Remove first dimension
        image = image.squeeze(0)
        # Denormalize tensor from [-1, 1] to [0, 255]
        image = (image + 1) * 255 / 2
        # Convert tensor to numpy array
        image = np.transpose(image.numpy(), (1, 2, 0))
        # Convert numpy array to image
        image = Image.fromarray(image.astype("uint8"))
        # Convert image to base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue())
    return flask.make_response(flask.jsonify({ "image": image.decode("utf-8") }), 200)
