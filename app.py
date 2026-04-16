import os
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = os.urandom(24)

DEMO_USER = {"username": "admin", "password": "password"}
MODEL_PATH = "ecosort_ai_model.keras"
IMG_SIZE = (96, 96)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

import io
import base64

model = None
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_DETAILS = {
    "cardboard": {
        "bin_color": "Blue Bin", 
        "reuse_percentage": "95%", 
        "description": "Highly recyclable. Break down boxes before disposing.",
        "upcycle": "DIY Idea: Create a desktop organizer, drawer dividers, or a fun playhouse for your pets!",
        "impact": "By recycling this cardboard, you save significant water and prevent deforestation."
    },
    "glass": {
        "bin_color": "Green/Blue Bin", 
        "reuse_percentage": "100%", 
        "description": "100% recyclable without loss in quality.",
        "upcycle": "DIY Idea: Wash it and turn it into a painted flower vase, a terrarium, or a beautiful candle holder!",
        "impact": "Recycling glass reduces related air pollution by 20% and water pollution by 50%!"
    },
    "metal": {
        "bin_color": "Blue Bin", 
        "reuse_percentage": "100%", 
        "description": "Metals can be recycled indefinitely.", 
        "e_waste": True,
        "upcycle": "DIY Idea: Clean metal cans can be painted to make bold pen holders, rustic planters, or hanging lanterns!",
        "impact": "Recycling just 1 aluminum can saves enough energy to run a TV for 3 hours!"
    },
    "paper": {
        "bin_color": "Blue Bin", 
        "reuse_percentage": "90%", 
        "description": "Recyclable, ensure it is dry and clean.",
        "upcycle": "DIY Idea: Blend old paper with water to create your own handmade paper or origami art.",
        "impact": "Recycling paper saves trees and reduces water consumption."
    },
    "plastic": {
        "bin_color": "Blue/Yellow Bin", 
        "reuse_percentage": "50-80%", 
        "description": "Rinse containers before recycling.",
        "upcycle": "DIY Idea: Use bottles as plant holders or bird feeders!",
        "impact": "Recycling plastic saves energy and reduces pollution."
    },
    "trash": {
        "bin_color": "Black/Red Bin", 
        "reuse_percentage": "0%", 
        "description": "Non-recyclable waste.",
        "upcycle": "Try reducing usage of such items.",
        "impact": "Reducing trash helps minimize landfill pollution."
    }
}

tf = None
try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
except ImportError:
    tf = None
except Exception:
    model = None


def model_available():
    return model is not None and tf is not None


def demo_predict(filename):
    base = filename.lower()
    for waste_type in class_names:
        if waste_type in base:
            return waste_type, 0.82
    return random.choice(class_names), random.uniform(0.5, 0.88)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(file_stream):
    image_data = file_stream.read()
    image = tf.io.decode_image(image_data, channels=3, dtype=tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.expand_dims(image, 0)
    preds = model.predict(image)
    pred_index = int(tf.argmax(preds[0]))
    confidence = float(tf.reduce_max(preds[0]))
    return class_names[pred_index], confidence


def get_dashboard_context():
    return {
        "username": session.get("username", "User"),
        "model_available": model_available(),
        "demo_mode": not model_available(),
    }


def is_logged_in():
    return session.get("logged_in") is True


@app.route("/")
def home():
    if is_logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == DEMO_USER["username"] and password == DEMO_USER["password"]:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid credentials"

    return render_template("login.html", error=error)


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html", **get_dashboard_context())


@app.route("/upload", methods=["POST"])
def upload():
    if not is_logged_in():
        return redirect(url_for("login"))

    prediction = None
    prediction_prob = None

    if "image" in request.files:
        image_file = request.files["image"]
        if model_available():
            prediction, prediction_prob = predict_image(image_file)
        else:
            prediction, prediction_prob = demo_predict(image_file.filename)

    return render_template(
        "dashboard.html",
        prediction=prediction,
        prediction_prob=prediction_prob,
        **get_dashboard_context(),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ✅ ✅ IMPORTANT FIX FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)