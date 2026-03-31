import os
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = os.urandom(24)

DEMO_USER = {"username": "admin", "password": "password"}
MODEL_PATH = "ecosort_ai_model.keras"
IMG_SIZE = (96, 96)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = None
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

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
    model_status = "loaded" if model is not None else "missing"
    tf_status = "available" if tf is not None else "missing"
    stats = {
        "Users": 1,
        "Model status": model_status,
        "TensorFlow": tf_status,
        "Demo account": DEMO_USER["username"],
    }
    return {
        "username": session.get("username", "User"),
        "stats": stats,
        "model_available": model_available(),
        "model_path": MODEL_PATH,
        "tf_available": tf is not None,
        "class_names": class_names,
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
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == DEMO_USER["username"] and password == DEMO_USER["password"]:
            session["logged_in"] = True
            session["username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."

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
    upload_error = None

    if "image" not in request.files:
        upload_error = "No file uploaded. Please choose an image."
    else:
        image_file = request.files["image"]
        if image_file.filename == "":
            upload_error = "No file selected. Please choose an image."
        elif not allowed_file(image_file.filename):
            upload_error = "Unsupported file type. Upload PNG or JPG images only."
        else:
            if model_available():
                try:
                    prediction, prediction_prob = predict_image(image_file)
                except Exception as exc:
                    upload_error = f"Prediction failed: {exc}"
            else:
                prediction, prediction_prob = demo_predict(image_file.filename)
                upload_error = None
                flash("Demo prediction shown because the saved model is not available.", "warning")

    return render_template(
        "dashboard.html",
        prediction=prediction,
        prediction_prob=prediction_prob,
        upload_error=upload_error,
        **get_dashboard_context(),
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
