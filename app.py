import os
import uuid
import json
from datetime import datetime
from leaf_condition import check_leaf_condition

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, send_from_directory, flash
)
from werkzeug.utils import secure_filename

from utils.disease_detector import analyze_leaf
from utils.pdf_report import build_pdf_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.secret_key = "leafcare-secret-key-change-me"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
USERS_FILE = os.path.join(BASE_DIR, "users.json")


def load_users():
    if not os.path.exists(USERS_FILE):
        return []

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("upload_page"))
    return redirect(url_for("login"))


@app.route("/forgot_username", methods=["GET", "POST"])
def forgot_username():
    revealed = None

    if request.method == "POST":
        mobile = request.form.get("mobile", "").strip()
        users = load_users()

        user = next(
            (u for u in users
             if u.get("mobile", "").strip() == mobile),
            None
        )

        if user:
            revealed = user["username"]
        else:
            flash("இந்த மொபைல் எண்ணுடன் கணக்கு கிடைக்கவில்லை.")

    return render_template(
        "forgot_username.html",
        revealed=revealed
    )
@app.route("/login", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        users = load_users()

        for user in users:

            if user["username"] == username and user["password"] == password:

                session["logged_in"] = True
                session["username"] = username
                session["display_name"] = user["name"]

                return redirect(url_for("upload_page"))

        error = "Invalid Username or Password"

    return render_template("login.html", error=error)
@app.route("/register", methods=["GET", "POST"])
def register():

    error = None

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        mobile = request.form.get("mobile", "").strip()
        password = request.form.get("password", "").strip()

        if not name or not username or not mobile or not password:
            error = "அனைத்து விவரங்களையும் நிரப்பவும்."
            return render_template("register.html", error=error)

        users = load_users()

        # Username already exists
        for user in users:
            if user["username"] == username:
                error = "இந்த Username ஏற்கனவே பயன்படுத்தப்படுகிறது."
                return render_template("register.html", error=error)

        users.append({
            "name": name,
            "username": username,
            "mobile": mobile,
            "password": password
        })

        save_users(users)

        flash("பதிவு வெற்றிகரமாக முடிந்தது. இப்போது Login செய்யவும்.")

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        mobile = request.form.get("mobile", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(new_password) < 10:
            error = "புதிய கடவுச்சொல் குறைந்தது 10 எழுத்துகள் இருக்க வேண்டும்."

        elif new_password != confirm_password:
            error = "இரண்டு கடவுச்சொற்களும் ஒன்றாக இல்லை."

        else:
            users = load_users()

            user = next(
                (
                    u for u in users
                    if u.get("username", "").lower() == username.lower()
                    and u.get("mobile", "").strip() == mobile
                ),
                None
            )

            if user:
                from werkzeug.security import generate_password_hash

                user["password"] = generate_password_hash(new_password)
                save_users(users)

                flash("கடவுச்சொல் மாற்றப்பட்டது. இப்போது Login செய்யவும்.")
                return redirect(url_for("login"))

            error = "Username அல்லது மொபைல் எண் சரியாக இல்லை."

    return render_template("forgot_password.html", error=error)
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":

        # ---------------------------------------------
        # 1. Get uploaded image
        # ---------------------------------------------
        file = request.files.get("leaf_image")

        if not file or file.filename == "":
            flash("முதலில் ஒரு இலை படத்தை தேர்ந்தெடுக்கவும்.")
            return redirect(url_for("upload_page"))

        # ---------------------------------------------
        # 2. Check file extension
        # ---------------------------------------------
        if not allowed_file(file.filename):
            flash("PNG / JPG / JPEG / BMP படங்கள் மட்டுமே ஏற்கப்படும்.")
            return redirect(url_for("upload_page"))

        # ---------------------------------------------
        # 3. Save uploaded image
        # ---------------------------------------------
        ext = file.filename.rsplit(".", 1)[1].lower()

        unique_name = f"{uuid.uuid4().hex}.{ext}"

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_name
        )

        file.save(save_path)

        # ---------------------------------------------
        # 4. Check whether leaf is dry or not
        # ---------------------------------------------
        condition = check_leaf_condition(save_path)

        # ---------------------------------------------
        # 5. If image is DRY, stop processing
        # ---------------------------------------------
        if condition["status"] == "DRY":

            # Remove the uploaded image
            # because it is not suitable for
            # further disease processing.
            try:
                os.remove(save_path)
            except OSError:
                pass

            flash(
                f"⚠️ {condition['message']} "
                f"Confidence: {condition['confidence']}%"
            )

            return redirect(url_for("upload_page"))

        # ---------------------------------------------
        # 6. If condition check has an error
        # ---------------------------------------------
        if condition["status"] == "ERROR":

            try:
                os.remove(save_path)
            except OSError:
                pass

            flash(
                f"❌ படத்தை சரிபார்க்க முடியவில்லை: "
                f"{condition['message']}"
            )

            return redirect(url_for("upload_page"))

        # ---------------------------------------------
        # 7. Leaf is NOT DRY
        #    Continue with image processing
        # ---------------------------------------------

        result = analyze_leaf(save_path)

        # Add condition information
        result["leaf_condition"] = condition["status"]
        result["condition_confidence"] = condition["confidence"]

        # Add uploaded image name
        result["image_file"] = unique_name

        # Add timestamp
        result["timestamp"] = datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        )

        # ---------------------------------------------
        # 8. Store result in session
        # ---------------------------------------------
        session["last_result"] = result

        # ---------------------------------------------
        # 9. Go to result page
        # ---------------------------------------------
        return redirect(url_for("result_page"))

    return render_template("upload.html")

@app.route("/result")
def result_page():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    result = session.get("last_result")
    if not result:
        return redirect(url_for("upload_page"))
    return render_template("result.html", result=result)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/download-report")
def download_report():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    result = session.get("last_result")
    if not result:
        return redirect(url_for("upload_page"))

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], result["image_file"])
    report_name = f"leaf_report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(app.config["REPORT_FOLDER"], report_name)

    build_pdf_report(
        output_path=report_path,
        image_path=image_path,
        result=result,
        farmer_name=session.get("display_name", session.get("username", "Farmer")),
    )

    return send_from_directory(
        app.config["REPORT_FOLDER"], report_name, as_attachment=True
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
