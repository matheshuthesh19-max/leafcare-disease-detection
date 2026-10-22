# 🌿 LeafCare — இலை நோய் கண்டறியும் வெப்சைட் (Plant Leaf Disease Detector)

Tamil-language Flask web app for farmers: log in, upload a leaf photo, and
get a disease diagnosis using **grayscale conversion + colour
thresholding / segmentation** (classical OpenCV image processing). Every
page has a farm-themed animated background (flying birds, butterflies,
falling leaves, swaying trees, a farmer illustration) and results can be
downloaded as a **Tamil PDF report**.

## 1. Setup (VS Code / any machine)

```bash
cd leaf_disease_app
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

**Demo login:** username `farmer`, password `1234`
(edit `DEMO_USERNAME` / `DEMO_PASSWORD` / `DEMO_NAME` / `DEMO_MOBILE` in
`app.py`, or wire up a real user database, before going live).

## 2. What's new in this version

- **முழுவதும் தமிழ் UI** — login, upload, result pages, and the PDF
  report are all in Tamil so every farmer can understand it.
- **Login page** — username + password fields, plus **"பயனர்பெயர்
  மறந்துவிட்டதா?" (forgot username)** and **"கடவுச்சொல்
  மறந்துவிட்டதா?" (forgot password)** links. These are demo flows (no
  real SMS/email is sent) — they check against `DEMO_MOBILE` in
  `app.py` and reveal/confirm reset for testing. Wire up real SMS/email
  + a user database before production use.
- **More animations** — birds flying across the sky, butterflies
  fluttering, leaves falling, swaying trees, a farmer illustration, a
  "growing plant" loader while the image is analysed, and a leaf-confetti
  burst on the result page.
- **Image-processing stages shown on the result page**:
  1. மூலப் படம் (Original / RGB)
  2. Grayscale படம்
  3. Segmentation / Threshold mask (Otsu thresholding)
  4. நோய்வாய்ப்பட்ட பகுதி சிவப்பு நிறத்தில் குறிக்கப்பட்ட படம்
     (diseased area highlighted with contours)
- **Disease name + "எப்படி தடுப்பது" (how to prevent/treat it)** shown
  on-screen and in the PDF, for every disease class in
  `utils/disease_detector.py`.
- **PDF report download** — now in Tamil, using a bundled Tamil font
  (`static/fonts/Lohit-Tamil.ttf`) with automatic Latin-glyph fallback
  so mixed Tamil/English text (e.g. "RGB", "Otsu Thresholding") renders
  correctly instead of empty boxes.

## 3. How the detection works

```
Upload image
   -> RGB to Grayscale (cv2.cvtColor)                     [saved as *_gray.jpg]
   -> Otsu thresholding (cv2.threshold)                    [saved as *_segmented.jpg]
      -> separates leaf from background (segmentation)
   -> HSV colour thresholding
      -> separates healthy green vs diseased brown/yellow/dark tissue
   -> contours drawn on diseased regions                   [saved as *_highlighted.jpg]
   -> % diseased area -> severity + probable disease name + prevention advice
   -> Result page (Tamil) + downloadable Tamil PDF report
```

All of that logic lives in `utils/disease_detector.py`; the intermediate
stage images are saved next to the uploaded photo and shown in a gallery
on the result page.

## 4. Project structure

```
leaf_disease_app/
├── app.py                      Flask routes (login, forgot-password/
│                                username, upload, result, PDF)
├── requirements.txt
├── utils/
│   ├── disease_detector.py     grayscale + thresholding + segmentation
│   │                           + disease name + Tamil prevention advice
│   └── pdf_report.py           Tamil PDF report generator (reportlab)
├── templates/
│   ├── base.html
│   ├── login.html              username/password + forgot links
│   ├── forgot_username.html    demo "forgot username" flow
│   ├── forgot_password.html    demo "forgot password" flow
│   ├── upload.html             drag & drop leaf upload + loader animation
│   ├── result.html             diagnosis + stage gallery + PDF download
│   ├── _scene.html             shared birds/butterflies/leaves/trees animation
│   └── _farmer.html            farmer SVG illustration
├── static/
│   ├── css/style.css           all animations + design system
│   ├── js/animations.js
│   └── fonts/Lohit-Tamil.ttf   bundled Tamil font for the PDF report
├── uploads/                    uploaded + stage images (created at runtime)
└── reports/                    generated PDF reports (created at runtime)
```

## 5. Upgrading with the Kaggle PlantVillage dataset (real ML accuracy)

The current detector is a fast, dependency-light **heuristic** — good for
a working demo/prototype, no training data needed. For production-grade
accuracy:

1. Download **"PlantVillage Dataset"** from Kaggle (search
   `plantvillage`; needs a free Kaggle account + API token, since Kaggle
   requires authentication and isn't reachable from this build
   environment).
2. Train a lightweight CNN (e.g. MobileNetV2 transfer learning) on the
   dataset classes (Healthy, Bacterial Spot, Early Blight, Late Blight,
   Powdery Mildew, Leaf Rust, etc.).
3. Save the trained model (`model.h5` / `.pt`).
4. Replace the body of `analyze_leaf()` in `utils/disease_detector.py`
   with `model.predict(...)`, keeping the **same return dictionary
   shape** (`status`, `disease_name`, `severity`, `diseased_percent`,
   `healthy_percent`, `advice`, `gray_file`, `segmentation_file`,
   `highlighted_file`) — so no other file needs to change.

## 6. Further ideas

- Replace the demo login with a real user database (SQLite/PostgreSQL)
  and real SMS/email delivery for forgot-password/username.
- Store diagnosis history per farmer so they can track a leaf over time.
- Add an English/Tamil language toggle.
- Add an SMS/WhatsApp alert when a "கடுமையானது (Severe)" result is
  detected.
