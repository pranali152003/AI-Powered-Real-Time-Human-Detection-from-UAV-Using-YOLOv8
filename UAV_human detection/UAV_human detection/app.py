from flask import Flask, render_template, Response, send_from_directory, send_file, abort
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import csv
import os
import time
import cv2
import json
import pandas as pd
import platform

# =========================================================
# 1. PROJECT PATHS
# =========================================================
BASE_DIR = Path(r"C:\Users\admin\Desktop\UAV_human detection_dashboard\UAV_human detection\UAV_human detection")
SNAPSHOT_DIR = BASE_DIR / "snapshots"
DATA_DIR = BASE_DIR / "data"
ALERTS_CSV = DATA_DIR / "alerts.csv"
REPORT_XLSX = DATA_DIR / "UAV_Report.xlsx"

# Create only the intended project folders.
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

print("App directory:", BASE_DIR)
print("Template directory:", TEMPLATE_DIR)
print("Snapshot template exists:",
      os.path.isfile(os.path.join(TEMPLATE_DIR, "view_snapshot.html")))

# =========================================================
# 2. LOAD YOLO MODEL
# =========================================================
MODEL_PATH = Path(r"C:\Users\admin\Desktop\drone_human_detection\clg_project\best1.pt")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO model was not found at:\n{MODEL_PATH}\n"
        "Please correct MODEL_PATH to the actual location of best1.pt."
    )

model = YOLO(str(MODEL_PATH))

# Sound is optional. This allows the app to run if winsound is unavailable.
try:
    import winsound
except ImportError:
    winsound = None


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def ensure_alert_csv():
    """Create the CSV with the expected columns only if it doesn't exist."""
    if not ALERTS_CSV.exists():
        with ALERTS_CSV.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Humans", "Level", "Snapshot"])


def append_alert(date_text, time_text, human_count, level, filename):
    """Append one alert without deleting or rewriting previous records."""
    ensure_alert_csv()
    with ALERTS_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([date_text, time_text, human_count, level, filename])


def read_alert_rows():
    """
    Read alerts without deleting rows.
    Supports older CSV files that have no header and newer files with a header.
    """
    if not ALERTS_CSV.exists():
        return []

    with ALERTS_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        all_rows = [row for row in csv.reader(file) if row]

    if all_rows and all_rows[0][0].strip().lower() == "date":
        all_rows = all_rows[1:]

    return all_rows


def alert_level(human_count):
    if human_count <= 2:
        return "LOW"
    if human_count <= 5:
        return "MEDIUM"
    return "HIGH"


# =========================================================
# 4. PAGE ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/alerts")
def alerts():
    # Do NOT rewrite the CSV or delete entries when a snapshot is missing.
    rows = read_alert_rows()
    return render_template("alerts.html", rows=rows)


@app.route("/snapshots")
def snapshots():
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    files = sorted(
        [
            file.name
            for file in SNAPSHOT_DIR.iterdir()
            if file.is_file() and file.suffix.lower() in allowed_extensions
        ],
        reverse=True,
    )
    return render_template("snapshots.html", files=files)


@app.route("/snapshots/<path:filename>")
def snapshot_file(filename):
    # Serve the real image from the fixed snapshots folder.
    file_path = SNAPSHOT_DIR / filename
    if not file_path.is_file():
        abort(404, description=f"Snapshot image not found: {filename}")
    return send_from_directory(str(SNAPSHOT_DIR), filename)


@app.route("/view_snapshot/<path:filename>")
def view_snapshot(filename):
    file_path = SNAPSHOT_DIR / filename
    if not file_path.is_file():
        return (
            f"Snapshot not found: {filename}<br>"
            f"Expected location: {file_path}",
            404,
        )
    return render_template("view_snapshot.html", filename=filename)


@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/report")
def report():
    rows = read_alert_rows()
    records = []

    for row in rows:
        # Skip malformed rows instead of crashing the report.
        if len(row) < 5:
            continue
        records.append(row[:5])

    df = pd.DataFrame(records, columns=["Date", "Time", "Humans", "Level", "Snapshot"])
    if df.empty:
        df = pd.DataFrame(columns=["Date", "Time", "Humans", "Level", "Snapshot"])

    df["Humans"] = pd.to_numeric(df["Humans"], errors="coerce").fillna(0).astype(int)
    df["Level"] = df["Level"].fillna("").astype(str)
    df = df.fillna("")

    total_alerts = len(df)
    total_humans = int(df["Humans"].sum())
    low_count = int((df["Level"] == "LOW").sum())
    medium_count = int((df["Level"] == "MEDIUM").sum())
    high_count = int((df["Level"] == "HIGH").sum())

    rows_json = json.dumps(df.values.tolist())

    return render_template(
        "report.html",
        total_alerts=total_alerts,
        total_humans=total_humans,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        rows_json=rows_json,
    )


@app.route("/download_report")
def download_report():
    rows = read_alert_rows()
    records = [row[:5] for row in rows if len(row) >= 5]
    df = pd.DataFrame(records, columns=["Date", "Time", "Humans", "Level", "Snapshot"])
    df.to_excel(REPORT_XLSX, index=False)
    return send_file(str(REPORT_XLSX), as_attachment=True, download_name="UAV_Report.xlsx")


# =========================================================
# 5. LIVE HUMAN DETECTION
# =========================================================
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Camera could not be opened. Check the camera connection.")

    font = cv2.FONT_HERSHEY_SIMPLEX
    last_alert = 0.0
    alert_gap = 1.0
    last_human_detected = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.35, verbose=False)
            human_count = 0

            for box in results[0].boxes:
                cls = int(box.cls[0])

                # Class 0 is person for a COCO-trained model.
                if cls != 0:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1

                # Keep the project's existing detection filters.
                if width < 10 or height < 20:
                    continue
                if width > 120 or height > 200:
                    continue
                if height / max(width, 1) < 1.2:
                    continue

                human_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Human {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    font,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            current_time = time.time()

            if human_count > 0 and (current_time - last_alert > alert_gap):
                level = alert_level(human_count)

                # Play a sound only when a new detection event starts.
                if not last_human_detected and winsound is not None:
                    try:
                        if level == "LOW":
                            winsound.Beep(800, 200)
                        elif level == "MEDIUM":
                            winsound.Beep(1200, 300)
                            winsound.Beep(1200, 300)
                        else:
                            winsound.Beep(1800, 500)
                            winsound.Beep(1800, 500)
                            winsound.Beep(1800, 500)
                    except RuntimeError:
                        pass

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"snapshot_{timestamp}.jpg"
                filepath = SNAPSHOT_DIR / filename

                # Save the image first; only record an alert if saving succeeds.
                if cv2.imwrite(str(filepath), frame):
                    print(f"Snapshot saved: {filepath}")
                    now = datetime.now()
                    append_alert(
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        human_count,
                        level,
                        filename,
                    )
                else:
                    print(f"ERROR: Could not save snapshot: {filepath}")

                last_alert = current_time
                last_human_detected = True

            if human_count == 0:
                last_human_detected = False

            if human_count == 0:
                display_level = "NONE"
            else:
                display_level = alert_level(human_count)

            cv2.putText(
                frame,
                f"Human Count: {human_count}",
                (20, 80),
                font,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Alert: {display_level}",
                (20, 120),
                font,
                1,
                (0, 255, 255),
                2,
            )

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        cap.release()


# =========================================================
# 6. LIVE STREAM ROUTE
# =========================================================
@app.route("/live")
def live():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    ensure_alert_csv()
    print(f"Project folder : {BASE_DIR}")
    print(f"Snapshots folder: {SNAPSHOT_DIR}")
    print(f"Alerts CSV      : {ALERTS_CSV}")
    app.run(debug=True)
