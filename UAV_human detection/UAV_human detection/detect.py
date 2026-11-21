from ultralytics import YOLO
import cv2
import time
import platform
import os

# -------------------------------
# 1. Optional sound setup
# -------------------------------
system_os = platform.system()
if system_os == "Windows":
    import winsound  # Built-in Windows sound module
else:
    from playsound import playsound  # Use .mp3/.wav on Linux/Mac

# -------------------------------
# 2. Load Pretrained Model (COCO dataset)
# -------------------------------
# model = YOLO("yolov8m.pt")  # You can also use yolov8s.pt for better accuracy
model = YOLO(r"C:\Users\admin\Desktop\drone_human_detection\clg_project\best1.pt")
#model = YOLO("yolov8m.pt")
# model = YOLO("model = YOLO("/content/drive/MyDrive/Colab Notebooks/yolo/ResplitDataset/ResplitDataset/Training_Results/Resplit_human_detection/weights/best.pt")")

# -------------------------------
# 3. Video Capture Source
# -------------------------------
# For webcam use 0
# For ESP32-CAM stream, replace with: "http://10.87.0.49:81/stream"
# url = "http://10.140.120.87:81/stream"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0)
 #cap = cv2.VideoCapture("http://100.73.210.65:8080/video")

if not cap.isOpened():
    print("❌ Camera not detected. Check connection or IP address.")
    exit()

print("✅ Camera connected. Starting YOLOv8 Human Detection...")

# -------------------------------
# 4. Alert Settings 
# -------------------------------
last_alert_time = 0
alert_interval = 5  # seconds between alert sounds
font = cv2.FONT_HERSHEY_SIMPLEX

# Create snapshots folder
snapshots_folder = "snapshots"
if not os.path.exists(snapshots_folder):
    os.makedirs(snapshots_folder)

# -------------------------------
# 5. Main Detection Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Check camera.")
        break

    # Run YOLOv8 model
    results = model(frame, conf=0.45, verbose=False)

    human_count = 0

    # Draw boxes for detected humans
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Class 0 = Person in COCO dataset
            human_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (x1, y1 - 10), font, 0.7, (0, 255, 0), 2)

    # -------------------------------
    # 6. Alert System (Sound + Snapshot + Console)
    # -------------------------------
    if human_count > 0:
        current_time = time.time()
        if current_time - last_alert_time > alert_interval:
            print(f"🚨 ALERT: {human_count} human(s) detected!")

            # Play alert sound
            if system_os == "Windows":
                winsound.Beep(1000, 600)  # frequency=1000Hz, duration=600ms
            else:
                playsound("alert.wav")  # Add alert.wav in same folder

            # Save snapshot
            snapshot_filename = os.path.join(
                snapshots_folder, f"snapshot_{int(current_time)}.jpg"
            )
            cv2.imwrite(snapshot_filename, frame)
            print(f"📸 Snapshot saved as {snapshot_filename}")

            last_alert_time = current_time

    # -------------------------------
    # 7. Display Information on Screen
    # -------------------------------
    cv2.putText(frame, f"Human Count: {human_count}", (20, 40), font, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Press 'Q' to Quit", (20, 80), font, 0.7, (200, 200, 200), 2)

    cv2.imshow("🧠 YOLOv8 Human Detection System", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        print("👋 Exiting detection...")
        break

# -------------------------------
# 8. Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped successfully.")