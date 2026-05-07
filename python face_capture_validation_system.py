import cv2
import os
import numpy as np
from PIL import Image

# Create folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Initialize face detector and recognizer
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ==============================
# STEP 1 — FACE CAPTURE
# ==============================
def capture_faces():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    user_name = input("\nEnter User Name for Face Capture: ").strip()
    user_folder = os.path.join("dataset", user_name)
    os.makedirs(user_folder, exist_ok=True)

    print("\n[INFO] Capturing faces. Look at the camera... Press 'q' to stop.")
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Camera not accessible.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(user_folder, f"{user_name}_{count}.jpg"), face_img)
            cv2.imshow("Face Capture", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= 30:
            break

    print(f"\n[INFO] Collected {count} face samples for '{user_name}'.")
    cam.release()
    cv2.destroyAllWindows()
    return user_name

# ==============================
# STEP 2 — TRAINING THE MODEL
# ==============================
def train_model():
    print("\n[INFO] Training the model...")

    faces = []
    ids = []

    for folder in os.listdir("dataset"):
        folder_path = os.path.join("dataset", folder)
        if not os.path.isdir(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            gray_img = Image.open(img_path).convert("L")
            img_numpy = np.array(gray_img, "uint8")

            face_id = folder
            detected_faces = face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in detected_faces:
                faces.append(img_numpy[y:y + h, x:x + w])
                ids.append(face_id)

    # Map names to integer IDs
    unique_ids = list(set(ids))
    label_to_id = {name: i for i, name in enumerate(unique_ids)}
    id_to_label = {i: name for name, i in label_to_id.items()}

    # Train recognizer
    recognizer.train(faces, np.array([label_to_id[name] for name in ids]))

    # Save model and labels
    recognizer.save("trainer/trainer.yml")
    np.save("trainer/labels.npy", id_to_label)

    print(f"[INFO] Training complete. Model saved with {len(unique_ids)} user(s).")

# ==============================
# STEP 3 — VALIDATION / RECOGNITION
# ==============================
def validate_faces():
    print("\n[INFO] Starting Face Validation... Press 'q' to quit.")
    recognizer.read("trainer/trainer.yml")
    id_to_label = np.load("trainer/labels.npy", allow_pickle=True).item()

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(face_roi)

            if confidence < 60:
                name = id_to_label[label_id]
                color = (0, 255, 0)
                label = f"{name} ({100 - confidence:.0f}%)"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x + 5, y - 5), font, 0.8, color, 2)

        cv2.imshow("Face Validation System", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Validation ended.")

# ==============================
# MAIN FLOW
# ==============================
if __name__ == "__main__":
    print("\n==== FACE CAPTURE VALIDATION SYSTEM ====")
    print("1. Capture New Faces")
    print("2. Train Model")
    print("3. Validate Faces (Recognition)")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_model()
        elif choice == '3':
            validate_faces()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
