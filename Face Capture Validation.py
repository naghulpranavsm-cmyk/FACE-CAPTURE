import cv2
import os

def create_directory(path):
    """Create directory if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def capture_faces():
    # Initialize the webcam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height

    # Load the OpenCV face detector (Haar Cascade)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Enter the name or ID for dataset creation
    face_id = input('\nEnter user ID or name: ')
    print("\n[INFO] Initializing face capture. Look at the camera and wait...")

    # Directory to store captured images
    dataset_path = "dataset"
    create_directory(dataset_path)
    user_folder = os.path.join(dataset_path, str(face_id))
    create_directory(user_folder)

    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image from camera. Exiting...")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

            # Save the captured image into the dataset folder
            file_path = os.path.join(user_folder, f"User_{face_id}_{count}.jpg")
            cv2.imwrite(file_path, gray[y:y + h, x:x + w])

            cv2.imshow('Face Capture - Validation System', img)

        # Press 'q' to quit or automatically stop after 30 samples
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= 30:
            print("\n[INFO] Successfully captured 30 face samples.")
            break

    # Cleanup
    print("\n[INFO] Exiting and cleaning up...")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()