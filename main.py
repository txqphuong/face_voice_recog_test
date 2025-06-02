import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import os
import pickle

# ========== Config ==========
SCRFD_MODEL_PATH = "/Users/phuongtxq/Desktop/test/scrfd_person_2.5g.onnx"
EMBEDDING_PATH = "face_db.pkl"
SAVE_DIR = "faces"
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Initialize Detectors ==========
detector = SCRFD(SCRFD_MODEL_PATH)
detector.prepare(ctx_id=0, input_size=(640, 640))

face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# ========== Load Face Database ==========
if os.path.exists(EMBEDDING_PATH):
    with open(EMBEDDING_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}

# ========== Similarity Function ==========
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ========== Start Camera ==========
cap = cv2.VideoCapture(0)

unknown_face = None
unknown_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    outputs = detector.detect(img_rgb)

    bboxes = outputs[0] if isinstance(outputs, tuple) else outputs

    unknown_face = None
    unknown_embedding = None

    for bbox in bboxes:
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(int, bbox[:4])
        score = bbox[4] if len(bbox) == 5 else 1.0

        # Skip invalid box
        if score < THRESHOLD or x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        # Crop and validate person region
        person_crop_rgb = img_rgb[y1:y2, x1:x2]
        person_crop_bgr = frame[y1:y2, x1:x2]

        if person_crop_rgb.size == 0 or person_crop_rgb.shape[0] == 0 or person_crop_rgb.shape[1] == 0:
            continue

        faces = face_app.get(person_crop_rgb)

        label = "No face"
        best_score = 0
        matched_name = "Unknown"

        if faces:
            face = faces[0]
            embedding = face.embedding

            for name, db_emb in face_db.items():
                sim = cosine_similarity(embedding, db_emb)
                if sim > THRESHOLD and sim > best_score:
                    matched_name = name
                    best_score = sim

            if matched_name != "Unknown":
                label = f"{matched_name} ({best_score:.2f})"
            else:
                label = "Unknown"
                unknown_face = person_crop_bgr
                unknown_embedding = embedding

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save new face with label
    if key == ord('y') and unknown_face is not None:
        name = input("Enter name for this person: ")
        if name.strip() == "":
            print("Invalid name. Skipped saving.")
            continue

        filename = f"{name.strip().replace(' ', '_')}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, unknown_face)
        face_db[name] = unknown_embedding
        with open(EMBEDDING_PATH, "wb") as f:
            pickle.dump(face_db, f)
        print(f"Saved new face as {name} to database and image as {filepath}")

    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyWindow("Face Recognition")
