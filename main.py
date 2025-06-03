import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# ========== Config ==========
EMBEDDING_PATH = "face_db.pkl"
SAVE_DIR = "faces"
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Initialize FaceAnalysis ==========
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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========== Start Camera ==========
cap = cv2.VideoCapture(0)

unknown_face = None
unknown_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect and recognize faces in one call
    faces = face_app.get(img_rgb)

    unknown_face = None
    unknown_embedding = None

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        score = face.det_score

        if score < THRESHOLD:
            continue

        embedding = face.embedding

        # Match with database
        label = "Unknown"
        best_score = 0
        for name, db_emb in face_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > THRESHOLD and sim > best_score:
                label = f"{name} ({sim:.2f})"
                best_score = sim

        if label == "Unknown":
            unknown_face = frame[y1:y2, x1:x2]
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
cv2.destroyAllWindows()
