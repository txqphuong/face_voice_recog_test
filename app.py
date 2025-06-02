import streamlit as st
import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import os
import pickle
from PIL import Image

# ========== Config ==========
SCRFD_MODEL_PATH = "/Users/phuongtxq/Desktop/test/scrfd_person_2.5g.onnx"
EMBEDDING_PATH = "face_db.pkl"
SAVE_DIR = "faces"
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Load Models ==========
@st.cache_resource
def load_models():
    detector = SCRFD(SCRFD_MODEL_PATH)
    detector.prepare(ctx_id=0, input_size=(640, 640))
    face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0)
    return detector, face_app

detector, face_app = load_models()

# ========== Load Database ==========
def load_face_db():
    if os.path.exists(EMBEDDING_PATH):
        with open(EMBEDDING_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_face_db(db):
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(db, f)

face_db = load_face_db()

# ========== Similarity Function ==========
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ========== Streamlit UI ==========
st.title("Face Recognition with SCRFD + ArcFace")

source = st.radio("Choose Input Source", ["Upload Image", "Webcam (snapshot)"])

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
elif source == "Webcam (snapshot)":
    picture = st.camera_input("Take a picture")
    if picture:
        image_bgr = cv2.imdecode(np.frombuffer(picture.read(), np.uint8), cv2.IMREAD_COLOR)

if 'image_bgr' in locals():
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    outputs = detector.detect(img_rgb)
    bboxes = outputs[0] if isinstance(outputs, tuple) else outputs

    unknowns = []

    for bbox in bboxes:
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = map(int, bbox[:4])
        score = bbox[4] if len(bbox) == 5 else 1.0
        if score < THRESHOLD:
            continue

        person_rgb = img_rgb[y1:y2, x1:x2]
        person_bgr = image_bgr[y1:y2, x1:x2]

        if person_rgb.size == 0:
            continue

        faces = face_app.get(person_rgb)

        label = "No face"
        matched_name = "Unknown"
        best_score = 0

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
                unknowns.append((person_bgr, embedding))
                label = "Unknown"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), channels="RGB")

    # Save new unknowns
    if unknowns:
        st.subheader("Unknown Face(s) Found")
        for idx, (face_img, emb) in enumerate(unknowns):
            st.image(face_img, caption=f"Unknown #{idx+1}", width=150)
            name = st.text_input(f"Enter name for Unknown #{idx+1}", key=f"name_{idx}")
            if st.button(f"Save Unknown #{idx+1}", key=f"btn_{idx}"):
                if name.strip() == "":
                    st.warning("Please enter a valid name.")
                else:
                    filename = f"{name.strip().replace(' ', '_')}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(filepath, face_img)
                    face_db[name] = emb
                    save_face_db(face_db)
                    st.success(f"Saved {name} to database!")

