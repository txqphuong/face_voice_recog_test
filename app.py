import streamlit as st
import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from PIL import Image

# ========== Config ==========
EMBEDDING_PATH = "face_db.pkl"
SAVE_DIR = "faces"
THRESHOLD = 0.5

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Initialize FaceAnalysis ==========
@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    return app

face_app = load_face_app()

# ========== Load Face Database ==========
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
st.title("Real-time Face Recognition with Streamlit & InsightFace")

source = st.radio("Input source", ["Upload Image", "Webcam Snapshot"])

unknowns = []  # Store unknown faces with embeddings

def recognize_faces(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    labels = []

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        score = face.det_score
        if score < THRESHOLD:
            continue

        embedding = face.embedding
        label = "Unknown"
        best_score = 0
        for name, db_emb in face_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > THRESHOLD and sim > best_score:
                label = f"{name} ({sim:.2f})"
                best_score = sim

        if label == "Unknown":
            # Append unknown faces for user input later
            unknowns.append((image[y1:y2, x1:x2], embedding))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        labels.append(label)
    return image, labels

image_bgr = None

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif source == "Webcam Snapshot":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image_pil = Image.open(picture)
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

if image_bgr is not None:
    result_img, labels = recognize_faces(image_bgr)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")

    if unknowns:
        st.subheader("Unknown Faces Found")
        for idx, (face_img, emb) in enumerate(unknowns):
            st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), width=150, caption=f"Unknown #{idx + 1}")
            name = st.text_input(f"Enter name for Unknown #{idx + 1}", key=f"name_{idx}")
            if st.button(f"Save Unknown #{idx + 1}", key=f"save_{idx}"):
                if name.strip() == "":
                    st.warning("Please enter a valid name.")
                else:
                    filename = f"{name.strip().replace(' ', '_')}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(filepath, face_img)
                    face_db[name.strip()] = emb
                    save_face_db(face_db)
                    st.success(f"Saved {name} to database!")
                    # Remove saved unknown face from list (optional)
                    unknowns.pop(idx)
