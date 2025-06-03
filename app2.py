import streamlit as st
import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from PIL import Image
import uuid

# ========== Config ==========
EMBEDDING_PATH = "face_db.pkl"
SAVE_DIR = "faces"
THRESHOLD = 0.5
DUPLICATE_THRESHOLD = 0.8  # To filter same face twice

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

# ========== Session States ==========
if "unknown_counter" not in st.session_state:
    st.session_state.unknown_counter = 1
if "unknowns" not in st.session_state:
    st.session_state.unknowns = []

# ========== Streamlit UI ==========
st.title("Real-time Face Recognition with Streamlit & InsightFace")

source = st.radio("Input source", ["Upload Image", "Webcam Snapshot"])

def recognize_faces(image, save_unknown=False):
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
        matched_name = None

        # Check against known faces
        for name, db_emb in face_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > THRESHOLD and sim > best_score:
                best_score = sim
                matched_name = name

        if matched_name:
            label = f"{matched_name}_{str(uuid.uuid4())[:5]}"
        else:
            # Avoid adding duplicate unknowns
            is_duplicate = False
            for _, existing_emb, _ in st.session_state.unknowns:
                if cosine_similarity(embedding, existing_emb) > DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break

            label = f"Unknown_{st.session_state.unknown_counter:03d}"
            if save_unknown and not is_duplicate:
                st.session_state.unknowns.append((image[y1:y2, x1:x2], embedding, label))
                st.session_state.unknown_counter += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        labels.append(label)
    return image, labels

image_bgr = None

# ========== Image Input ==========
if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif source == "Webcam Snapshot":
    picture = st.camera_input("Take a picture")  # bounding boxes shown after snapshot
    if picture is not None:
        image_pil = Image.open(picture)
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# ========== Face Recognition ==========
if image_bgr is not None:
    result_img, labels = recognize_faces(image_bgr, save_unknown=True)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")

# ========== Handle Unknowns ==========
if st.session_state.unknowns:
    st.subheader("Create Profile for Unknown Faces")

    for idx, (face_img, emb, auto_label) in enumerate(st.session_state.unknowns.copy()):
        st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), width=150, caption=auto_label)
        name = st.text_input(f"Enter name for {auto_label}", key=f"name_{idx}")
        if st.button(f"Create Profile for {auto_label}", key=f"save_{idx}"):
            if name.strip() == "":
                st.warning("Please enter a valid name.")
            else:
                final_name = f"{name.strip().replace(' ', '_')}_{auto_label.split('_')[-1]}"
                filename = f"{final_name}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, face_img)
                face_db[name.strip()] = emb
                save_face_db(face_db)
                st.success(f"Saved {final_name} to database!")

                # Remove from list
                st.session_state.unknowns.pop(idx)
                st.rerun()  # Refresh UI after update
