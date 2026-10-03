import streamlit as st
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(
    page_title="Water Bottle Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

MODEL_DIR = Path("model")

# ค้นหาไฟล์โมเดล .pt ทั้งหมดในโฟลเดอร์ model
model_files = sorted(MODEL_DIR.glob("*.pt"))

if not model_files:
    st.error("ไม่พบไฟล์โมเดล .pt ในโฟลเดอร์ model/")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("## 💧 Water Bottle Detection")

    st.markdown(
        "ระบบตรวจจับขวดน้ำด้วย **YOLO**"
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Model selection
    # -----------------------------------------------------------------------

    st.markdown("### 🤖 เลือกโมเดล")

    selected_model = st.selectbox(
        "Model",
        model_files,
        format_func=lambda x: x.stem,
        label_visibility="collapsed",
    )

    st.markdown("---")


# ---------------------------------------------------------------------------
# Load selected model
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="กำลังโหลดโมเดล YOLO...")
def load_model(model_path):

    return YOLO(str(model_path))


model = load_model(selected_model)

class_names = model.model.names


# ---------------------------------------------------------------------------
# Sidebar - supported classes
# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("### 🏷️ ประเภทขวดที่รองรับ")

    if isinstance(class_names, dict):

        for cls_name in class_names.values():
            st.markdown(f"- {cls_name}")

    else:

        for cls_name in class_names:
            st.markdown(f"- {cls_name}")

    st.markdown("---")

    st.caption(
        f"โมเดล: `{selected_model.name}`"
    )

    st.caption(
        "ไฟล์อัปโหลดจะถูกเก็บไว้ที่โฟลเดอร์ `upload/`"
    )

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Upload directory
# ---------------------------------------------------------------------------

UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------

BOX_COLOR = (216, 138, 61)
LABEL_TEXT_COLOR = (11, 15, 25)


# ---------------------------------------------------------------------------
# Draw detections
# ---------------------------------------------------------------------------

def draw_detections(frame, results):
    """วาดกรอบ ชื่อคลาส และ confidence score ลงบนเฟรม"""

    if results[0].boxes is None:
        return frame

    boxes = results[0].boxes.xyxy.int().cpu().tolist()
    class_ids = results[0].boxes.cls.int().cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()

    # Track IDs อาจไม่มี ถ้าไม่ได้ใช้ tracking
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        track_ids = [None] * len(boxes)

    for box, class_id, track_id, conf in zip(
        boxes,
        class_ids,
        track_ids,
        confs
    ):
        x1, y1, x2, y2 = box

        class_name = class_names[class_id]

        if track_id is not None:
            label = f"#{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            2
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        # ป้องกัน label ล้นด้านบนของภาพ
        label_y1 = max(y1 - th - 12, 0)
        label_y2 = max(y1, th + 12)

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + tw + 8, label_y2),
            BOX_COLOR,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            LABEL_TEXT_COLOR,
            2
        )

    return frame


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("💧 ระบบตรวจจับขวดน้ำด้วย YOLO")

st.caption(
    "ระบบตรวจจับและจำแนกประเภทขวดน้ำจากวิดีโอและเว็บแคม"
)

st.markdown("---")



# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_video, tab_webcam = st.tabs(
    [
        "📹  อัปโหลดวิดีโอ",
        "🎥  เว็บแคมเรียลไทม์",
    ]
)


# ===========================================================================
# TAB 1: VIDEO
# ===========================================================================

with tab_video:

    st.subheader("อัปโหลดวิดีโอเพื่อตรวจจับขวดน้ำ")

    col_upload, col_setting = st.columns([2, 1])

    with col_upload:

        uploaded_file = st.file_uploader(
            "เลือกไฟล์วิดีโอ (mp4, avi, mov, mkv)",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
        )

    with col_setting:

        conf_video = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.4,
            0.05,
            key="conf_video",
        )

        skip_frame = st.checkbox(
            "ข้ามเฟรมเพื่อเพิ่มความเร็ว",
            value=True,
        )

    if uploaded_file is not None:

        save_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name,
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"บันทึกไฟล์ไว้ที่ `{save_path}` เรียบร้อยแล้ว"
        )

        start_video = st.button(
            "▶️  เริ่มตรวจจับ",
            key="start_video",
            use_container_width=True,
        )

        if start_video:

            cap = cv2.VideoCapture(save_path)

            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_frames = (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                or 1
            )

            count = 0

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                count += 1

                if skip_frame and count % 2 != 0:
                    continue

                frame = cv2.resize(
                    frame,
                    (640, 640),
                )

                results = model.track(
                    frame,
                    persist=True,
                    conf=conf_video,
                )

                frame = draw_detections(
                    frame,
                    results,
                )

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frame_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True,
                )

                progress_bar.progress(
                    min(
                        count / total_frames,
                        1.0,
                    )
                )

                status_text.caption(
                    f"ประมวลผลเฟรมที่ {count} / {total_frames}"
                )

            cap.release()

            status_text.empty()
            progress_bar.empty()

            st.success(
                "✅ ตรวจจับวิดีโอเสร็จสิ้น"
            )


# ===========================================================================
# TAB 2: WEBCAM
# ===========================================================================

with tab_webcam:

    st.subheader(
        "ตรวจจับขวดน้ำแบบเรียลไทม์จากเว็บแคม"
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:

        conf_cam = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.4,
            0.05,
            key="conf_cam",
        )

    with col_b:

        run_cam = st.checkbox(
            "🔴 เปิดกล้องเว็บแคม"
        )

    cam_placeholder = st.empty()

    st.caption(
        "หมายเหตุ: ฟีเจอร์เว็บแคมเหมาะสำหรับการรันแอปบนเครื่อง "
        "ที่มีกล้องเว็บแคมต่ออยู่"
    )

    if run_cam:

        cap = cv2.VideoCapture(0)

        count = 0

        while run_cam:

            ret, frame = cap.read()

            if not ret:

                st.error(
                    "ไม่พบเว็บแคม กรุณาตรวจสอบการเชื่อมต่อกล้อง"
                )

                break

            count += 1

            if count % 2 != 0:
                continue

            frame = cv2.resize(
                frame,
                (640, 640),
            )

            results = model.track(
                frame,
                persist=True,
                conf=conf_cam,
            )

            frame = draw_detections(
                frame,
                results,
            )

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            cam_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True,
            )

        cap.release()