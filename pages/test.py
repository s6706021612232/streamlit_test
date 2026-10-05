import streamlit as st
import os
import cv2
import av

from ultralytics import YOLO
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ===========================================================================
# PAGE SETTINGS
# ===========================================================================

st.set_page_config(
    page_title="Water Bottle Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# MODEL SETTINGS
# ===========================================================================

MODEL_DIR = Path("model")

model_files = sorted(
    MODEL_DIR.glob("*.pt")
)

if not model_files:

    st.error(
        "ไม่พบไฟล์โมเดล .pt ในโฟลเดอร์ model/"
    )

    st.stop()


# ===========================================================================
# MAIN HEADER
# ===========================================================================

st.title(
    "💧 ระบบตรวจจับขวดน้ำด้วย YOLO"
)

st.caption(
    "ระบบตรวจจับและจำแนกประเภทขวดน้ำจากวิดีโอและเว็บแคม"
)

st.markdown("---")


col_model, col_info, col_f1 = st.columns(
    [2, 1, 1]
)

with col_model:

    st.markdown(
        "### 🤖 เลือกโมเดล"
    )

    selected_model = st.selectbox(
        "Model",
        model_files,
        format_func=lambda x: x.stem,
        label_visibility="collapsed",
    )


with col_info:

    st.markdown(
        "### 📦 โมเดลที่เลือก"
    )

    st.info(
        selected_model.name
    )


# ===========================================================================
# F1 CONFIDENCE STATISTICS
# ===========================================================================

F1_STATS = {
    "yolo8n": (0.93, 0.279),
    "yolo11n": (0.93, 0.577),
    "yolo12n": (0.93, 0.649),
    "yolo26n": (0.93, 0.674),
}


with col_f1:

    model_key = selected_model.stem.lower()

    if model_key in F1_STATS:

        f1, confidence = F1_STATS[model_key]

        st.markdown(
            "### 📊 F1 Confidence"
        )

        st.info(
            f"All classes {f1:.2f} at {confidence:.3f}"
        )



# ===========================================================================
# LOAD MODEL
# ===========================================================================

@st.cache_resource(
    show_spinner="กำลังโหลดโมเดล YOLO..."
)
def load_model(model_path):

    return YOLO(
        str(model_path)
    )


model = load_model(
    selected_model
)

class_names = model.model.names


# ===========================================================================
# CLASS NAME HELPER
# ===========================================================================

def get_class_name(class_id):

    if isinstance(class_names, dict):

        return class_names.get(
            class_id,
            str(class_id)
        )

    return class_names[class_id]


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:

    st.markdown(
        "## 💧 Water Bottle Detection"
    )

    st.markdown(
        "ระบบตรวจจับขวดน้ำด้วย **YOLO**"
    )

    st.markdown("---")

    st.markdown(
        "### 🏷️ ประเภทขวดที่รองรับ"
    )

    if isinstance(class_names, dict):

        for cls_name in class_names.values():

            st.markdown(
                f"- {cls_name}"
            )

    else:

        for cls_name in class_names:

            st.markdown(
                f"- {cls_name}"
            )

    st.markdown("---")

    st.caption(
        f"โมเดล: `{selected_model.name}`"
    )

    st.caption(
        "ไฟล์อัปโหลดจะถูกเก็บไว้ที่โฟลเดอร์ `upload/`"
    )


# ===========================================================================
# UPLOAD DIRECTORY
# ===========================================================================

UPLOAD_DIR = "upload"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ===========================================================================
# DETECTION SETTINGS
# ===========================================================================

BOX_COLOR = (
    216,
    138,
    61
)

LABEL_TEXT_COLOR = (
    11,
    15,
    25
)


# ===========================================================================
# DRAW DETECTIONS
# ===========================================================================

def draw_detections(
    frame,
    results
):

    if (
        not results
        or results[0].boxes is None
    ):

        return frame


    boxes = (
        results[0]
        .boxes
        .xyxy
        .int()
        .cpu()
        .tolist()
    )


    class_ids = (
        results[0]
        .boxes
        .cls
        .int()
        .cpu()
        .tolist()
    )


    confs = (
        results[0]
        .boxes
        .conf
        .cpu()
        .tolist()
    )


    if results[0].boxes.id is not None:

        track_ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

    else:

        track_ids = [
            None
        ] * len(boxes)


    for (
        box,
        class_id,
        track_id,
        conf
    ) in zip(
        boxes,
        class_ids,
        track_ids,
        confs
    ):

        x1, y1, x2, y2 = box

        class_name = get_class_name(
            class_id
        )


        # ===============================================================
        # GOOD / BAD BOTTLE
        # ===============================================================

        if class_name in [
            "good_bottle",
            "bad_bottle"
        ]:

            if class_name == "good_bottle":

                color = (
                    0,
                    200,
                    0
                )

            else:

                color = (
                    0,
                    0,
                    255
                )


            label = (
                f"{class_name} {conf:.2f}"
            )


            # -----------------------------------------------------------
            # Bounding box
            # -----------------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # -----------------------------------------------------------
            # Label above box
            # -----------------------------------------------------------

            (
                tw,
                th
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )


            label_y1 = max(
                y1 - th - 12,
                0
            )


            label_y2 = max(
                y1,
                th + 12
            )


            cv2.rectangle(
                frame,
                (x1, label_y1),
                (
                    x1 + tw + 8,
                    label_y2
                ),
                color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (
                    x1 + 4,
                    label_y2 - 6
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255
                ),
                2
            )


        # ===============================================================
        # HAVE CAP / NO CAP
        # ===============================================================

        elif class_name in [
            "have_cap",
            "no_cap"
        ]:

            if class_name == "have_cap":

                color = (
                    0,
                    180,
                    0
                )

            else:

                color = (
                    0,
                    0,
                    255
                )


            label = (
                f"{class_name} {conf:.2f}"
            )


            # -----------------------------------------------------------
            # Original cap bounding box
            # -----------------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # -----------------------------------------------------------
            # Label under cap box
            # -----------------------------------------------------------

            (
                tw,
                th
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )


            label_x = x1

            label_y1 = y2

            label_y2 = (
                y2 + th + 12
            )


            # -----------------------------------------------------------
            # If label goes outside image,
            # move it above the box.
            # -----------------------------------------------------------

            if label_y2 >= frame.shape[0]:

                label_y1 = max(
                    y1 - th - 12,
                    0
                )

                label_y2 = y1


            cv2.rectangle(
                frame,
                (
                    label_x,
                    label_y1
                ),
                (
                    label_x + tw + 8,
                    label_y2
                ),
                color,
                -1
            )


            cv2.putText(
                frame,
                label,
                (
                    label_x + 4,
                    label_y2 - 6
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (
                    255,
                    255,
                    255
                ),
                2
            )


    return frame


# ===========================================================================
# WEBCAM VIDEO PROCESSOR
# ===========================================================================

class YOLOVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.model = model

        self.conf = 0.25

        # Prevent multiple frames from being
        # processed at the same time.
        self.processing = False


    def recv(
        self,
        frame
    ):

        # ---------------------------------------------------------------
        # Browser laptop webcam -> OpenCV BGR
        # ---------------------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ---------------------------------------------------------------
        # Prevent frame-processing overload
        # ---------------------------------------------------------------

        if self.processing:

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )


        self.processing = True


        try:

            # -----------------------------------------------------------
            # YOLO Tracking
            # -----------------------------------------------------------

            results = self.model.track(
                img,
                persist=True,
                conf=self.conf,
                imgsz=640,
                verbose=False,
            )


            # -----------------------------------------------------------
            # Draw detections
            # -----------------------------------------------------------

            img = draw_detections(
                img,
                results
            )

        finally:

            self.processing = False


        # ---------------------------------------------------------------
        # OpenCV BGR -> Browser WebRTC
        # ---------------------------------------------------------------

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ===========================================================================
# TABS
# ===========================================================================

tab_video, tab_webcam = st.tabs(
    [
        "📹  อัปโหลดวิดีโอ",
        "🎥  เว็บแคมเรียลไทม์",
    ]
)


# ===========================================================================
# VIDEO TAB
# ===========================================================================

with tab_video:

    st.subheader(
        "อัปโหลดวิดีโอเพื่อตรวจจับขวดน้ำ"
    )


    col_upload, col_setting = st.columns(
        [2, 1]
    )


    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------

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


    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    with col_setting:

        conf_video = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.65,
            0.05,
            key="conf_video",
        )


        skip_frame = st.checkbox(
            "ข้ามเฟรมเพื่อเพิ่มความเร็ว",
            value=True,
            key="skip_video",
        )


    # -----------------------------------------------------------------------
    # Process uploaded video
    # -----------------------------------------------------------------------

    if uploaded_file is not None:

        save_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name,
        )


        with open(
            save_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )


        st.success(
            f"บันทึกไฟล์ไว้ที่ `{save_path}` เรียบร้อยแล้ว"
        )


        start_video = st.button(
            "▶️  เริ่มตรวจจับ",
            key="start_video",
            use_container_width=True,
        )


        if start_video:

            cap = cv2.VideoCapture(
                save_path
            )


            if not cap.isOpened():

                st.error(
                    "ไม่สามารถเปิดไฟล์วิดีโอได้"
                )

                st.stop()


            frame_placeholder = st.empty()

            progress_bar = st.progress(
                0
            )

            status_text = st.empty()


            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )


            if total_frames <= 0:

                total_frames = 1


            count = 0


            while cap.isOpened():

                ret, frame = cap.read()


                if not ret:

                    break


                count += 1


                if (
                    skip_frame
                    and count % 2 != 0
                ):

                    continue


                # -------------------------------------------------------
                # YOLO
                # -------------------------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    conf=conf_video,
                    imgsz=640,
                    verbose=False,
                )


                # -------------------------------------------------------
                # Draw
                # -------------------------------------------------------

                frame = draw_detections(
                    frame,
                    results,
                )


                # -------------------------------------------------------
                # BGR -> RGB
                # -------------------------------------------------------

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )


                # -------------------------------------------------------
                # Display
                # -------------------------------------------------------

                frame_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    width=400,
                )


                progress_bar.progress(
                    min(
                        count / total_frames,
                        1.0,
                    )
                )


                status_text.caption(
                    f"ประมวลผลเฟรมที่ "
                    f"{count} / {total_frames}"
                )


            cap.release()


            status_text.empty()

            progress_bar.empty()


            st.success(
                "✅ ตรวจจับวิดีโอเสร็จสิ้น"
            )


# ===========================================================================
# WEBCAM TAB
# ===========================================================================

with tab_webcam:

    st.subheader(
        "ตรวจจับขวดน้ำแบบเรียลไทม์จากเว็บแคม"
    )


    st.markdown(
        """
        เว็บแคมส่วนนี้ใช้ **WebRTC** แทน `cv2.VideoCapture(0)`
        ดังนั้นกล้องจะเป็นกล้องของ Browser ที่กำลังเปิด Streamlit
        """
    )


    # -----------------------------------------------------------------------
    # Confidence
    # -----------------------------------------------------------------------

    conf_cam = st.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.25,
        0.05,
        key="conf_cam",
    )


    st.markdown("---")


    # =========================================================================
    # WEBRTC
    # =========================================================================

    webrtc_ctx = webrtc_streamer(
        key="water-bottle-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=False,
    )


    # =========================================================================
    # UPDATE CONFIDENCE
    # =========================================================================

    if webrtc_ctx.video_processor:

        webrtc_ctx.video_processor.conf = conf_cam


    # =========================================================================
    # CAMERA STATUS
    # =========================================================================

    if webrtc_ctx.state.playing:

        st.success(
            "🟢 กล้องกำลังทำงาน"
        )

    else:

        st.info(
            "กดปุ่ม START ด้านบนเพื่อเปิดเว็บแคม "
            "และกด Allow เมื่อ Browser ขอสิทธิ์ใช้กล้อง"
        )


    st.caption(
        "หมายเหตุ: Browser ต้องได้รับอนุญาตให้เข้าถึงกล้อง "
        "และหากนำไปใช้งานออนไลน์ แนะนำให้เปิดผ่าน HTTPS"
    )
