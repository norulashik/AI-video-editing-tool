import streamlit as st
import os
import shutil
from types import SimpleNamespace
from rokey_pipeline import process_video_with_rokey

st.set_page_config(page_title="🔧 RotoKey - Streamlit App", layout="centered")

st.title("RocKey - AI Video Object Removal & Background Replacement")

# === Step 1: Upload video ===
uploaded_video = st.file_uploader("📹 Upload your video", type=["mp4", "mov", "avi"])

if uploaded_video:
    input_path = os.path.join("video_input", uploaded_video.name)
    os.makedirs("video_input", exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"✅ Uploaded: {uploaded_video.name}")

    # === Step 2: Options ===
    st.markdown("### ⚙️ Options")
    threshold = st.slider("Scene Detection Threshold", 10.0, 50.0, 30.0)
    gen_masks = st.checkbox("Generate SAM Masks", value=True)
    remove_obj = st.checkbox("Remove Object with OpenCV Inpainting", value=False)
    genai_prompt = st.text_input(
        "🧠 AI Background Prompt", value="cyberpunk city at night"
    )
    make_video = st.checkbox("Create Output Masked Video", value=False)

    # === Step 3: Run button ===
    if st.button("🚀 Run RotoKey Pipeline"):
        st.info("⏳ Processing... This may take a few minutes.")

        args = SimpleNamespace(
            video=input_path,
            out="keyframes",
            threshold=threshold,
            generate_masks=gen_masks,
            remove_mask=remove_obj,
            genai_bg=genai_prompt if genai_prompt.strip() else None,
            make_video=make_video,
        )

        process_video_with_rokey(args)

        st.success("✅ Done!")

        # === Step 4: Show example output ===
        st.markdown("### 🖼 Output Preview")
        example_dir = "output/genai_composite"
        if os.path.exists(example_dir):
            for file in sorted(os.listdir(example_dir)):
                if file.endswith(".jpg"):
                    st.image(
                        os.path.join(example_dir, file),
                        caption=file,
                        use_column_width=True,
                    )
                    break
        elif os.path.exists("removed"):
            for file in sorted(os.listdir("removed")):
                if file.endswith(".jpg"):
                    st.image(
                        os.path.join("removed", file),
                        caption=file,
                        use_column_width=True,
                    )
                    break
        else:
            st.warning("No composited or inpainted output found.")
