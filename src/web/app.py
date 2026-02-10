import streamlit as st
from streamlit_image_comparison import image_comparison
import requests
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

st.set_page_config(page_title="LiverAI Analysis", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Liver Tumor Segmentation & Volumetrics")
st.markdown("Automated Clinical Decision Support System")
st.divider()

with st.sidebar:
    st.header("System Status")
    try:
        requests.get("http://localhost:8000/scans/")
        st.success("Backend: Connected")
    except:
        st.error("Backend: Disconnected")

    st.divider()
    st.info(
        "This system uses a 3D ResUNet to segment liver and tumor tissue from NIfTI volumes."
    )

col_upload, col_results = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("Upload Patient Scan")
    uploaded_file = st.file_uploader("Upload .nii or .nii.gz", type=["nii", "gz"])

    if uploaded_file:
        if st.button("Run AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("AI is counting voxels..."):
                try:
                    files = {
                        "my_upload": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/octet-stream",
                        )
                    }
                    response = requests.post(
                        "http://localhost:8000/upload/", files=files, timeout=600
                    )

                    if response.status_code == 200:
                        st.session_state["last_result"] = response.json()
                        st.success("Analysis Complete!")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

with col_results:
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]

        st.subheader("Analysis Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Liver Volume", f"{res['liver_volume_cm3']} cm³")
        m2.metric("Tumor Volume", f"{res['tumor_volume_cm3']} cm³")

        pct = res["tumor_percentage"]
        delta_color = "normal" if pct < 5 else "inverse"
        m3.metric("Tumor Burden", f"{pct}%", delta=f"{pct}%", delta_color=delta_color)

        st.divider()

        st.markdown("### Clinical Recommendation")
        proc = res["procedure"]

        if "Surgical" in proc:
            st.warning(f"**Recommended:** {proc}")
        elif "Transplant" in proc:
            st.error(f"**Recommended:** {proc}")
        else:
            st.success(f"**Recommended:** {proc}")

        with st.expander("View Raw Data"):
            st.json(res)
    else:
        st.info("Upload and process a scan to see the volumetric breakdown here.")

st.divider()

if "last_result" in st.session_state:
    st.divider()
    st.subheader("Interactive 3D Visualization")

    res = st.session_state["last_result"]
    original_path = res.get("original_file_path")
    mask_path = res.get("mask_file_path")

    if original_path and os.path.exists(original_path):
        try:

            @st.cache_data
            def load_nifti(path):
                img = nib.load(path)
                img = nib.as_closest_canonical(img)
                spacing = img.header.get_zooms()[:3]
                return img.get_fdata(), spacing

            vol_orig, spacing = load_nifti(original_path)

            if mask_path and os.path.exists(mask_path):
                vol_mask, _ = load_nifti(mask_path)
            else:
                vol_mask = np.zeros_like(vol_orig)

            view_option = st.radio(
                "Select View Plane:",
                ["Axial (Top-Down)", "Coronal (Front-Back)", "Sagittal (Side-View)"],
                horizontal=True,
                key="view_selection",
            )

            if view_option == "Axial (Top-Down)":
                display_vol = vol_orig
                display_mask = vol_mask
                aspect_ratio = spacing[1] / spacing[0]

            elif view_option == "Coronal (Front-Back)":
                display_vol = np.transpose(vol_orig, (0, 2, 1))
                display_mask = np.transpose(vol_mask, (0, 2, 1))
                aspect_ratio = spacing[2] / spacing[0]

            elif view_option == "Sagittal (Side-View)":
                display_vol = np.transpose(vol_orig, (1, 2, 0))
                display_mask = np.transpose(vol_mask, (1, 2, 0))
                aspect_ratio = spacing[2] / spacing[1]

            max_slice = display_vol.shape[2] - 1
            if "slice_idx" not in st.session_state:
                st.session_state.slice_idx = max_slice // 2

            current_slice = st.slider(
                "Slice Index",
                0,
                max_slice,
                min(st.session_state.slice_idx, max_slice),
                key="slice_slider",
            )
            st.session_state.slice_idx = current_slice

            img_slice = display_vol[:, :, current_slice]

            mask_depth = display_mask.shape[2]
            vol_depth = display_vol.shape[2]

            if mask_depth != vol_depth:
                ratio = mask_depth / vol_depth
                mask_idx = int(current_slice * ratio)
                mask_idx = min(mask_idx, mask_depth - 1)
                mask_slice = display_mask[:, :, mask_idx]
            else:
                mask_slice = display_mask[:, :, current_slice]

            if mask_slice.shape != img_slice.shape:
                zoom_factors = np.array(img_slice.shape) / np.array(mask_slice.shape)
                mask_slice = scipy.ndimage.zoom(mask_slice, zoom_factors, order=0)

            img_slice = np.rot90(img_slice)
            mask_slice = np.rot90(mask_slice)

            if aspect_ratio != 1.0:
                img_slice = scipy.ndimage.zoom(img_slice, (aspect_ratio, 1.0), order=1)
                mask_slice = scipy.ndimage.zoom(
                    mask_slice, (aspect_ratio, 1.0), order=0
                )

            img_display = np.clip(img_slice, -100, 400)
            img_display = (
                (img_display - img_display.min())
                / (img_display.max() - img_display.min())
                * 255
            ).astype(np.uint8)

            img_overlay = np.stack([img_display] * 3, axis=-1)

            img_overlay[mask_slice == 1, 1] = 255
            img_overlay[mask_slice == 1, 0] = 0
            img_overlay[mask_slice == 1, 2] = 0

            img_overlay[mask_slice == 2, 0] = 255
            img_overlay[mask_slice == 2, 1] = 0
            img_overlay[mask_slice == 2, 2] = 0

            image_comparison(
                img1=img_display,
                img2=img_overlay,
                label1="Original Scan",
                label2="AI Prediction",
                width=700,
                in_memory=True,
            )

        except Exception as e:
            st.error(f"Visualization Error: {e}")

st.divider()
st.subheader("Recent Analysis History")

try:
    history_response = requests.get("http://localhost:8000/scans/")

    if history_response.status_code == 200:
        history_data = history_response.json()

        if history_data:
            import pandas as pd

            df = pd.DataFrame(history_data)

            display_df = df[
                [
                    "id",
                    "patient_id",
                    "status",
                    "liver_volume_cm3",
                    "tumor_percentage",
                    "procedure",
                    "upload_time",
                ]
            ].copy()

            display_df["upload_time"] = pd.to_datetime(
                display_df["upload_time"]
            ).dt.strftime("%Y-%m-%d %H:%M")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tumor_percentage": st.column_config.NumberColumn(
                        "Tumor %", format="%.2f%%"
                    ),
                    "liver_volume_cm3": st.column_config.NumberColumn(
                        "Liver Vol (cm³)", format="%.1f"
                    ),
                },
            )
        else:
            st.info("No history found in the database yet.")
    else:
        st.error("Could not fetch history from the server.")
except Exception as e:
    st.caption(f"History unavailable: {e}")
