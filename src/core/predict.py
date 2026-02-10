import torch
import numpy as np
import nibabel as nib
import gc
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

from src.core.config import ROI_SIZE, SPACING, DEVICE, MODEL_PATH
from src.core.arch import LiverSegModel
from src.core.transforms import get_inference_transforms


def load_brain():
    print(f"Loading Model on: {DEVICE}")
    model = LiverSegModel().to(DEVICE)

    try:
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Failed to load brain: {e}")
        return None


def predict_volume(model, image_path):
    transforms = get_inference_transforms()

    files = [{"image": image_path}]
    ds = Dataset(data=files, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    result = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            outputs = sliding_window_inference(
                inputs=images,
                roi_size=ROI_SIZE,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

            del outputs
            del images
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            voxel_vol_mm3 = SPACING[0] * SPACING[1] * SPACING[2]
            voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

            liver_count = (preds == 1).sum()
            tumor_count = (preds == 2).sum()

            liver_vol = liver_count * voxel_vol_cm3
            tumor_vol = tumor_count * voxel_vol_cm3

            pct = 0
            if (liver_vol + tumor_vol) > 0:
                pct = (tumor_vol / (liver_vol + tumor_vol)) * 100

            save_path = None
            try:
                orig_nifti = nib.load(image_path)

                mask_nifti = nib.Nifti1Image(
                    preds.astype(np.uint8), orig_nifti.affine, orig_nifti.header
                )

                save_path = (
                    image_path.replace(".nii.gz", "").replace(".nii", "")
                    + "_seg.nii.gz"
                )
                nib.save(mask_nifti, save_path)
                print(f"Saved mask to: {save_path}")

                del orig_nifti
                del mask_nifti
                gc.collect()

            except Exception as e:
                print(f"Error saving mask: {e}")

            result = {
                "liver_volume_cm3": round(float(liver_vol), 2),
                "tumor_volume_cm3": round(float(tumor_vol), 2),
                "tumor_percentage": round(float(pct), 2),
                "mask_path": save_path,
            }
            break

    return result
