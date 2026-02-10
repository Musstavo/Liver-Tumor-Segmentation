from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    EnsureTyped,
)
from src.core.config import SPACING, ROI_SIZE


def get_inference_transforms():
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=SPACING,
                mode=("bilinear"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-100,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=ROI_SIZE),
            EnsureTyped(keys=["image"]),
        ]
    )
