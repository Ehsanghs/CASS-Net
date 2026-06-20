import re
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class AISDataset(Dataset):
    """Load AISD slices and construct four-channel 2.5D inputs."""

    def __init__(
        self,
        data_dir,
        patient_ids,
        transform=None,
        hu_clip=(-100.0, 300.0),
        hu_divisor=100.0,
        stroke_window=(40.0, 80.0),
        brain_window=(80.0, 200.0),
        lesion_labels=(1, 2, 3, 5),
        return_metadata=False,
        mode="dataset",
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.patient_ids = [str(patient_id).strip() for patient_id in patient_ids]
        self.transform = transform
        self.hu_min, self.hu_max = map(float, hu_clip)
        self.hu_divisor = float(hu_divisor)
        self.stroke_window = tuple(map(float, stroke_window))
        self.brain_window = tuple(map(float, brain_window))
        self.lesion_labels = tuple(int(label) for label in lesion_labels)
        self.return_metadata = return_metadata
        self.mode = mode

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        if not self.patient_ids:
            raise ValueError("patient_ids is empty.")

        if len(self.patient_ids) != len(set(self.patient_ids)):
            raise ValueError("patient_ids contains duplicate values.")

        self.samples = self._prepare_samples()

    @staticmethod
    def _mask_instance(mask_path):
        match = re.fullmatch(r"(\d+)\.png", mask_path.name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    @staticmethod
    def _dicom_header(dicom_path):
        try:
            dcm = pydicom.dcmread(
                str(dicom_path),
                specific_tags=["InstanceNumber", "SliceLocation"],
                stop_before_pixels=True,
            )
        except Exception:
            return None

        if "InstanceNumber" not in dcm or "SliceLocation" not in dcm:
            return None

        return {
            "path": dicom_path,
            "instance": int(dcm.InstanceNumber),
            "slice_location": float(dcm.SliceLocation),
        }

    def _prepare_samples(self):
        samples = []
        skipped = 0
        represented_patients = 0

        for patient_id in self.patient_ids:
            dicom_dir = self.images_dir / patient_id / "CT"
            mask_dir = self.masks_dir / patient_id

            if not dicom_dir.is_dir() or not mask_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing image or mask directory for patient {patient_id}."
                )

            masks = {}
            for mask_path in mask_dir.glob("*.png"):
                instance = self._mask_instance(mask_path)
                if instance is not None:
                    masks[instance] = mask_path

            dicom_records = []
            for dicom_path in dicom_dir.glob("*.dcm"):
                record = self._dicom_header(dicom_path)
                if record is None:
                    skipped += 1
                    continue
                dicom_records.append(record)

            dicom_records.sort(key=lambda item: item["slice_location"])

            paired_records = []
            for record in dicom_records:
                mask_path = masks.get(record["instance"])
                if mask_path is None:
                    skipped += 1
                    continue

                paired_records.append(
                    {
                        "patient_id": patient_id,
                        "instance": record["instance"],
                        "dicom_path": record["path"],
                        "mask_path": mask_path,
                    }
                )

            if not paired_records:
                raise RuntimeError(
                    f"No valid DICOM-mask pairs found for patient {patient_id}."
                )

            records_by_instance = {
                record["instance"]: record for record in paired_records
            }

            for record in paired_records:
                instance = record["instance"]
                previous_record = records_by_instance.get(instance - 1, record)
                next_record = records_by_instance.get(instance + 1, record)

                samples.append(
                    {
                        "patient_id": patient_id,
                        "instance": instance,
                        "center_path": record["dicom_path"],
                        "previous_path": previous_record["dicom_path"],
                        "next_path": next_record["dicom_path"],
                        "mask_path": record["mask_path"],
                    }
                )

            represented_patients += 1

        if not samples:
            raise RuntimeError(f"No valid samples found for {self.mode}.")

        print(
            f"Found {len(samples)} verified pairs for {self.mode} "
            f"from {represented_patients} patients."
        )

        if skipped:
            print(f"Skipped {skipped} unpaired or incomplete DICOM records.")

        return samples

    def _load_dicom(self, dicom_path):
        try:
            dcm = pydicom.dcmread(str(dicom_path))
            image = dcm.pixel_array.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Could not read DICOM file: {dicom_path}") from exc

        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

        image = image * slope + intercept
        image = np.clip(image, self.hu_min, self.hu_max)
        image = image / self.hu_divisor

        return image, dcm

    def _apply_window(self, image, window):
        center, width = window
        center /= self.hu_divisor
        width /= self.hu_divisor

        lower = center - width / 2.0
        upper = center + width / 2.0

        image = np.clip(image, lower, upper)
        return ((image - lower) / (upper - lower + 1e-6)).astype(np.float32)

    def _load_mask(self, mask_path, expected_shape):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read mask file: {mask_path}")

        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape "
                f"{expected_shape} for {mask_path}."
            )

        return np.isin(mask, self.lesion_labels).astype(np.float32)

    @staticmethod
    def _image_tensor(image):
        if isinstance(image, torch.Tensor):
            return image.float()

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float()

    @staticmethod
    def _mask_tensor(mask):
        if isinstance(mask, torch.Tensor):
            tensor = mask.float()
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        center, center_dicom = self._load_dicom(sample["center_path"])
        previous, _ = self._load_dicom(sample["previous_path"])
        next_image, _ = self._load_dicom(sample["next_path"])

        if previous.shape != center.shape or next_image.shape != center.shape:
            raise ValueError(
                f"Adjacent slices have inconsistent shapes for patient "
                f"{sample['patient_id']}, instance {sample['instance']}."
            )

        image = np.stack(
            [
                self._apply_window(previous, self.stroke_window),
                self._apply_window(center, self.stroke_window),
                self._apply_window(center, self.brain_window),
                self._apply_window(next_image, self.stroke_window),
            ],
            axis=-1,
        )

        mask = self._load_mask(sample["mask_path"], center.shape)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = self._image_tensor(image)
        mask = self._mask_tensor(mask)

        if image.shape[0] != 4:
            raise ValueError(f"Expected 4 input channels, received {image.shape[0]}.")

        if image.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"Image and mask shapes differ after transformation: "
                f"{tuple(image.shape)} and {tuple(mask.shape)}."
            )

        result = {
            "image": image,
            "mask": mask,
        }

        if self.return_metadata:
            pixel_spacing = getattr(center_dicom, "PixelSpacing", [float("nan")] * 2)
            slice_thickness = float(
                getattr(center_dicom, "SliceThickness", float("nan"))
            )

            result.update(
                {
                    "patient_id": sample["patient_id"],
                    "instance_number": sample["instance"],
                    "pixel_spacing": torch.tensor(
                        [float(pixel_spacing[0]), float(pixel_spacing[1])],
                        dtype=torch.float32,
                    ),
                    "slice_thickness": torch.tensor(
                        slice_thickness,
                        dtype=torch.float32,
                    ),
                }
            )

        return result

    def __len__(self):
        return len(self.samples)
