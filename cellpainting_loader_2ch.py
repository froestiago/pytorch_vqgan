import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def read_image_paths(file_path):
    with open(file_path, 'r') as file:
        image_paths = file.read().splitlines()
    return image_paths

class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, names_file, images_root, split='train', to_normal=True):
        self.names_file = read_image_paths(names_file)
        self.images_root = images_root
        self.split = split
        self.to_normal = to_normal

        self.transforms = A.Compose([
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            ToTensorV2()
        ])

    def _normalize_image(self, image):
        # Normalize to [-1, 1] range assuming input is in [0, 255]
        # If your data has a different range, adjust accordingly
        norm_image = image / 127.5 - 1.0
        return norm_image

    def __len__(self):
        return len(self.names_file)

    def __getitem__(self, index):
        # get file name from the list
        image_name = self.names_file[index]

        # Load the already-concatenated image (assumed shape: C x H x W)
        img_path = os.path.join(self.images_root, image_name)
        img = np.load(img_path)
        img = np.array(img, dtype=np.float32)

        # add extra all zero channel to img
        # img = np.concatenate([img, np.zeros_like(img[:1])], axis=0)

        img = img.transpose(1, 2, 0)  # Convert to H x W x C format

        # Apply Albumentations (expects H x W x C)
        transformed = self.transforms(image=img)
        img = transformed["image"]

        # Normalize if required
        if self.to_normal:
            img = self._normalize_image(img)

        return {"images": img}
    

class PaintingDataset(torch.utils.data.Dataset):
    def __init__(self, names_file, images_root, split='train', to_normal=True):
        self.names_file = read_image_paths(names_file)
        self.images_root = images_root
        self.split = split
        self.to_normal = to_normal

        self.transforms = A.Compose([
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            ToTensorV2()
        ])

    def _normalize_image(self, image):
        # Normalize to [-1, 1] range assuming input is in [0, 255]
        # If your data has a different range, adjust accordingly
        norm_image = image / 127.5 - 1.0
        return norm_image

    def __len__(self):
        return len(self.names_file)

    def __getitem__(self, index):
        # get file name from the list
        image_name = self.names_file[index]

        # Load the already-concatenated image (assumed shape: C x H x W)
        brightfield_path = os.path.join(self.images_root, f"brightfield_{image_name}")
        brightfield = np.load(brightfield_path)
        brightfield = np.array(brightfield, dtype=np.float32)
        # brightfield = np.concatenate([brightfield, np.zeros_like(brightfield[:1])], axis=0)
        brightfield = brightfield.transpose(1, 2, 0)  # Convert to H x W x C format

        fluorescent_path = os.path.join(self.images_root, f"fluorescent_{image_name}")
        fluorescent = np.load(fluorescent_path)
        fluorescent = np.array(fluorescent, dtype=np.float32)
        # fluorescent = np.concatenate([fluorescent, np.zeros_like(fluorescent[:1])], axis=0)
        fluorescent = fluorescent.transpose(1, 2, 0)  # Convert to H x W x C format

        # concat both images to apply transformations
        img = np.concatenate([brightfield, fluorescent], axis=2)

        # Apply Albumentations (expects H x W x C)
        transformed = self.transforms(image=img)
        img = transformed["image"]

        # split transformed in two on the channel
        brightfield = img[:3]
        fluorescent = img[3:]

        # Normalize if required
        if self.to_normal:
            brightfield = self._normalize_image(brightfield)
            fluorescent = self._normalize_image(fluorescent)

        return {"brightfield": brightfield,
                "fluorescent": fluorescent}


def get_dataloader(names_file, images_root, split='train', to_normal=True, batch_size=32, num_workers=0, shuffle=True):
    """
    Creates a DataLoader for 2-channel image reconstruction.

    Args:
        names_file (str): Path to the file containing image names.
        images_root (str): Root directory containing the image files.
        split (str, optional): "train" or "test" mode. Defaults to 'train'.
        to_normal (bool, optional): Whether to normalize images. Defaults to True.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader for the reconstruction dataset.
    """

    dataset = ReconstructionDataset(names_file=names_file,
                                    images_root=images_root,
                                    split=split,
                                    to_normal=to_normal)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
    return dataloader