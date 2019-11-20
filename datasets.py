import glob
import random
import os
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
from main import*

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image
    # Image transformations

def transforms_for_dataset(height, width):

    transforms_ = [
        transforms.Resize(int(height * 1.12), Image.BICUBIC),
        transforms.RandomCrop(height, width),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms_


def get_train_data(name_of_dataset, bs, cpu_num, height, width):

    transforms__=transforms_for_dataset(height, width)

    train_set = DataLoader(
        ImageDataset("data/%s" % name_of_dataset, transforms_=transforms__, unaligned=True),
        batch_size=bs,
        shuffle=True,
        num_workers=cpu_num,
    )

    return train_set

def get_test_data(name_of_dataset, height, width):

    transforms__ = transforms_for_dataset(height, width)

    val_set = DataLoader(
        ImageDataset("data/%s" % name_of_dataset, transforms_=transforms__, unaligned=True, mode="test"),
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    return val_set


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A_64" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B_64" % mode) + "/*.*"))


    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))