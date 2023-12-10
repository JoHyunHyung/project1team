import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class ButterfliesDataset(Dataset) :

    def __init__(self, path , transform=None):
        self.transform = transform
        self.path_list = glob.glob(os.path.join(path, "*.png"))

    def __getitem__(self, item):
        image = Image.open(self.path_list[item])
        image = image.convert('RGB')

        if self.transform is not None :
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.path_list)