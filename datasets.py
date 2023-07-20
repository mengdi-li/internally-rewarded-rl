from torch.utils.data import Dataset
import numpy as np

class CustomedMNIST(Dataset):
    def __init__(self, dataset_file, train=True, transform=None):
        data = np.load(dataset_file)
        if train: 
            self.X_data = data['X_train'] # numpy array
            self.y_data = data['y_train'].squeeze().astype(np.int64)
        else:
            self.X_data = data['X_test'] # numpy array
            self.y_data = data['y_test'].squeeze().astype(np.int64)

        self.image_size = np.sqrt(self.X_data[0].shape[-1]).astype(int)

        self.transform = transform

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, idx):
        image = self.X_data[idx].reshape((self.image_size, self.image_size))
        label = self.y_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
