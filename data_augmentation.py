import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms

class UpdateLabelsMNSIT(Dataset):
  def __init__(self, dataset, labels_shift):
    self.dataset = dataset
    self.labels_shift = labels_shift

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    image, label = self.dataset[idx]
    label = label + self.labels_shift
    return image, label

class AugmentedMNIST(Dataset):
  def __init__(self, dataset, augmented_retina_list):
    self.augmented_retina_list = augmented_retina_list
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset) + len(self.augmented_retina_list)

  def __getitem__(self, idx):
    
    to_tensor = transforms.ToTensor()
    if idx < len(self.dataset):
       image, label = self.dataset[idx]
    else:
       image, label = self.augmented_retina_list[idx - len(self.dataset)]
    if not isinstance(image, torch.Tensor):
      image = to_tensor(image)
    return image, label


def ApplyTransformation(dataset, transformation):
  augmented_data_list = []
  to_tensor = transforms.ToTensor()
  for i in range(len(dataset)):
    img = dataset[i][0]
    label = dataset[i][1]
    if not isinstance(img, torch.Tensor):
      img = to_tensor(img)
        

    img = transformation(img)
    augmented_data_list.append([img, label])

  return AugmentedMNIST(dataset, augmented_data_list)


def DatasetAugmentation(dataset):

  first_transformation = v2.Compose([
      v2.Lambda(lambda img: v2.functional.rotate(img, angle=45.0)),
      v2.ToDtype(torch.float32, scale=True),
  ])

  second_transformation = v2.Compose([
      v2.ColorJitter(brightness=0.4, contrast=0.5,saturation=0.4,hue=0.3),
      v2.ToDtype(torch.float32, scale=True),
  ])

  third_transformation = v2.Compose([
      v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
      v2.ToDtype(torch.float32, scale=True),
  ])

  augmented_dataset_1 = ApplyTransformation(dataset, first_transformation)
  augmented_dataset_2 = ApplyTransformation(augmented_dataset_1, second_transformation)
  augmented_dataset_3 = ApplyTransformation(augmented_dataset_2, third_transformation)


  return augmented_dataset_3


def ConcatDataset(path_dataset, derma_dataset, blood_dataset, retina_dataset):
    """
    e.g. path_dataset = PathMNIST(split="train", download=True)
    """
    DERMA_LABELS_SHIFT = 9
    BLOOD_LABELS_SHIFT = 16
    RETINA_LABELS_SHIFT = 24
    
    subset_count = torch.arange(1080)

    path_subset_dataset = Subset(path_dataset, subset_count)
    blood_subset_dataset = Subset(blood_dataset, subset_count)
    derma_subset_dataset = Subset(derma_dataset, subset_count)

    updated_derma_dataset = UpdateLabelsMNSIT(derma_subset_dataset, DERMA_LABELS_SHIFT)
    updated_blood_dataset = UpdateLabelsMNSIT(blood_subset_dataset, BLOOD_LABELS_SHIFT)
    updated_retina_dataset = UpdateLabelsMNSIT(retina_dataset, RETINA_LABELS_SHIFT)

    concat_datasets = torch.utils.data.ConcatDataset([path_subset_dataset, updated_derma_dataset, updated_blood_dataset, updated_retina_dataset])

    return DatasetAugmentation(concat_datasets)





  
