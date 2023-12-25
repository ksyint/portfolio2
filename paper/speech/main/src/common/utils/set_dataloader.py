from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

# def set_dataloader(dataset_train: Dataset, dataset_val: Dataset, dataset_test: Dataset, batch_size: int, num_workers: int) -> dict:
#     data_loader = {}
#     data_loader['train'] = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=num_workers,
#                                         drop_last=False, pin_memory=True)
#     data_loader['validation'] = DataLoader(dataset_val, batch_size, shuffle=False, num_workers=num_workers,
#                                             drop_last=False, pin_memory=True)
#     data_loader['test'] = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=num_workers,
#                                             drop_last=False, pin_memory=True)
#     return data_loader

def set_dataloader(datasets: dict, batch_size: int, num_workers: int) -> dict:
    data_loader = {}
    for k, v in datasets.items():
        data_loader[k] = DataLoader(v, batch_size, shuffle=True, num_workers=num_workers,
                                        drop_last=False, pin_memory=True)

    return data_loader