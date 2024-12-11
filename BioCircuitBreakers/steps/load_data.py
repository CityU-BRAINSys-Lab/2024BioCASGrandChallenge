from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from neurobench.datasets import PrimateReaching

def LoadData(proj):
    ## Load the data
    # Define an Empty Dataset for concatenation
    class EmptyDataset(Dataset):
        def __init__(self):
            super(EmptyDataset, self).__init__()

        def __len__(self):
            return 0

        def __getitem__(self, index):
            return None
    
    train_dataset = EmptyDataset()
    valid_dataset = EmptyDataset()

    for filename in proj.all_files:
        # The dataset was split by the approach that the first train ratio of the dataset is used for training, 
        # the first half of the rest for validation, and the second half for testing. 
        # So, the train_ratio is 0.5 to realize 75% training + validation and 25% testing as demanded.
        dataset = PrimateReaching(file_path=proj.data_dir, filename=filename,
                                num_steps=proj.dataset_num_steps, train_ratio=0.5, bin_width=proj.dataset_bin_width,
                                biological_delay=0, remove_segments_inactive=False, label_series=proj.dataset_label_series)

        train_dataset = ConcatDataset([train_dataset, Subset(dataset, dataset.ind_train)])
        valid_dataset = ConcatDataset([valid_dataset, Subset(dataset, dataset.ind_val)])
        
    train_set_loader = DataLoader(train_dataset, batch_size=proj.dataloader_batch_size[0], 
                                  shuffle=proj.dataloader_shuffle[0], drop_last=True)
    valid_set_loader = DataLoader(valid_dataset, batch_size=proj.dataloader_batch_size[1], 
                                  shuffle=proj.dataloader_shuffle[1])

    ## Check the process of data loading
    print(
        "check num of data train/valid %d/%d"
        % (len(train_dataset), len(valid_dataset))
    )

    return dataset.input_feature_size, (train_set_loader, valid_set_loader)