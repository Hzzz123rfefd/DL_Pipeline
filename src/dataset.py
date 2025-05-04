from torch.utils.data import Dataset
import datasets
from abc import abstractmethod
from src.utils import recursive_collate_fn

class DatasetBase(Dataset):
    def __init__(
        self, 
        train_data_path:str = None,
        test_data_path:str = None,
        valid_data_path:str = None,
        data_type:str = "train"
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.valid_data_path = valid_data_path
        self.data_type = data_type
        
        if self.data_type == "train":
            self.data_path = train_data_path
        elif self.data_type == "test":
            self.data_path = test_data_path
        elif self.data_type == "valid":
            self.data_path = valid_data_path
            
        self.dataset = datasets.load_dataset('json', data_files = self.data_path, split = "train")
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len

    @abstractmethod    
    def __getitem__(self, idx):
        pass
        
    def collate_fn(self, batch):
        return recursive_collate_fn(batch)
