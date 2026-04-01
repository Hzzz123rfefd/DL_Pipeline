import argparse
from torch.utils.data import DataLoader
from src import models, datasets
from third_party.DL_Pipeline.src.utils import load_config

def main(args):
    config = load_config(args.model_config_path)

    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])
    net.load_pretrained(
        save_model_dir = config["logging"]["save_dir"],
    )
    net.eval()
    
    """get data loader"""
    valid_datasets = datasets[config["dataset_type"]](**config["dataset"], data_type = "valid")

    valid_dataloader = DataLoader(
        valid_datasets, 
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = valid_datasets.collate_fn
    )
    net.eval_model(val_dataloader = valid_dataloader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/template.yml")
    args = parser.parse_args()
    main(args)
