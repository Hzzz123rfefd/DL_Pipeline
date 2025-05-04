import os
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import abstractmethod
from tqdm import tqdm
import torch.nn.functional as F
from src.utils import AverageMeter
import logging

class ModelBase(nn.Module):
    def __init__(
        self,
        device = "cuda"
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
    
    def trainning(
            self,
            train_dataloader: DataLoader = None,
            test_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 0,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            eval_interval:int = 10,
            save_model_dir:str = None
        ):
            self.total_epoch = total_epoch
            self.eval_interval = eval_interval
            self.clip_max_norm = clip_max_norm
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader
            self.val_dataloader = val_dataloader
            self.to(self.device)

            ## get optimizer
            self.configure_optimizers(optimizer_name, lr, weight_decay)
            
            ## get lr_scheduler
            self.configure_lr_scheduler(factor, patience)

            ## some trainning  setting 
            self.configure_train_set(save_model_dir)
            
            ## get net pretrain parameters if need 
            self.init_model()
            
            self.configure_train_logging()

            ## get train log
            self.configure_train_log()
            
            self.begin_trainning()
            
    def configure_optimizers(self, optimizer_name, lr, weight_decay):
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr, weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(), lr, weight_decay = weight_decay)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr, weight_decay = weight_decay)
    
    def configure_lr_scheduler(self, factor, patience, mode = "min"):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = self.optimizer, 
                mode = mode, 
                factor = factor, 
                patience = patience
            )
    
    def configure_train_set(self, save_model_dir):
        self.first_trainning = True
        self.save_model_dir = save_model_dir
        self.check_point_path =  self.save_model_dir  + "/checkpoint.pth"
        self.log_path =  self.save_model_dir + "/train.log"
        os.makedirs(self.save_model_dir,  exist_ok=True)
        with open(self.log_path, "w") as f:
            pass  
    
    def configure_train_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 文件 handler
        file_handler = logging.FileHandler(self.log_path, mode="a")
        file_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加 handler
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def init_model(self):
        if  os.path.isdir(self.save_model_dir) and os.path.exists(self.check_point_path) and os.path.exists(self.log_path):
            self.load_pretrained(self.save_model_dir)  
            self.first_trainning = False
        else:
            with open(self.log_path, "w") as f:
                pass  
            os.makedirs(self.save_model_dir, exist_ok = True)
            self.first_trainning = True
    
    def configure_train_log(self):
        if self.first_trainning:
            self.best_loss = float("inf")
            self.last_epoch = 0
        else:
            checkpoint = torch.load(self.check_point_path, map_location = self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.best_loss = checkpoint["loss"]
            self.last_epoch = checkpoint["epoch"] + 1
        
    def begin_trainning(self):
            try:
                for each in range(self.last_epoch, self.total_epoch):
                    self.epoch = each
                    self.logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
                    self.train_one_epoch()
                    test_loss = self.test_one_epoch()
                    self.lr_scheduler.step(test_loss)
                    is_best = test_loss < self.best_loss
                    self.best_loss = min(test_loss, self.best_loss)
                    if self.epoch != 0 and self.epoch % self.eval_interval == 0:
                        self.eval_model(self.val_dataloader)
                    if is_best:
                        self.save_pretrained(self.save_model_dir)
                        
            # interrupt trianning
            except KeyboardInterrupt:
                self.save_train_log()
            
            self.save_train_log()
        
    def save_train_log(self):
        torch.save({
                "epoch": self.epoch,
                "loss": self.best_loss,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict()
            }, 
            self.check_point_path
        )
        print("model saved !")
    
    def train_one_epoch(self):
        self.train().to(self.device)
        pbar = tqdm(self.train_dataloader,desc="Processing epoch "+str(self.epoch), unit="batch")
        total_loss = AverageMeter()
        
        for _, inputs in enumerate(self.train_dataloader):
            """ grad zeroing """
            self.optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_max_norm)

            """ modify parameters """
            self.optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "Train Epoch: {:d}, total_loss: {:.4f}, use_memory: {:.1f}G".format(
                self.epoch,
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
            
        self.logger.info(postfix_str)

    def test_one_epoch(self):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(self.test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())

        self.logger.info("Test Epoch: {:d}, total_loss: {:.4f}".format(self.epoch,total_loss.avg))
        return total_loss.avg
    
    @abstractmethod
    def eval_model(self,val_dataloader):
        pass
        
    @abstractmethod    
    def compute_loss(self, input):
        pass
    
    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self, save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
        
class ModelRegression(ModelBase):
    def compute_loss(self, input):
        output = {
            "total_loss": F.mse_loss(input["predict"], input["label"])
        }
        return output

class ModelBinaryClassification(ModelBase):
    def compute_loss(self, input):
        output = {
            "total_loss": F.binary_cross_entropy_with_logits(input["predict"],input["label"])
        }
        return output
    
class ModelMultiClassification(ModelBase):
    def compute_loss(self, input):
        output = {
            "total_loss": F.cross_entropy(input["predict"],input["label"])
        }
        return output
    
class ModelDiffusionBase(ModelRegression):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 256, 
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(device)
        self.width = width
        self.height = height
        self.channel = channel
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.time_dim = time_dim
        
        for k, v in self.ddpm_schedules(self.beta_start, self.beta_end, self.noise_steps).items():
            self.register_buffer(k, v)
    
    def ddpm_schedules(self, beta1, beta2, T):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")    