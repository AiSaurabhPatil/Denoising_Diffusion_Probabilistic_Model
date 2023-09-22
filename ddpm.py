import os 
import torch 
import torch.nn as nn
from torch import optim 
import matplotlib.pyplot as plt 
import tqdm as tqdm
from utils import get_data ,save_images
from modules import Unet
import logging
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps = 1000 , beta_start = 1e-4 , beta_end = 0.02 , img_size = 256):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end 
        self.img_size = img_size 
        
        # Initializting beta hyperparameter
        self.beta = self.prepare_noise_schedule().to(device="cuda")
        self.alpha = 1 - self.beta
        self.alpha_hat  = torch.cumprod(self.alpha , dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start , self.beta_end , self.noise_steps)


    def noise_images(self, x ,t): 
        """
        function to add noise into input image at different timesteps

        Args: 
            x : input image 
            t : timestep
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        epsilon = torch.rand_like(x)
        noisy_img = (sqrt_alpha_hat * x) + (sqrt_one_minus_alpha_hat * epsilon) 

        return noisy_img ,epsilon 
    
    def sample_timestep(self , n):
        """
            function to generate a random timesteps of size n 
            example : for n = 10 
                    output: tensor([34, 85, 7, 93, 17, 54, 26, 6, 98, 79])
        """
        return torch.randint(low = 1 , high = self.noise_steps , size = (n,))
    


    def sampling_new_image(self , model , n):
        model.eval()
        with torch.no_grad():
            x = torch.randn( n ,3 , self.img_size , self.img_size).to(device='cuda') 
            """
            n specifies the number of samples 
            3 specifies of no of channels in sample image 
            other 2 arguments (self.img_size) defines the height and width of a sample image 
            """
            for i in tqdm(reversed(range(1,self.noise_steps)),position=0):
                t = (torch.ones(n) * i).long().to(device='cuda')
                predicted_noise = model(x , t)
                alpha = self.alpha[t][:,None,None,None] # (1,1,1,1)
                alpha_hat = self.alpha_hat[t][:,None,None,None] # (1,1,1,1)
                beta = self.beta[t][:,None,None,None]

                if i > 1 : 
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1/torch.sqrt(alpha)) * ( x - ((1- alpha)/(torch.sqrt(1-alpha))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1)/2
        x = (x * 255).type(torch.unint8)

        return x  

def train(args):
    dataloader = get_data(args)
    model = Unet().to(device='cuda')
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size)
    l = len(dataloader)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device="cuda")
            t = diffusion.sample_timesteps(images.shape[0]).to(device='cuda')
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            
    
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"./cifar_train"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()