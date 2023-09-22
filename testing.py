import torch
from tqdm import tqdm

# Set the number of noise steps
noise_steps = 100

# Create a tensor of shape (10,) containing the values of i from 99 to 1, converted to longs and moved to the GPU
for i in tqdm(reversed(range(1,noise_steps)),position=0):
    t = (torch.ones(10) * i)

# Print the tensor
print(t)