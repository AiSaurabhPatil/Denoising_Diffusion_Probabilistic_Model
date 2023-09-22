import torch 
import torch.nn as nn 
import torch.nn.functional as F 


# Self attention implementation 

class Self_attention(nn.Module):
    def __init__(self ,channels , size): 
        super(Self_attention , self).__init__()
        self.channels = channels
        self.size = size 
        self.multi_head_attention = nn.MultiheadAttention(channels , 4 , batch_first=True) 
        self.layer_norm = nn.LayerNorm([channels])
        self.network = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels,channels),
            nn.GELU(),
            nn.Linear(channels,channels)
        )
    
    def forward(self , x ): 
        x = x.view(-1 , self.channels ,self.size* self.size).swapaxes(1,2)
        norm_x = self.layer_norm(x)
        attention_value , _ = self.multi_head_attention(norm_x ,norm_x,norm_x)
        attention_value = attention_value + x 
        attention_value = self.network(attention_value) + attention_value
        return attention_value.swapaxes(2,1).view(-1 , self.channels ,self.size , self.size)
    

class DoubleConv(nn.Module):
    def __init__(self , in_channels , out_channels , mid_channels=None , residual = False):
        super.__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels , mid_channels , kernel_size=3 ,padding= 1 , bias=False),
            nn.GroupNorm(1 , mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels , kernel_size=3 , padding=1 , bias=False),
            nn.GroupNorm(1, out_channels)
        )
    
    def forward(self,x):
        if self.residual:
            return F.gelu(x + self.double(x))
        
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self , in_channels , out_channels ,emb_dim = 256):
        super.__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels , in_channels , residual=True),
            DoubleConv(in_channels , out_channels)
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim , out_channels)
        )

    def forward(self,x ,t):
        x = self.maxpool_conv(x)
        timestep_emb = self.embedding_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + timestep_emb
    

class Up(nn.Module):
    def __init__(self ,in_channels, out_channels , emb_dim = 256):
        super.__init__()

        self.up = nn.Upsample(scale_factor= 2 , mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels , in_channels , residual=True),
            DoubleConv(in_channels , out_channels , in_channels//2)
        )
    
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim , out_channels)
        )

    def forward(self ,x,skip_x ,t):
        x = self.up(x)
        x = torch.cat([skip_x,x],dim=1)
        x = self.conv(x)
        timestep_emb = self.embedding_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        return x + timestep_emb
    

class Unet(nn.Module):
    def __init__(self ,in_channels=3 , out_channels = 3 , time_dim=256 ): 
        super.__init__()
        self.time_dim = time_dim

        self.input_conv = DoubleConv(in_channels , 64)
        self.down1 = Down(64,128)
        self.sa1 = Self_attention(128 , 32)
        self.down2 = Down(128 , 256)
        self.sa2 = Self_attention(256,16)
        self.down3 = Down(256 , 256)
        self.sa3 = Self_attention(256,8)

        self.bottom1 = DoubleConv(256 , 512)
        self.bottom2 = DoubleConv(512 , 512)
        self.bottom3 = DoubleConv(512 , 256)

        self.up1 = Up(512 ,128)
        self.sa4 = Self_attention(128,16)
        self.up2 = Up(256 ,16)
        self.sa5 = Self_attention(64,32)
        self.up3 = Up(128 , 64)
        self.sa6 = Self_attention(64,64)
        self.output_conv = nn.Conv2d(64 , out_channels , kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output





    


