import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

ALPHA_W =  [[0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.6123, 0.3877, 0.0, 0.0],
            [0.4557, 0.3436, 0.2007, 0.0],
            [0.3317, 0.2858, 0.2346, 0.1479]]

CLIP_W = [0.5, 0.98, 2.6, 4.16, 5.66]

def build_quant_base(alpha):
    alpha = np.array(alpha)
    bit_range,bit_num = alpha.shape
    assert bit_range == bit_num + 1, f'bit_range must be bit_num + 1, while bit_range is {bit_range} and bit_num is {bit_num}.'
    bases = [-1, 1]
    quant_bases = np.zeros((bit_range,2**bit_num))
    recur_index = 0
    def recur(index,previous):
        nonlocal recur_index
        if index >= bit_num:
            # print(previous)
            quant_bases[:,recur_index] = previous
            recur_index += 1
            return
        for base in bases:
            curr = previous + base * alpha[:,index]
            recur(index+1,curr)
    
    recur(0,np.zeros(bit_range))
    return quant_bases

QUANT_BASES = build_quant_base(ALPHA_W)
EPS = 1e-6

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def quantize_wgt(data,clip_scale,proj_set,layerwise):
    x = data / (clip_scale + EPS)
    x = torch.clamp(x,-1,1)
    xshape = x.shape
    pshape = proj_set.shape
    if not layerwise:
        assert xshape[0] == pshape[0], f'xshape[0]:{xshape[0]} must be equal to pshape[0]:{pshape[0]} with channelwise mode.'
        xhard = x.view(xshape[0],1,-1)
        phard = proj_set.view(pshape[0],-1,1)
        idxs = (xhard - phard).abs().min(dim=1)[1]
        xhard = torch.gather(input=proj_set,dim=1,index=idxs).view(xshape)
    else:
        assert pshape[0] == 1, f'pshape[0]:{pshape[0]} must be 1 with layerwise mode.'
        xhard = x.view(1,-1)
        phard = proj_set.view(-1,1)
        idxs = (xhard - phard).abs().min(dim=0)[1]
        xhard = torch.gather(input=proj_set,dim=0,index=idxs).view(xshape)
    y = (xhard - x).detach() + x
    y = y * clip_scale
    return y

def quantize_act(data,clip_scale,bitwidth):
    quant_scale = 2**bitwidth - 1
    x = data / (clip_scale + EPS)
    x = torch.clamp(x,0,1) * quant_scale
    y = round_pass(x)
    y = y * clip_scale / (quant_scale + EPS)
    return y


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,bit_wgt=4,bit_act=4,layerwise=False,**kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        
        self.k_size = kernel_size[0] if isinstance(kernel_size,(tuple,list)) else kernel_size
        if layerwise:
            self.groups_wgt = 1
            self.groups_act = 1
        else:
            self.groups_wgt = out_channels
            self.groups_act = in_channels
        self.layerwise = layerwise
        self.ema_decay = 0.99
        
        self.register_buffer('alpha_wgt',torch.full((self.groups_wgt,),1.0))
        self.register_parameter('alpha_act',nn.Parameter(torch.tensor([3.0])))        
        self.register_buffer('clip_wgt',torch.tensor(CLIP_W))
        self.register_buffer('bit_wgt',torch.full((self.groups_wgt,),bit_wgt,dtype=torch.long))
        self.register_buffer('bit_act',torch.full((self.groups_act,),bit_act,dtype=torch.long))
        points_wgt = torch.tensor(QUANT_BASES,dtype=torch.float32).index_select(0,self.bit_wgt)
        self.register_buffer('points_wgt',points_wgt)
        self.init = False
    
    def forward(self,x):
        if self.training:
                with torch.no_grad():
                    if self.layerwise:
                        scale_wgt = self.weight.abs().mean()
                    else:
                        scale_wgt = self.weight.abs().mean(dim=(1,2,3))
                    alpha_wgt_cur = self.clip_wgt[self.bit_wgt] * scale_wgt
                    if not self.init:
                        self.alpha_wgt.data = alpha_wgt_cur
                        self.init = True
                    else:
                        self.alpha_wgt.data = alpha_wgt_cur * (1 - self.ema_decay) + self.ema_decay * self.alpha_wgt.data
        
        self.x_q = quantize_act(x,self.alpha_act.view(1,-1,1,1),self.bit_act.view(1,-1,1,1))
        self.w_q = quantize_wgt(self.weight,self.alpha_wgt.view(-1,1,1,1),self.points_wgt,self.layerwise)
        
        if self.training:
            self.x_q.retain_grad()
            self.w_q.retain_grad()
            self.x_res = x - self.x_q
            self.w_res = self.weight - self.w_q
        
        return F.conv2d(self.x_q, self.w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def compute_residual(self):

        w_num = float(self.w_res.nelement()/self.groups_wgt)
        x_num = float(self.x_res.nelement()/self.groups_act)
        with torch.no_grad():
            if self.layerwise:
                self.w_err = self.w_res.mul(self.w_q.grad).sum().abs().div(w_num).view(1).detach()
                self.x_err = self.x_res.mul(self.x_q.grad).sum().abs().div(x_num).view(1).detach()
            
            else:
                self.w_err = self.w_res.mul(self.w_q.grad).sum(dim=(1,2,3),keepdim=False).abs().div(w_num).detach()
                self.x_err = self.x_res.mul(self.x_q.grad).sum(dim=(0,2,3),keepdim=False).abs().div(x_num).detach()

        return self.w_err,self.x_err
    
    def update_quant_points(self,device):
        self.points_wgt.data = torch.tensor(QUANT_BASES,dtype=torch.float32).to(device).index_select(0,self.bit_wgt)         
        

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, groups_wgt=None,groups_act=None,bit_wgt=4,bit_act=4,layerwise=False,**kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)

        if layerwise:
            groups_wgt = 1
            groups_act = 1
        self.layerwise = layerwise
        self.groups_wgt = groups_wgt if groups_wgt else out_features // 16
        self.groups_act = groups_act if groups_act else in_features // 4
        self.ema_decay = 0.99

        self.register_buffer('alpha_wgt',torch.full((self.groups_wgt,),1.0))
        self.register_parameter('alpha_act',nn.Parameter(torch.tensor([3.0])))
        self.register_buffer('clip_wgt',torch.tensor(CLIP_W))
        self.register_buffer('bit_wgt',torch.full((self.groups_wgt,),bit_wgt,dtype=torch.long))
        self.register_buffer('bit_act',torch.full((self.groups_act,),bit_act,dtype=torch.long))
        points_wgt = torch.tensor(QUANT_BASES,dtype=torch.float32).index_select(0,self.bit_wgt)
        self.register_buffer('points_wgt',points_wgt)
        self.init = False
    
    def forward(self,x):
        w_r = self.weight.view(self.groups_wgt,-1,self.in_features)
        x_r = x.view(x.shape[0],self.groups_act,-1)
        if self.training:
            with torch.no_grad():
                if self.layerwise:
                    scale_wgt = w_r.abs().mean()
                else:
                    scale_wgt = w_r.abs().mean(dim=(1,2))
                alpha_wgt_cur = self.clip_wgt[self.bit_wgt] * scale_wgt
                if not self.init:
                    self.alpha_wgt.data = alpha_wgt_cur
                    self.init = True
                else:
                    self.alpha_wgt.data = alpha_wgt_cur * (1 - self.ema_decay) + self.ema_decay * self.alpha_wgt.data
        
        self.x_q = quantize_act(x_r,self.alpha_act,self.bit_act.view(1,self.in_groups,1)).view(x.shape[0],self.in_features) 
        self.w_q = quantize_wgt(w_r,self.alpha_wgt.view(self.groups_wgt,1,1),self.points_wgt,self.layerwise).view(self.out_features,self.in_features)
        if self.training:
            self.x_q.retain_grad()
            self.w_q.retain_grad()
            self.x_res = x - self.x_q
            self.w_res = self.weight - self.w_q
        
        return F.linear(self.x_q, self.w_q, self.bias)    
    
    def compute_residual(self):

        w_num = float(self.w_res.nelement()/self.groups_wgt)
        x_num = float(self.x_res.nelement()/self.groups_act)
        
        if self.layerwise:
            self.w_err = self.w_res.mul(self.w_q.grad).sum().abs().div(w_num).view(1).detach()
            self.x_err = self.x_res.mul(self.x_q.grad).sum().abs().div(x_num).view(1).detach()
        else:
            self.w_err = self.w_res.mul(self.w_q.grad).view(self.groups_wgt,-1,self.in_features).sum(dim=(1,2),keepdim=False).abs().div(w_num).detach()
            self.x_err = self.x_res.mul(self.x_q.grad).view(self.x_q.shape[0],self.groups_act,-1).sum(dim=(0,2),keepdim=False).abs().div(x_num).detach()

        return self.w_err,self.x_err
    
    def update_quant_points(self,device):
        self.points_wgt.data = torch.tensor(QUANT_BASES,dtype=torch.float32).index_select(0,self.bit_wgt).to(device)
        
class FirstConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,bit_wgt:int=8,bit_act:int=8,**kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.Qn_W = - 2**(bit_wgt-1) + 0.5
        self.Qp_W = 2**(bit_wgt-1) - 0.5
    def forward(self, x):
        max_W = self.weight.data.abs().max()
        w_q = ((self.weight/max_W*self.Qp_W - self.Qn_W).round() + self.Qn_W) / self.Qp_W * max_W
        w_q = (w_q - self.weight).detach() + self.weight
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LastLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bit_wgt:int=8,bit_act:int=8,**kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.Qn_W = - 2**(bit_wgt-1) + 0.5
        self.Qp_W = 2**(bit_wgt-1) - 0.5
        self.Qn_A = 0
        self.Qp_A = 2**bit_act - 1
    def forward(self, x):
        max_W = self.weight.data.abs().max()
        w_q = ((self.weight/max_W*self.Qp_W - self.Qn_W).round() + self.Qn_W) / self.Qp_W * max_W
        w_q = (w_q - self.weight).detach() + self.weight
        max_A = x.abs().max()
        x_q = ((x/max_A*self.Qp_A - self.Qn_A).round() + self.Qn_A) / self.Qp_A * max_A
        x_q = (x_q - x).detach() + x
        return F.linear(x_q, w_q, self.bias)
            

if __name__ == '__main__':
    
    weight = torch.randn(16,16,3,3)
    alpha = torch.full((16,),1.0)
    bit = torch.full((16,),4,dtype=torch.long)
    points = torch.tensor(QUANT_BASES,dtype=torch.float32).index_select(0,bit)
    quantized = quantize_wgt(weight,alpha.view(-1,1,1,1),points,False)
    print('ok')
    
    