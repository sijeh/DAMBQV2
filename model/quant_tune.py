from numpy import mod
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from quant_layers import QuantConv2d,QuantLinear,FirstConv2d,LastLinear

def _convert2quant_module(m,bit_wgt,bit_act,fc_groups_wgt,fc_groups_act,names_8bit,prefix=None,layerwise=False):
    for name,child in m.named_children():
        module_name = '.'.join((prefix,name)) if prefix else name
            
        if type(child) == nn.Conv2d:
            kwargs = {}
            kwargs['in_channels'] = child.in_channels
            kwargs['out_channels'] = child.out_channels
            kwargs['kernel_size'] = child.kernel_size
            kwargs['stride'] = child.stride
            kwargs['padding'] = child.padding
            kwargs['dilation'] = child.dilation
            kwargs['groups'] = child.groups
            kwargs['bias'] = False
            kwargs['padding_mode'] = child.padding_mode
            
            if module_name in names_8bit:
                new_conv = FirstConv2d(**kwargs)
            else:
                kwargs['layerwise'] = layerwise
                kwargs['bit_wgt'] = bit_wgt
                kwargs['bit_act'] = bit_act
                new_conv = QuantConv2d(**kwargs)
            new_conv.weight = child.weight
            setattr(m,name,new_conv)
        elif type(child) == nn.Linear:
            kwargs = {}
            kwargs['in_features'] = child.in_features
            kwargs['out_features'] = child.out_features
            kwargs['bias'] = False
            
            if module_name in names_8bit:
                new_linear = LastLinear(**kwargs)
            else:
                kwargs['layerwise'] = layerwise
                kwargs['bit_wgt'] = bit_wgt
                kwargs['bit_act'] = bit_act
                kwargs['groups_wgt'] = fc_groups_wgt
                kwargs['groups_act'] = fc_groups_act
                new_linear = QuantLinear(**kwargs)
            new_linear.weight = child.weight
            setattr(m,name,new_linear)
        else:
            _convert2quant_module(child,bit_wgt,bit_act,fc_groups_wgt,fc_groups_act,names_8bit,module_name,layerwise)
class QuantTune():
    def __init__(self,model:nn.Module,wgt_target:float,act_target:float,tune_ratio_range:float = 0.3,duration=(0,10)) -> None:
        super().__init__()
        self.model = model
        self.wgt_target = wgt_target
        self.act_target = act_target
        self.tune_ratio_range = tune_ratio_range
        self.duration = duration
        
        self.init_err_buffer()
        
    def convert2quant_module(self,bit_wgt_init,bit_act_init,fc_groups_wgt,fc_groups_act,names_8bit,layerwise):
        _convert2quant_module(self.model,bit_wgt_init,bit_act_init,fc_groups_wgt,fc_groups_act,names_8bit,layerwise)
        
    def init_err_buffer(self):
        groups_wgt = 0
        groups_act = 0
        device = None
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                groups_wgt += module.groups_wgt
                groups_act += module.groups_act
                device = module.weight.device
        self.groups_wgt = groups_wgt
        self.groups_act = groups_act
        self.w_err_total = torch.zeros(self.groups_wgt,dtype=torch.float,device=device)
        self.x_err_total = torch.zeros(self.groups_act,dtype=torch.float,device=device)
        
    def compute_avg_bit(self):
        x_num_total = 0
        w_num_total = 0
        x_bit_total = 0
        w_bit_total = 0
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d)):
                w_num = module.k_size * module.k_size * module.out_channels * module.in_channels
                x_num = module.x_q.shape[1] * module.x_q.shape[2] * module.x_q.shape[3]
                w_bit = w_num * module.bit_wgt.mean().item()
                x_bit = x_num * module.bit_act.mean().item()
                
            elif isinstance(module,(QuantLinear)):
                w_num = module.out_features * module.in_features
                x_num = module.in_features
                w_bit = w_num * module.bit_wgt.mean().item()
                x_bit = x_num * module.bit_act.mean().item()
            else:
                continue
            w_num_total += w_num
            x_num_total += x_num
            w_bit_total += w_bit
            x_bit_total += x_bit
        
        self.x_bit_avg = x_bit_total / x_num_total
        self.w_bit_avg = w_bit_total / w_num_total
            
    
    def compute_quant_residual(self):
        w_err_list = []
        x_err_list = []
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                w_err,x_err = module.compute_residual()
                w_err_list.append(w_err)
                x_err_list.append(x_err)
        
        self.w_err_total += torch.cat(w_err_list,dim=0)
        self.x_err_total += torch.cat(x_err_list,dim=0)

    def load_bit_config(self):
        bit_wgt_list = []
        bit_act_list =[]
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                bit_wgt = module.bit_wgt
                bit_act = module.bit_act
                
                bit_wgt_list.append(bit_wgt)
                bit_act_list.append(bit_act)
        
        self.bit_wgt= torch.cat(bit_wgt_list,dim=0)
        self.bit_act = torch.cat(bit_act_list,dim=0)
    
    def save_bit_config(self):
        bit_wgt_aux = self.bit_wgt.clone()
        bit_act_aux = self.bit_act.clone()
        
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                groups_wgt = module.groups_wgt
                groups_act = module.groups_act
                module.bit_wgt.data = bit_wgt_aux[0:groups_wgt]
                module.bit_act.data = bit_act_aux[0:groups_act]
                bit_wgt_aux = bit_wgt_aux[groups_wgt:]
                bit_act_aux = bit_act_aux[groups_act:]

    def sort_and_tune(self,iter):
        
        tune_ratio = (self.tune_ratio_range - 0.01) * iter / (self.duration[1]-self.duration[0]) + 0.01
        self.load_bit_config()
        self.compute_avg_bit()
        if self.x_bit_avg > self.act_target:
            x_k = int(len(self.x_err_total) * tune_ratio)
            assert len(self.bit_act_total) == len(self.x_err_total)
            _,x_indices = torch.topk(self.x_err_total,x_k,largest=False)
            self.bit_act[x_indices] -= 1
            self.bit_act = torch.clamp(self.bit_act,min=0)
        if self.w_bit_avg > self.wgt_target:
            w_k = int(len(self.w_err_total) * tune_ratio)        
            assert len(self.bit_wgt_total) == len(self.w_err_total)
            _,w_indices = torch.topk(self.w_err_total,w_k,largest=False)
            self.bit_wgt[w_indices] -= 1
            self.bit_wgt = torch.clamp(self.bit_wgt,min=0)
        
        self.save_bit_config()
        self.init_err_buffer()
        
        
if __name__ == '__main__':
    model = resnet18()
    tuner = QuantTune(model,2.0,2.0)
    print('ok')
    names_8bit = ['conv1','fc']
    names = []
    for name,child in model.named_modules():
        names.append(name)
    print(names)

    tuner.convert2quant_module(4,4,1,1,names_8bit,False)
    
    print('ok')
    