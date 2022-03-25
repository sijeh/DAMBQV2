
import torch
import torch.nn as nn
from torch.nn.modules import module
from torchvision.models.resnet import resnet18
from .quant_layers import QuantConv2d,QuantLinear,FirstConv2d,LastLinear
import torch.distributed as dist

def _convert2quant_module(m,bit_wgt,bit_act,names_8bit,fc_groups_wgt=None,fc_groups_act=None,prefix=None,layerwise=False,device=torch.device('cpu')):
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
                new_conv = FirstConv2d(**kwargs).to(device)
            else:
                kwargs['layerwise'] = layerwise
                kwargs['bit_wgt'] = bit_wgt
                kwargs['bit_act'] = bit_act
                new_conv = QuantConv2d(**kwargs).to(device)
            new_conv.weight = child.weight
            setattr(m,name,new_conv)
        elif type(child) == nn.Linear:
            kwargs = {}
            kwargs['in_features'] = child.in_features
            kwargs['out_features'] = child.out_features
            kwargs['bias'] = False
            
            if module_name in names_8bit:
                new_linear = LastLinear(**kwargs).to(device)
            else:
                kwargs['layerwise'] = layerwise
                kwargs['bit_wgt'] = bit_wgt
                kwargs['bit_act'] = bit_act
                kwargs['groups_wgt'] = fc_groups_wgt
                kwargs['groups_act'] = fc_groups_act
                new_linear = QuantLinear(**kwargs).to(device)
            new_linear.weight = child.weight
            setattr(m,name,new_linear)
        else:
            _convert2quant_module(child,bit_wgt,bit_act,names_8bit,fc_groups_wgt,fc_groups_act,module_name,layerwise,device)
class QuantTune():
    def __init__(self,model,wgt_target,act_target,tune_ratio_range=0.3,layerwise=False,duration=(0,10),logger=None,device=torch.device('cpu')):
        super().__init__()
        self.model = model
        self.wgt_target = wgt_target + 0.07
        self.act_target = act_target + 0.07
        self.tune_ratio_range = tune_ratio_range
        self.duration = duration
        self.device = device
        self.quantize_cur = False
        self.logger = logger
        self.get_first_last_name()
        self.convert2quant_module(4,4,layerwise=layerwise,)
        self.init_err_buffer()
        
    def convert2quant_module(self,bit_wgt_init,bit_act_init,fc_groups_wgt=None,fc_groups_act=None,layerwise=False):
        _convert2quant_module(self.model,bit_wgt_init,bit_act_init,self.first_last_module_name,fc_groups_wgt,fc_groups_act,None,layerwise,self.device)
        
    def init_err_buffer(self):
        groups_wgt = 0
        groups_act = 0
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                groups_wgt += module.groups_wgt
                groups_act += module.groups_act
        self.groups_wgt = groups_wgt
        self.groups_act = groups_act
        self.w_err_total = torch.zeros(self.groups_wgt,dtype=torch.float,device=self.device)
        self.x_err_total = torch.zeros(self.groups_act,dtype=torch.float,device=self.device)
        
    def compute_avg_bit(self):
        x_num_total = 0
        w_num_total = 0
        x_bit_total = 0
        w_bit_total = 0
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d)):
                w_num = module.k_size * module.k_size * module.out_channels * module.in_channels
                x_num = module.x_q.shape[1] * module.x_q.shape[2] * module.x_q.shape[3]
                w_bit = w_num * module.bit_wgt.float().mean().item()
                x_bit = x_num * module.bit_act.float().mean().item()
                
            elif isinstance(module,(QuantLinear)):
                w_num = module.out_features * module.in_features
                x_num = module.in_features
                w_bit = w_num * module.bit_wgt.float().mean().item()
                x_bit = x_num * module.bit_act.float().mean().item()
            else:
                continue
            w_num_total += w_num
            x_num_total += x_num
            w_bit_total += w_bit
            x_bit_total += x_bit
        
        self.x_bit_avg = x_bit_total / x_num_total
        self.w_bit_avg = w_bit_total / w_num_total
        
        return self.w_bit_avg,self.x_bit_avg
            
    
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
                module.update_quant_points(self.device)

    def sort_and_tune(self,epoch):
        
        tune_ratio = (self.tune_ratio_range - 0.01) * (epoch-self.duration[0]) / (self.duration[1]-self.duration[0]) + 0.01
        self.load_bit_config()
        self.compute_avg_bit()
        if self.x_bit_avg > self.act_target:
            x_k = int(len(self.x_err_total) * tune_ratio)
            assert len(self.bit_act) == len(self.x_err_total)
            x_v,x_indices = torch.topk(self.x_err_total,x_k,largest=False)
            self.bit_act[x_indices] -= 1
            self.bit_act = torch.clamp(self.bit_act,min=0)
            print('Values of selected act: ',x_v)
            print('indices of selected act: ',x_indices,'total: ',len(self.bit_act))
        if self.w_bit_avg > self.wgt_target:
            w_k = int(len(self.w_err_total) * tune_ratio)        
            assert len(self.bit_wgt) == len(self.w_err_total)
            w_v,w_indices = torch.topk(self.w_err_total,w_k,largest=False)
            self.bit_wgt[w_indices] -= 1
            self.bit_wgt = torch.clamp(self.bit_wgt,min=0)
            print('Values of selected wgt: ',w_v)
            print('indices of selected wgt: ',w_indices,'total: ',len(self.bit_wgt))
        
        self.save_bit_config()
        self.init_err_buffer()
    
    def broadcast(self,src_rank):
        dist.broadcast(self.w_err_total,src_rank)
        dist.broadcast(self.x_err_total,src_rank)

    def show_avg_bit(self):
        for name,module in self.model.named_modules():
            if isinstance(module,(QuantConv2d,QuantLinear)):
                bit_wgt_avg = module.bit_wgt.float().mean().item()
                bit_act_avg = module.bit_act.float().mean().item()
                bit_wgt_min = module.bit_wgt.float().min().item()
                bit_act_min = module.bit_act.float().min().item()
                alpha_wgt_min = module.alpha_wgt.min().item()
                alpha_act_min = module.alpha_act.min().item()
                print(name,' bw_avg:%.2f ba_avg:%.2f  bw_min:%.1f ba_min:%.1f aw_min:%.3f aa_min:%.3f'%(bit_wgt_avg,bit_act_avg,bit_wgt_min,bit_act_min,alpha_wgt_min,alpha_act_min))
                
    def get_first_last_name(self):
        names = []
        if isinstance(self.model,(torch.nn.parallel.DataParallel,torch.nn.parallel.DistributedDataParallel)):
            for name,module in self.model.named_modules():
                names.append(name)
            self.first_last_module_name = [names[2]] + [names[-1]] 
        else:
            for name,module in self.model.named_modules():
                names.append(name)
            self.first_last_module_name = [names[1]] + [names[-1]]
        
   
if __name__ == '__main__':
    # model = torch.nn.parallel.DataParallel(model)
    # x = torch.randn(1,3,224,224)
    # y = torch.randn(1000)
    # tuner = QuantTune(model,2.0,2.0)
    # print('ok')

    # model.train()
    # pred = model(x)
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(pred,y)
    # loss.backward()
    # tuner.compute_avg_bit()
    # print(tuner.x_bit_avg,tuner.w_bit_avg)
    # tuner.compute_quant_residual()
    # tuner.sort_and_tune(8)
    # tuner.compute_avg_bit()
    # print(tuner.x_bit_avg,tuner.w_bit_avg)
    local_rank = 0
    # dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    model = resnet18()
    model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    model = torch.nn.parallel.DataParallel(model)
    print('000')
    tuner = QuantTune(model,2.0,2.0,layerwise=False,duration=(0,1),device=torch.device(local_rank))
    print('111')
    x = torch.randn(1,3,224,224).cuda(local_rank)
    y = torch.randn(1000).cuda(local_rank)
    y_pred = model(x)
    print('222')
    
    
    print('ok')
    