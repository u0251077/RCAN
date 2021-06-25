import torch
import utility
import data
import model
from option import args
from trainer import Trainer
import time 
import tqdm
import os
from skimage import io, transform
import numpy as np
import imageio
st = time.time()
loader = data.Data(args)
checkpoint = utility.checkpoint(args)
args.model='RCAN'#6s
#args.model='EDSR'#0.02

#mylite
'''
args.n_resgroups=1
args.n_resblocks=1
args.n_feats = 32
args.pre_train ='../model/model_best.pt'
'''
model = model.Model(args, checkpoint)

print(model)

def save_results_nopostfix( filename, save_list, scale):
        st1 = time.time()
        if args.degradation == 'BI':
            filename = filename.replace("LRBI", args.save)
        elif args.degradation == 'BD':
            filename = filename.replace("LRBD", args.save)
        filename = '{}/{}/x{}/{}'.format("../SR/BI/RCAN", args.testset, scale, filename)
        postfix = ('SR', 'LR', 'HR')
        
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / 255)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            #misc.imsave('{}.png'.format(filename), ndarr)
            imageio.imsave('{}.png'.format(filename), ndarr)
            
def prepare( l, volatile=False):
        device = torch.device('cuda')
        def _prepare(tensor):
            #tensor = tensor.half() 
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

scale = 'x2'


#ckp = ckp

loader_test = loader.loader_test
model = model
optimizer = utility.make_optimizer(args, model)
scheduler = utility.make_scheduler(args, optimizer)

if args.load != '.':
    self.optimizer.load_state_dict(
        torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
    )
    for _ in range(len(ckp.log)): scheduler.step()




epoch = scheduler.last_epoch + 1


model.eval()

timer_test = utility.timer()
with torch.no_grad():
    
            idx_scale=0
            scale=2
            eval_acc = 0
            loader_test.dataset.set_scale(idx_scale)

            lr = io.imread('../LR/LRBI/Set5/x2/123.jpg')
            lr = np.transpose(lr, (2, 0, 1))
                  
            lr = lr.reshape((1,) + lr.shape) 

            
            
            lr = torch.Tensor(lr.astype(np.float32))
            
            
            for a in range(0,1):# for measuring speed
                        st = time.time()

                        start= time.time()

            


                        lr = prepare([lr])[0]

                        sr = model(lr, idx_scale)

                        sr = utility.quantize(sr, args.rgb_range)

                        save_list = [sr]
                        #print(save_list)

            
                        save_results_nopostfix("result", save_list, 2)



