import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import units
import models.GPNN_CAD as cad

# cad.main()

import torch

weight_mask = torch.autograd.Variable(torch.ones((2,3)),requires_grad=True)
s=torch.sum(weight_mask*weight_mask)
print(s)
s.backward()