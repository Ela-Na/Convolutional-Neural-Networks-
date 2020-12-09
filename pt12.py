import numpy as np
import torch

a = torch.ones(3,4)
print(a)

from torch import autograd

m = torch.rand(1,3)

print(m)

n = autograd.Variable(torch.rand(1, 3))
print(n)

from torch.autograd import Variable

r = Variable(torch.rand(1, 3))
print(r)


