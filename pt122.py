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

print (type(r))

#w = torch.ones(4, requires_grad=True)
#w.grad.zero_()
'''
optimizer = torch.optim.SGD(w, lr=0.01)
optimizer.step()
optimizer.zero_grad()
'''
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

#forward pass
y_hat = w*x
loss= (y_hat-y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)


