import numpy as np

import torch

'''
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
# optimizer = torch.optim.SGD(w, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
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
'''

#f = w * x
#f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction

def forward(x):
    return w * x


# loss
    
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()
    
    

# gradient

# MSE = 1/N * (w*x - y)**2
# dj (objective function)/dw = 1/N 2x(w*x -y)
    
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
print(f'prediction before training: f(5) = {forward(5):.3f}')

# training 

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients
    dw = gradient(X,Y,y_pred)
    
    # update weights
    w -= learning_rate * dw
    # steps of running
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after trainng: f(5) = {forward(5):.3f}')
 
       