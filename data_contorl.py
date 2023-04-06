import torch
import os
import pandas as pd

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
x = torch.arange(12)
X = x.reshape(3,4)

print(a ,b)
print(a + b)
print(X[-1])
print(a.size())
