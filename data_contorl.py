import torch
import os
import pandas as pd

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
x = torch.arange(12)
X = x.reshape(3,4)
'''
print(a ,b)
print(a + b)
print(X[-1])
'''
os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') #Name of colum
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
data

