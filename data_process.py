import torch
import os
import pandas as pd

os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  #Name of column
    f.write('NA,Pave,127500\n')     # Every row shows a data sample
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
#print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] #Iloc: index location
inputs = inputs.fillna(inputs.mean())   #Fill the not-number column to the mean of this column withnum
inputs = pd.get_dummies(inputs, dummy_na=True)  #Convert the string type to num type

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)

