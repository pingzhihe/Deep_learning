#Scalars are represented by tensors with only one element.
#标量由只有一个元素的张量显示
import torch
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print("x + y = {}, x *y ={}, x/y = {}".format(x+y,x*y,x/y))

#You can view vectors as lists of scalars
#你可以将向量视为标量的列表
x = torch.arange(4)
#print(x)
#print(x[3]) #Use index to find element in tensor
#print(len(x))

A = torch.arange(20).reshape(5,4)
#print(A.T)  #Transpose of a martix

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
#print(B == B.T)     #Symmetric matrix: A = AT

X = torch.arange(24).reshape(2,3,4)     #3D matrix
#print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone()   #Allocate a new memory for B
#print(A, A+B)   

x = torch.arange(4, dtype=torch.float32) #Calculate the sum of elements
#print(x,x.sum()) 

#A = torch.arange(20*2, dtype=torch.float32).reshape(2,5,4)
#print(A.shape,A.sum())

print(A.mean())
y = torch.tensor([1.,1,1,1])
#print(x,y,torch.dot(x,y)) #dot multiply

print(torch.mv(A,x)) 

B = torch.ones(4,3)
print(torch.mm(A, B))

u = torch.tensor([3., -4])  #Norm
print(torch.norm(u))
print(torch.abs(u).sum())   #L1Norm

print("The frobenius norm is: {}".format(torch.norm(torch.ones(4,9))))
