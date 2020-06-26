import torch
import time
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# DATA
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

print(x.shape)
print(y.shape)

plt.plot(x_numpy, y_numpy, 'ro')

# MODEL
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# COST FUNCTION AND OTIMIZER
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print (model.parameters())

# LOOP OF TRAIN
num_epochs = 1000 
contador_custo = []
for epoch in range(num_epochs):
  #forward pass and loos
  y_hat = model(x)
  loss = criterion(y_hat, y)
  contador_custo.append(loss)

  
  #backward pass (calculate gradients)
  loss.backward()

  # update weights
  optimizer.step()

  if (epoch+1)%10 == 0:
      print('Epoch: ', epoch)
      print('Custo: {:.20f}'.format(loss.item())) 
      print('Coeficientes: ')
      print('m: {:.20f}'.format(model.weight.data.detach().item()))
      print('m (gradiente): {:.20f}'.format(model.weight.grad.detach().item()))
      print('b: {:.20f}'.format(model.bias.data.detach().item()))
      print('b (gradiente): {:.20f}'.format(model.bias.grad.detach().item()))
      #for p in model.parameters():
      #  print('{:.2f}'.format(p.data.detach().item()))
      #  print('{:.2f}'.format(p.grad.detach().item()))
      previsao_final = y_hat.detach().numpy()
      plt.plot(x_numpy, y_numpy, 'ro') 
      plt.plot(x_numpy, previsao_final, 'b')
      plt.show()
      
  # clean otimizer
  optimizer.zero_grad()

# plot the graph of cust function 
print("GRÁFICO DA FUNÇÃO DE CUSTO")
plt.plot(contador_custo, 'b')
plt.show()

print('Other references:/nhttps://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd')