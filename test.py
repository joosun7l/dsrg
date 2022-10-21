import torch

x_train = torch.FloatTensor([[1], [2], [3],[4],[5]])
y_train = torch.FloatTensor([[2], [4], [6],[8],[3]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.01)

hypothesis = x_train * W + b

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
 hypothesis = x_train * W + b
 cost = torch.mean((hypothesis - y_train) ** 2)
 optimizer.zero_grad()
 cost.backward()
 optimizer.step()
print(hypothesis)