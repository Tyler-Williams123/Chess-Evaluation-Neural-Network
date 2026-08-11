import torch
import Network

data = torch.load("trainingData.pt")

inputs = data["inputs"]
evals = data["evals"]

# inputs = inputs[]
# evals = evals[]

evals = evals.unsqueeze(1)

model = Network.Network(837, 1, 32)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epcoh in range(5000):
    y_hats = model(inputs)

    loss = loss_fn(y_hats, evals)

    optim.zero_grad()
    loss.backward()
    optim.step()
    if(epcoh % 100 == 0):
        print(loss.item())

torch.save(model.state_dict(), "chess_eval.pt")