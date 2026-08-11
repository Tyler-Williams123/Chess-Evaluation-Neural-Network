import Network
import torch

data = torch.load("trainingData.pt")
inputs = data["inputs"]
evals = data["evals"]

model = model = Network.Network(837, 1, 32)
model.load_state_dict(torch.load("chess_eval.pt"))

# print(evals)
# print(torch.mean(evals))

y_hats = model(inputs)
for i in range(5):
    item = torch.randint(low=0, high=inputs.shape[0], size=(1,)).item()
    prediction = torch.atanh(y_hats[item]) * 1500
    real = torch.atanh(evals[item]) * 1500
    print("prediction: " + str(prediction) + "\n")
    print("real: " + str(real) + "\n")