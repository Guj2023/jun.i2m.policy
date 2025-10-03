import torch
import json

model_dir = input("Enter the model directory: ")
model = torch.jit.load(model_dir + "/policy.pt")

info = model.normalizer.state_dict()
# Convert tensors to lists for JSON serialization
info_serializable = {k: v.tolist() for k, v in info.items()}
with open(model_dir + "/normalizer.json", "w") as f:
    json.dump(info_serializable, f, indent=2)
print("Saved to normalizer.json")
