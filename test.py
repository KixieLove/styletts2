import torch, pprint

path = r"logs\angelina_es\epoch_1st_00048.pth"   # or any other .pth

ckpt = torch.load(path, map_location="cpu")

print("Top-level keys:", ckpt.keys())
print("Epoch:", ckpt.get("epoch"))
print("Val loss:", ckpt.get("val_loss"))
print("Iters:", ckpt.get("iters"))

net = ckpt["net"]
print("\nModules in 'net':")
pprint.pp(list(net.keys()))

# Show a few parameter shapes per module
for name, sd in list(net.items()):
    print(f"\n== {name} ==")
    # sd is a state_dict: mapping param_name -> tensor
    count = 0
    for p_name, tensor in sd.items():
        print(" ", p_name, tuple(tensor.shape))
        count += 1
        if count >= 3:  # just show first 3 params per module
            break
