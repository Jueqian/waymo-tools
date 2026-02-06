import pickle

# pkl path
path = "/data/repo/waymo/converted/training/0.pkl" 

with open(path, 'rb') as f:
    data = pickle.load(f)

def inspect_structure(obj, indent=0):
    spacing = "  " * indent
    if isinstance(obj, dict):
        print(f"{spacing}Dictionary keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{spacing}Key [{k}]:")
            inspect_structure(v, indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{spacing}List/Tuple (length: {len(obj)})")
        if len(obj) > 0:
            inspect_structure(obj[0], indent + 1)
    elif hasattr(obj, 'shape'): # for NumPy arrays or PyTorch tensors
        print(f"{spacing}{type(obj).__name__} with shape: {obj.shape}")
    else:
        print(f"{spacing}Value: {type(obj).__name__}")

print("\n\n\n--- Data Structure ---")
inspect_structure(data)