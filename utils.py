import os
DATA_FN  = lambda pde : os.path.join("data", pde, "data.pkl")
MODEL_FN = lambda pde : os.path.join("data", pde, "model.pt")