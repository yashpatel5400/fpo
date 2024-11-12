import os
PDE_DIR  = lambda pde : os.path.join("data", pde)
DATA_FN  = lambda pde : os.path.join(PDE_DIR(pde), "data.pkl")
MODEL_FN = lambda pde : os.path.join(PDE_DIR(pde), "model.pt")