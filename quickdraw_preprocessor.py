import pandas as pd
from straight_renderer import StraightRenderer
import os
from tqdm import tqdm
import torch

GRID_SZ = 256
renderer = StraightRenderer(GRID_SZ)

# list of classes with '.ndjson' appended
classes = os.listdir('quickdraw-dataset/simplified')

for c in tqdm(classes):
    # Get all the drawings under this class
    df = pd.read_json(f"quickdraw-dataset/simplified/{c}", lines=True, nrows=100)
    
    # Randomly subsample 1000 drawings that were recognized
    df = df[df['recognized'] == True].sample(n=10)
    raw_drawings = df['drawing'].tolist()
    
    # Process the drawings into our stroke format
    drawings = [] # This will contain the drawings after they have been processed into torch Tensors
    for drawing in raw_drawings:
        # each drawing is a list of strokes
        strokes = []
        for stroke in drawing:
            # each stroke is [[x coords], [y coords]]
            # need to convert to torch.Tensor of shape (n,2)
            s = torch.Tensor(stroke)
            s[[0,1]] = s[[1,0]]
            s = s.T
            s /= GRID_SZ
            strokes.append(s)
        drawings.append(strokes)
    
    class_name = c[:-7].replace(' ', '_')
    directory = f"quickdraw-dataset/rendered/{class_name}"
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i, drawing in enumerate(drawings):
        strokes = drawing
        thicknesses = [torch.Tensor([1]) for _ in range(len(drawing))]
        colors = [torch.Tensor([1, 1, 1]) for _ in range(len(drawing))]
        img = renderer(strokes + thicknesses + colors)
        img = torch.permute(img, (1, 2, 0))
        torch.save(img, f"{directory}/{i:03d}.pt")
