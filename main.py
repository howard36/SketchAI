import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from similarity import similarity
from render import render_strokes

GRID_SZ = 224

# check for consecutive points
stroke = torch.rand((50, 2), dtype=torch.float, requires_grad=True)
# stroke = torch.tensor([(0.3, 0.3), (0.3, 0.7), (0.7, 0.7), (0.7,0.3), (0.2, 0.2), (0.3, 0.7), (0.7, 0.7), (0.7,0.3)], requires_grad=True)
# target = stroke_to_picture([torch.tensor([(0,0), (99,99)], dtype=torch.float)])


target = []
for i in range(GRID_SZ):
    target.append([])
    for j in range(GRID_SZ):
        d = abs(math.sqrt((i-GRID_SZ/2)**2 + (j-GRID_SZ/2)**2) - GRID_SZ/3)
        target[i].append(min(d/20, 1))
target = torch.tensor(target)

# grid = torch.ones((GRID_SZ, GRID_SZ, 3), requires_grad=True)

optimizer = optim.Adam([stroke], lr=0.02)

plt.ion()
# grid = torch.ones((GRID_SZ, GRID_SZ))
grid = target
fig, ax = plt.subplots()
axim = ax.imshow(grid.detach().numpy())

for i in range(1000):
    optimizer.zero_grad()
    grid = render_strokes([stroke], GRID_SZ, 5)
    # print(grid.sum())
    # grid = torch.stack((grid,grid,grid), dim=2)
    # grid.retain_grad()
    loss = torch.linalg.norm(grid-target)**2 + 0.001*GRID_SZ*GRID_SZ*torch.linalg.norm(stroke-0.5)**2
    # loss = similarity("a sketch of a cat", grid)
    loss.backward()
    optimizer.step()

    axim.set_data(grid.detach().numpy())
    fig.canvas.flush_events()
    print(f'Loss: {loss.item()}')
 
