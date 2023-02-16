import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from similarity import similarity
from render import render_strokes

# check for consecutive points
# stroke = torch.rand((10, 2), dtype=torch.float, requires_grad=True)
# stroke = torch.tensor([(0.3, 0.3), (0.3, 0.7), (0.7, 0.7), (0.7,0.3), (0.2, 0.2)], requires_grad=True)
# target = stroke_to_picture([torch.tensor([(0,0), (99,99)], dtype=torch.float)])


GRID_SZ = 224
# target = []
# for i in range(GRID_SZ):
#     target.append([])
#     for j in range(GRID_SZ):
#         if abs(math.sqrt((i-GRID_SZ/2)**2 + (j-GRID_SZ/2)**2) - GRID_SZ/3) < 5:
#             target[i].append(1.0)
#         else:
#             target[i].append(0.0)

# target = torch.tensor(target)

grid = torch.ones((GRID_SZ, GRID_SZ, 3), requires_grad=True)

optimizer = optim.Adam([grid], lr=0.1)

plt.ion()
# grid = target
fig, ax = plt.subplots()
axim = ax.imshow(grid.detach().numpy())

for i in range(1000):
    optimizer.zero_grad()
    # grid = render_strokes([stroke], GRID_SZ, 5)
    # grid = torch.stack((grid,grid,grid), dim=2)
    # loss = -torch.sum(grid * target)
    # loss = torch.linalg.norm(grid-target)**2
    grid.retain_grad()
    # transform = T.Resize((GRID_SZ, GRID_SZ))
    # larger_grid = transform(grid)
    loss = similarity("a photo of a cat", grid)
    loss.backward()
    # print(stroke.grad)
    # print(grid.requires_grad)
    # print(grid.grad)
    optimizer.step()

    axim.set_data(grid.detach().numpy())
    fig.canvas.flush_events()
    print(loss.item())
 
