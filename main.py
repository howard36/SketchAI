import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from similarity2 import similarity
from render import StrokeRenderer

GRID_SZ = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# check for consecutive points
strokes = [torch.rand((5, 2), dtype=torch.float, requires_grad=True) for _ in range(10)]
# stroke = torch.tensor([(0.3, 0.3), (0.3, 0.7), (0.7, 0.7), (0.7,0.3), (0.2, 0.2), (0.3, 0.7), (0.7, 0.7), (0.7,0.3)], requires_grad=True)
# target = stroke_to_picture([torch.tensor([(0,0), (99,99)], dtype=torch.float)])


target = []
for i in range(GRID_SZ):
    target.append([])
    for j in range(GRID_SZ):
        d = abs(math.sqrt((i-GRID_SZ/2)**2 + (j-GRID_SZ/2)**2) - GRID_SZ/3)
        target[i].append(min(d/5, 1))
target = torch.tensor(target)
target = target.to(device)

renderer = StrokeRenderer(GRID_SZ, 5)
renderer.to(device)

# grid = torch.ones((GRID_SZ, GRID_SZ, 3), requires_grad=True)

optimizer = optim.Adam(strokes, lr=0.02)

plt.ion()
# grid = torch.ones((GRID_SZ, GRID_SZ))
grid = target
fig, ax = plt.subplots()
axim = ax.imshow(grid.cpu().detach().numpy())

for i in range(1000):
    optimizer.zero_grad()
    strokes_on_device = [stroke.to(device) for stroke in strokes]
    grid = renderer(strokes_on_device)
    # grid = torch.stack((grid,grid,grid), dim=2)
    # sim_loss = similarity("a sketch of a cat", grid)
    sim_loss = torch.linalg.norm(grid-target)**2 / GRID_SZ**2
    far_loss = sum(map(lambda stroke: torch.linalg.norm(stroke-0.5)**2, strokes)) # penalizes points that are far from the center
    smooth_loss = sum(map(lambda stroke: torch.linalg.norm(stroke[1:]-stroke[:-1])**2, strokes))
    loss = sim_loss + 0.001*far_loss + 0.01*smooth_loss
    loss.backward()
    optimizer.step()

    axim.set_data(grid.cpu().detach().numpy())
    fig.canvas.flush_events()
    print(f'Loss: {loss.item()}')
 
