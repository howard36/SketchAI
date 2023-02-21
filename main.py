import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from similarity2 import similarity
from render import StrokeRenderer

GRID_SZ = 112

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# check for consecutive points
strokes = []
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

renderer = StrokeRenderer(GRID_SZ, 2)
renderer.to(device)

# grid = torch.ones((GRID_SZ, GRID_SZ, 3), requires_grad=True)

optimizer = optim.Adam(strokes, lr=0.02)

num_warmup_steps = 100 
num_total_steps = 300
def warmup(current_step: int):
    if current_step < num_warmup_steps:
        return 0.5 + 0.5 * float(current_step / num_warmup_steps)
    else:
        return 1.0 - (current_step - num_warmup_steps) / (num_total_steps - num_warmup_steps)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

plt.ion()
# grid = torch.ones((GRID_SZ, GRID_SZ))
grid = target
fig, ax = plt.subplots()
axim = ax.imshow(grid.cpu().detach().numpy())

for i in range(num_total_steps):
    optimizer.zero_grad()
    strokes_on_device = [stroke.to(device) for stroke in strokes]
    grid = renderer(strokes_on_device)

    grid = torch.stack((grid,grid,grid), dim=0)
    sim_loss = similarity("giraffe", grid)
    # sim_loss = torch.linalg.norm(grid-target)**2 / GRID_SZ**2

    far_loss = sum(map(lambda stroke: torch.max(torch.linalg.norm(stroke-0.5), torch.Tensor([0.25]).to(device))**2, strokes)) # penalizes points that are far from the center
    smooth_loss = sum(map(lambda stroke: torch.linalg.norm(stroke[1:]-stroke[:-1])**2, strokes))
    loss = sim_loss + 0.005*far_loss + 0.005*smooth_loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    grid = torch.permute(grid, (1, 2, 0))
    axim.set_data(grid.cpu().detach().numpy())
    fig.canvas.flush_events()
    print(f'Loss: {loss.item()}')
 
