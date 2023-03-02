import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from similarity2 import similarity
from straight_renderer import StraightRenderer

GRID_SZ = 112

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

target = []
for i in range(GRID_SZ):
    target.append([])
    for j in range(GRID_SZ):
        d = abs(math.sqrt((i-GRID_SZ/2)**2 + (j-GRID_SZ/2)**2) - GRID_SZ/3)
        target[i].append(min(d/5, 1))
target = torch.tensor(target)
target = target.to(device)

renderer = StraightRenderer(GRID_SZ)
params = renderer.random_params()
renderer.to(device)

# grid = torch.ones((GRID_SZ, GRID_SZ, 3), requires_grad=True)

optimizer = optim.Adam(params, lr=0.02)

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
    params_on_device = [param.to(device) for param in params]
    grid = renderer(params_on_device) # 3 x G x G

    sim_loss = similarity("orange cat", grid)
    custom_loss = renderer.get_custom_loss(params)
    loss = sim_loss + 0.005 * custom_loss
    loss.backward()

    optimizer.step()
    scheduler.step()

    grid = torch.permute(grid, (1, 2, 0))
    axim.set_data(grid.cpu().detach().numpy())
    fig.canvas.flush_events()
    print(f'Loss: {loss.item()}')
 
