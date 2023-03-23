import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
from similarity2 import similarity
from closed_bezier_renderer import ClosedBezierRenderer

GRID_SZ = 112

device = 'cuda' if torch.cuda.is_available() else 'cpu'

renderer = ClosedBezierRenderer(GRID_SZ)
shapes = []
colors = []
renderer.to(device)

plt.ion()
grid = torch.ones((GRID_SZ, GRID_SZ))
fig, ax = plt.subplots()
axim = ax.imshow(grid.cpu().detach().numpy())

circle_offsets = []
for i in range(3):
    theta = i/6 * math.pi*2
    circle_offsets.append([math.cos(theta), math.sin(theta)])
circle_offsets = torch.Tensor(circle_offsets)
circle_offsets /= 10

shapes = []
colors = []

for i in range(10):
    # Initialize shape as a circle around a random position
    center = torch.rand((2)) * 0.5
    shape = center + circle_offsets
    shape.requires_grad_()
    color = torch.rand((4), dtype=torch.float, requires_grad=True)
    shapes.append(shape)
    colors.append(color)
    params = shapes + colors
    
    optimizer = optim.Adam(params, lr=0.02)
    
    for j in range(100):    
        optimizer.zero_grad()
        params_on_device = [param.to(device) for param in params]
        grid = renderer(params_on_device) # 3 x G x G

        sim_loss = similarity("orange cat", grid)
        custom_loss = renderer.get_custom_loss(params)
        loss = sim_loss + 0.005 * custom_loss
        loss.backward()

        optimizer.step()

        t1 = time.time()

        grid = torch.permute(grid, (1, 2, 0))
        axim.set_data(grid.cpu().detach().numpy())
        fig.canvas.flush_events()

        t2 = time.time()
        print(f'Loss: {loss.item()}')
