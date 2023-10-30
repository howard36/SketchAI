import math
import torch
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time

from similarity2 import similarity

GRID_SZ = 112

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

num_warmup_steps = 100
num_total_steps = 800
def warmup(current_step: int):
    if current_step < num_warmup_steps:
        return 0.5 + 0.5 * float(current_step / num_warmup_steps)
    else:
        return 1.0 - (current_step - num_warmup_steps) / (num_total_steps - num_warmup_steps)

plt.ion()
grid = torch.tile(torch.randn(GRID_SZ, GRID_SZ), (3, 1, 1)).to(device).requires_grad_()
optimizer = optim.Adam([grid], lr=0.2)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

fig, ax = plt.subplots()
tmp = torch.permute(grid, (1, 2, 0))
axim = ax.imshow(tmp.cpu().detach().numpy())

for i in range(num_total_steps):
    t0 = time.time()
    
    optimizer.zero_grad()

    sim_loss = similarity("disneyland", grid)
    loss = sim_loss
    loss.backward()

    optimizer.step()
    scheduler.step()
    
    t1 = time.time()

    tmp = torch.permute(grid, (1, 2, 0))
    axim.set_data(tmp.cpu().detach().numpy())
    fig.canvas.flush_events()
    
    t2 = time.time()
    print(f'Loss: {loss.item()}')
    print(f"Total time: {t2-t0}, computation time: {t1-t0}, drawing time: {t2-t1}")
