import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

G = 100
K = 5

def stroke_to_picture(strokes):
    grid = torch.zeros((G,G), dtype=torch.float)
    for stroke in strokes:
        grid = torch.max(grid, render_stroke(stroke))
    return grid

def render_stroke(stroke):
    n = len(stroke)
    vs = stroke[:-1].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
    vs = torch.tile(vs, (1, G, G, 1)) # (n-1) x G x G x 2

    ws = stroke[1:].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
    ws = torch.tile(ws, (1, G, G, 1)) # (n-1) x G x G x 2

    idxs = torch.arange(G)
    x_coords, y_coords = torch.meshgrid(idxs, idxs, indexing='ij') # G x G
    coords = torch.stack((x_coords, y_coords), dim=2).reshape(1,G,G,2) # 1 x G x G x 2
    coords = torch.tile(coords, (n-1, 1, 1, 1)) # (n-1) x G x G x 2 

    distances = dist_line_segment(coords, vs, ws) # (n-1) x G x G
    darkness = 1 - distances/K # (n-1) x G x G
    return torch.max(darkness, dim=0).values # G x G

# distance from point p to line segment v--w
# assumes v != w
def dist_line_segment(p, v, w):
    d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
    dot = (p-v) * (w-v)
    dot_sum = torch.sum(dot, dim=3) / d**2
    t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
    t = torch.clamp(t, min=0, max=1) # (n-1) x G x G
    proj = v + t * (w-v) # (n-1) x G x G x 2
    return torch.linalg.norm(p-proj, dim=3)

# check for consecutive points
stroke = torch.rand((100, 2), dtype=torch.float, requires_grad=True)
# target = stroke_to_picture([torch.tensor([(), (99,99)], dtype=torch.float)])

target = []
for i in range(G):
    target.append([])
    for j in range(G):
        if abs(math.sqrt((i-G/2)**2 + (j-G/2)**2) - G/3) < 5:
            target[i].append(1.0)
        else:
            target[i].append(0.0)

target = torch.tensor(target)

optimizer = optim.Adam([stroke], lr=0.01)

for i in range(5000):
    optimizer.zero_grad()
    grid = stroke_to_picture([stroke * G])
    # loss = -torch.sum(grid * target)
    loss = torch.linalg.norm(grid-target)**2
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        _, axs = plt.subplots(2)
        axs[0].imshow(target)
        axs[1].imshow(grid.detach().numpy())
        plt.show()

print(stroke)


# print(grid)
# for i in range(G):
#     for j in range(G):
#         print(int(9*grid[i][j]), end=' ')
#     print()

# print(dist_line_segment(p2t((5,4)), p2t((1,2)), p2t((5,4))))
    