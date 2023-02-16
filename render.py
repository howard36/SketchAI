import torch

def render_strokes(strokes, G, K):
    grid = torch.ones((G,G), dtype=torch.float)
    for stroke in strokes:
        grid = torch.min(grid, render_stroke(stroke * G, G, K))
    return grid # G x G

def render_stroke(stroke, G, K):
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
    darkness = distances/K # (n-1) x G x G
    return torch.min(darkness, dim=0).values # G x G

# distance from point p to line segment v--w
def dist_line_segment(p, v, w):
    d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
    dot = (p-v) * (w-v)
    dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
    t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
    t = torch.clamp(t, min=0, max=1) # (n-1) x G x G
    proj = v + t * (w-v) # (n-1) x G x G x 2
    return torch.linalg.norm(p-proj, dim=3)
   