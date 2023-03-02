import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StrokeRenderer(nn.Module):
    def __init__(self, G, T, P):
        super(StrokeRenderer, self).__init__()
        self.G = G # grid dimensions (G x G)
        self.T = T # thickness
        self.P = P # number of pieces to split Bezier curve into
        assert(P >= 1)
        idxs = torch.arange(G)
        x_coords, y_coords = torch.meshgrid(idxs, idxs, indexing='ij') # G x G
        self.grid_coords = torch.stack((x_coords, y_coords), dim=2).reshape(1,G,G,2).to(device) # 1 x G x G x 2
        w1 = [(P-t)**3/P**3 for t in range(P)]
        w2 = [3*(P-t)*(P-t)*t/P**3 for t in range(P)]
        w3 = [3*(P-t)*t*t/P**3 for t in range(P)]
        w4 = [t**3/P**3 for t in range(P)]
        self.weights = torch.tensor([w1, w2, w3, w4]) # 4 x P
    
    def forward(self, curves):
        grid = torch.ones((self.G, self.G), dtype=torch.float)
        grid = grid.to(device)
        for curve in curves:
            stroke = self.curve_to_stroke(curve)
            grid = torch.min(grid, self.render_stroke(stroke * self.G))
        return grid # G x G

    def curve_to_stroke(self, curve): # curve is n x 4
        pts, derivs = torch.split(curve, [2, 2], dim=1) # both n x 2
        before = pts - derivs # n x 2
        after = pts + derivs # n x 2
        p1 = pts[:-1] # (n-1) x 2
        p2 = after[:-1] # (n-1) x 2
        p3 = before[1:] # (n-1) x 2
        p4 = pts[1:] # (n-1) x 2
        # print(p1, p2, p3, p4)
        control_pts = torch.stack([p1, p2, p3, p4], dim=2) # (n-1) x 2 x 4
        sample_pts = torch.matmul(control_pts, self.weights) # (n-1) x 2 x P
        # print(sample_pts)
        # assert(False)
        sample_pts = torch.permute(sample_pts, (0, 2, 1)) # (n-1) x P x 2
        sample_pts = torch.reshape(sample_pts, (-1, 2)) # (n-1)P x 2
        sample_pts = torch.cat([sample_pts, pts[-1:]]) # [(n-1)P + 1] x 2
        return sample_pts

    def render_stroke(self, stroke):
        n = len(stroke)
        vs = stroke[:-1].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        vs = torch.tile(vs, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        ws = stroke[1:].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        ws = torch.tile(ws, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)) # (n-1) x G x G x 2
        distances = self.dist_line_segment(coords, vs, ws) # (n-1) x G x G
        darkness = distances/self.T # (n-1) x G x G
        return torch.min(darkness, dim=0).values # G x G

    # distance from point p to line segment v--w
    def dist_line_segment(self, p, v, w):
        d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
        dot = (p-v) * (w-v)
        dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
        t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
        t = torch.clamp(t, min=0, max=1) # (n-1) x G x G
        proj = v + t * (w-v) # (n-1) x G x G x 2
        return torch.linalg.norm(p-proj, dim=3)
    