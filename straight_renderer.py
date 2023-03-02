import torch
from torch import nn
from renderer import Renderer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StraightRenderer(nn.Module, Renderer):
    def __init__(self, G):
        super(StraightRenderer, self).__init__()
        self.G = G
        idxs = torch.arange(G)
        x_coords, y_coords = torch.meshgrid(idxs, idxs, indexing='ij') # G x G
        self.grid_coords = torch.stack((x_coords, y_coords), dim=2).reshape(1,G,G,2).to(device) # 1 x G x G x 2
    
    def random_params(self):
        NUM_STROKES = 10
        strokes = [torch.rand((5, 2), dtype=torch.float, requires_grad=True) for _ in range(NUM_STROKES)]
        thicknesses = [torch.rand((1), dtype=torch.float, requires_grad=True) for _ in range(NUM_STROKES)]
        return strokes + thicknesses
    
    def get_custom_loss(self, params):
        n = len(params)
        strokes = params[:n//2]
        far_loss = sum(map(lambda stroke: torch.max(torch.linalg.norm(stroke-0.5), torch.Tensor([0.25]).to(device))**2, strokes)) # penalizes points that are far from the center
        smooth_loss = sum(map(lambda stroke: torch.linalg.norm(stroke[1:]-stroke[:-1])**2, strokes))
        return far_loss + smooth_loss
    
    def forward(self, params):
        n = len(params)
        strokes = params[:n//2]
        thicknesses = params[n//2:]

        grid = torch.ones((self.G, self.G), dtype=torch.float)
        grid = grid.to(device)
        for i in range(len(strokes)):
            stroke = torch.clamp(strokes[i], min=0.0, max=1.0) * self.G
            thickness = torch.max(thicknesses[i]*2 + 0.5, torch.Tensor([0.5]).to(device))
            grid = torch.min(grid, self.render_stroke(stroke, thickness))
        return grid # G x G

    def render_stroke(self, stroke, t):
        n = len(stroke)
        vs = stroke[:-1].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        vs = torch.tile(vs, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        ws = stroke[1:].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        ws = torch.tile(ws, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)) # (n-1) x G x G x 2
        distances = self.dist_line_segment(coords, vs, ws) # (n-1) x G x G
        darkness = distances/(2*t) # (n-1) x G x G
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
    