import torch
from torch import nn
from renderer import Renderer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BezierRenderer(nn.Module, Renderer):
    def __init__(self, G, P):
        super(BezierRenderer, self).__init__()
        self.G = G # grid dimensions (G x G)
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
    
    def random_params(self):
        NUM_STROKES = 10
        strokes = [torch.rand((2, 4), dtype=torch.float, requires_grad=True) for _ in range(NUM_STROKES)]
        thicknesses = [torch.rand((1), dtype=torch.float, requires_grad=True) for _ in range(NUM_STROKES)]
        colors = [torch.rand((3), dtype=torch.float, requires_grad=True) for _ in range(NUM_STROKES)]
        return strokes + thicknesses + colors
    
    def get_custom_loss(self, params):
        n = len(params)
        control_points = params[:n//3][:][:2]
        derivs = params[:n//3][:][2:]
        far_loss = sum(map(lambda stroke: torch.max(torch.linalg.norm(stroke-0.5), torch.Tensor([0.25]).to(device))**2, control_points)) # penalizes points that are far from the center
        far_loss_derivs = sum(map(lambda stroke: torch.max(torch.linalg.norm(stroke), torch.Tensor([0.25]).to(device))**2, derivs))
        smooth_loss = sum(map(lambda stroke: torch.linalg.norm(stroke[1:]-stroke[:-1])**2, control_points))
        return 0.3*(far_loss + far_loss_derivs) + 0.7*smooth_loss
    
    def forward(self, curves):
        grid = torch.ones((self.G, self.G), dtype=torch.float)
        grid = grid.to(device)
        for curve in curves:
            stroke = self.curve_to_stroke(curve)
            grid = torch.min(grid, self.render_stroke(stroke * self.G))
        return grid # G x G

    def forward(self, params):
        n = len(params)
        strokes = params[:n//3]
        thicknesses = params[n//3:2*n//3]
        colors = params[2*n//3:]

        grid = torch.zeros((3, self.G, self.G), dtype=torch.float)
        grid = grid.to(device)
        for i in range(len(strokes)):
            stroke = self.curve_to_stroke(strokes[i]) * self.G
            # stroke = torch.clamp(strokes[i], min=0.0, max=1.0) * self.G
            thickness = torch.max(thicknesses[i]*2 + 0.5, torch.Tensor([0.5]).to(device))
            color = torch.clamp(colors[i], min=0.0, max=1.0)
            grid = torch.max(grid, self.render_stroke(stroke, thickness, color)) # TODO: update this to stamp colors instead
        return grid

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
        sample_pts = torch.matmul(control_pts, self.weights.to(device)) # (n-1) x 2 x P
        # print(sample_pts)
        # assert(False)
        sample_pts = torch.permute(sample_pts, (0, 2, 1)) # (n-1) x P x 2
        sample_pts = torch.reshape(sample_pts, (-1, 2)) # (n-1)P x 2
        sample_pts = torch.cat([sample_pts, pts[-1:]]) # [(n-1)P + 1] x 2
        return sample_pts

    def render_stroke(self, stroke, t, color):
        n = len(stroke)
        vs = stroke[:-1].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        vs = torch.tile(vs, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        ws = stroke[1:].reshape((-1,1,1,2)) # (n-1) x 1 x 1 x 2
        ws = torch.tile(ws, (1, self.G, self.G, 1)) # (n-1) x G x G x 2

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)) # (n-1) x G x G x 2
        distances = self.dist_line_segment(coords, vs, ws) # (n-1) x G x G
        darknesses = torch.clamp((2*t - distances)/(2*t), min=0.0, max=1.0) # (n-1) x G x G
        darknesses = torch.max(darknesses, dim=0).values # G x G
        return torch.stack((darknesses*color[0], darknesses*color[1], darknesses*color[2]), dim=0)

    # distance from point p to line segment v--w
    def dist_line_segment(self, p, v, w):
        d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
        dot = (p-v) * (w-v)
        dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
        t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
        t = torch.clamp(t, min=0, max=1) # (n-1) x G x G
        proj = v + t * (w-v) # (n-1) x G x G x 2
        return torch.linalg.norm(p-proj, dim=3)
    