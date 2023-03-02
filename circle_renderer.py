import torch
from torch import nn
from renderer import Renderer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CircleRenderer(nn.Module, Renderer):
    def __init__(self, G):
        super(CircleRenderer, self).__init__()
        self.G = G
        idxs = torch.arange(G)
        x_coords, y_coords = torch.meshgrid(idxs, idxs, indexing='ij') # G x G
        self.grid_coords = torch.stack((x_coords, y_coords), dim=2).to(device) # G x G x 2
    
    def random_params(self):
        NUM_CIRCLES = 20
        positions = [torch.rand((2), dtype=torch.float, requires_grad=True) for _ in range(NUM_CIRCLES)]
        radii = [torch.rand((1), dtype=torch.float, requires_grad=True) for _ in range(NUM_CIRCLES)]
        colors = [torch.rand((3), dtype=torch.float, requires_grad=True) for _ in range(NUM_CIRCLES)]
        return positions + radii + colors
    
    def get_custom_loss(self, params):
        n = len(params)
        positions = params[:n//3]
        far_loss = sum(map(lambda stroke: torch.max(torch.linalg.norm(stroke-0.5), torch.Tensor([0.25]).to(device))**2, positions)) # penalizes points that are far from the center
        return far_loss
    
    def forward(self, params):
        n = len(params)
        positions = params[:n//3]
        radii = params[n//3:2*n//3]
        colors = params[2*n//3:]

        grid = torch.zeros((3, self.G, self.G), dtype=torch.float)
        grid = grid.to(device)
        for i in range(n//3):
            center = torch.clamp(positions[i], min=0.0, max=1.0) * self.G
            radius = torch.max(radii[i]*self.G/16 + 0.5, torch.Tensor([0.5]).to(device))
            color = torch.clamp(colors[i], min=0.0, max=1.0)
            grid = torch.max(grid, self.render_stroke(center, radius, color))
        return grid

    def render_stroke(self, center, radius, color):
        distances = torch.linalg.norm(self.grid_coords - center, dim=2) # G x G
        darknesses = torch.clamp((radius - distances)/(radius), min=0.0, max=1.0) # G x G
        darknesses = 1 - torch.pow(1-darknesses, 4)
        return torch.stack((darknesses*color[0], darknesses*color[1], darknesses*color[2]), dim=0)
    