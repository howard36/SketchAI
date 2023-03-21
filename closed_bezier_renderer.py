import torch
from torch import nn
from renderer import Renderer
import pydiffvg

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pydiffvg.set_use_gpu(torch.cuda.is_available())

class ClosedBezierRenderer(nn.Module, Renderer):
    def __init__(self, G):
        super(ClosedBezierRenderer, self).__init__()
        self.G = G
        self.N = 10
        self.render = pydiffvg.RenderFunction.apply
        self.num_control_points = torch.tensor([2, 2])
        # base point between each set of control points
        self.points_per_curve = torch.sum(self.num_control_points).item() + len(self.num_control_points)
    
    def random_params(self):
        shapes = [torch.rand((self.points_per_curve, 2), dtype=torch.float, requires_grad=True) for _ in range(self.N)]
        colors = [torch.rand((4), dtype=torch.float, requires_grad=True) for _ in range(self.N)]
        return shapes + colors
    
    def get_custom_loss(self, params):
        return 0

    def forward(self, params):
        shapes = params[:self.N]
        colors = params[self.N:self.N*2]
        
        paths = []
        shape_groups = []
        for i in range(self.N):
            path = pydiffvg.Path(
                num_control_points = self.num_control_points,
                points = shapes[i]*self.G,
                is_closed = True
            )
            shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([i]), fill_color = colors[i])
            
            paths.append(path)
            shape_groups.append(shape_group)
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.G, # canvas width
            self.G, # canvas height
            paths,
            shape_groups
        )
        img = self.render(
            self.G, # width
            self.G, # height
            2, # num_samples_x
            2, # num_samples_y
            0, # seed
            torch.ones((self.G, self.G, 4)), # background_image
            *scene_args
        )
        img = img[:,:,:3] # remove alpha channel
#         import matplotlib.pyplot as plt
#         import time
#         pydiffvg.imwrite(img.cpu(), 'test.png')
#         plt.imshow(img.cpu().detach().numpy())
        
        img = torch.permute(img, (2, 0, 1))
        
        print(img)
        return img
