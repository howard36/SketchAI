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
        self.render = pydiffvg.RenderFunction.apply
        self.num_control_points = torch.tensor([2])
        # base point between each set of control points
        self.points_per_curve = torch.sum(self.num_control_points).item() + len(self.num_control_points)
    
    def random_params(self):
        shapes = [torch.rand((self.points_per_curve, 2), dtype=torch.float, requires_grad=True) for _ in range(64)]
        colors = [torch.rand((4), dtype=torch.float, requires_grad=True) for _ in range(64)]
        background_color = [torch.rand((4), dtype=torch.float, requires_grad=True)]
        return shapes + colors + background_color
    
    def get_custom_loss(self, params):
        # XING loss from LIVE paper
        # tries to prevent each segment of Bezier path from self-intersecting
        N = len(params)//2
        shapes = params[:N]
        relu = nn.ReLU()
        
        tot = torch.Tensor([0])
        for i in range(N):
            shape = shapes[i]
            avg = torch.Tensor([0])
            
            # For each path A (base point) - B (control point) - C (control point) - D (base point)
            #   add XING loss
            for j in range(len(self.num_control_points)):
                idx = j*3
                A = shape[idx]
                B = shape[idx+1]
                C = shape[idx+2]
                D = shape[(idx+3)%len(shape)]
                
                AB = B - A
                CD = D - C
                
                cross = AB[0]*CD[1] - AB[1]*CD[0]
                magAB = torch.norm(AB)
                magCD = torch.norm(CD)
                
                D1 = (torch.sign(cross)+1) / 2 # 0 if angle is <180, 1 otherwise
                D2 = cross / (magAB * magCD) # sin(angle between AB and CD)
                
                loss = D1*relu(-D2) + (1-D1)*relu(D2)
                avg += loss
            avg /= len(self.num_control_points)
            tot += avg
        
        res = tot/N
        return res.to(device)

    def forward(self, params):
        N = len(params)//2
        shapes = params[:N]
        colors = params[N:N*2]
        background_color = params[-1]
        
        paths = []
        shape_groups = []
        
        # Background: one giant closed bezier curve over the whole canvas
        paths.append(pydiffvg.Path(
            num_control_points = torch.Tensor([2]),
            points = torch.Tensor([[-1.0,-1.0], [5.0, -1.0], [-1.0, 5.0]])*self.G,
            is_closed = True
        ))
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids = torch.tensor([0]),
            fill_color = background_color
        ))
        
        for i in range(N):
            path = pydiffvg.Path(
                num_control_points = self.num_control_points,
                points = shapes[i]*self.G,
                is_closed = True
            )
            shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([i+1]), fill_color = colors[i])
            
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
        img = torch.permute(img, (2, 0, 1))
        
        return img
