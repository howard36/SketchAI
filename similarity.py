from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



def similarity(text, image):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    # plt.imshow(inputs['pixel_values'][0][0])
    # plt.show()
    inputs['pixel_values'] = torch.permute(image, (2,0,1)).unsqueeze(0)
    # inputs['pixel_values'].requires_grad = True
    inputs['pixel_values'].retain_grad()

    outputs = model(**inputs, return_loss=True)

    loss = -outputs.logits_per_image[0]
    # print('inputs grad', inputs['pixel_values'].grad)
    # print('image grad', image.grad)
    return loss
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # return logits_per_image[0]
    # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

