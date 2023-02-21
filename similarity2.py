import torch
import torchvision
import torchvision.transforms as transforms

import clip

NUM_AUGS = 8 # More is usually better but slower and more CUDA memory

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device, jit=False)

augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def similarity(prompt, img):
    text_input = clip.tokenize(prompt).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_input)

    img_augs = []
    for n in range(NUM_AUGS):
        img_augs.append(augment_trans(img))

    # img_augs = [augment_trans(img) for _ in range(NUM_AUGS)]
    im_batch = torch.stack(img_augs, dim=0)
    image_features = model.encode_image(im_batch)

    loss = 0
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n], dim=1)
    loss /= NUM_AUGS

    return loss
