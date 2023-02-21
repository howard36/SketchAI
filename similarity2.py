import torchvision
import torchvision.transforms as transforms

import clip

NUM_AUGS = 4 # More is usually better but slower and more CUDA memory

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

    img_augs = [augment_trans(img) for _ in range(NUM_AUGS)]
    im_batch = torch.cat(img_augs)
    image_features = model.encode_image(im_batch)

    print(text_features.shape, image_features.shape)
    loss = 0
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
