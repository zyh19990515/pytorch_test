import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
import torch.nn.functional as F
from torchvision import transforms, models
import cv2

def load_image(img_path, max_size=400):
    image = Image.open(img_path)
    if(max(image.size)>max_size):
        size = max_size
    else:
        size = max(image.size)
    image_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))

    ])
    image = image_transform(image).unsqueeze(0)
    return image

def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze(0)
    image = image.transpose(1, 2, 0)
    image = image*np.array((0.229, 0.224, 0.225))+np.array((0.485, 0.456, 0.406))
    for k in range(3):
        for i in range(320):
            for j in range(320):
                if(image[i][j][k]>1.0):
                    image[i][j][k]=1.0
                if(image[i][j][k]<0):
                    image[i][j][k]=abs(image[i][j][k])
    plt.imshow(image)
    if(title is not None):
        plt.title(title)
    else:
        plt.pause(0.1)

def get_features(image, model, layers=None):
    if(layers is None):
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if(name in layers):
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def ContentLoss(target_features, content_features):
    content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])
    return content_loss

def StyleLoss(target_features, style_grams, style_weights):
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer]*F.mse_loss(target_gram, style_gram)
        style_loss +=layer_style_loss/(d*h*w)
    return style_loss

if __name__ == '__main__':
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    vgg.to(device)
    content = load_image(r"D:\code\content_1.jpg").to(device)
    style = load_image(r"D:\code\style_1.jpg").to(device)
    assert style.size() ==content.size()
    plt.ion()
    plt.figure()
    imshow(content, title='content image')
    plt.figure()
    imshow(style, title='style image')
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_gram={}
    for layer in style_features:
        style_gram[layer] = gram_matrix(style_features[layer])
    style_weights = {
        'conv1_1':1.,
        'conv2_1':0.75,
        'conv3_1':0.2,
        'conv4_1':0.2,
        'conv4_2':0.2
    }
    alpha = 1
    beta = 1e5
    show_every = 10
    steps = 2000
    target = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.003)

    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = ContentLoss(target_features, content_features)
        style_loss = StyleLoss(target_features, style_gram, style_weights)
        total_loss = alpha*content_loss + beta*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("ii:", ii)
        if(ii%show_every==0):
            print('Total loss:', total_loss.item())
            plt.figure()
            imshow(target)

    plt.figure()
    imshow(target, 'Target Image')
    plt.ioff()
    plt.show()