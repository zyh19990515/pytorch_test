from torchvision import models
alexnet = models.alexnet(pretrained=True)
print(alexnet)