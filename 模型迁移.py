from torchvision import models
rn = models.resnet34(pretrained=True)
print(rn)
