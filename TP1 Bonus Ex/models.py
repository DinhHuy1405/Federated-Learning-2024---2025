import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dim
        self.num_classes = output_dim
        self.fc = nn.Linear(self.input_dimension, self.num_classes, bias=bias)
        

    def forward(self, x):
        return self.fc(x)


class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        self.model = mobilenet_v2(weights=weights)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 10)
        )
        requires_grad_params = sum(1 for i in self.model.parameters() if i.requires_grad)
        print(f"Number of trainable parameters: {requires_grad_params}")

    def forward(self, x):
        return self.model(x)


mobi = Mobilenet()