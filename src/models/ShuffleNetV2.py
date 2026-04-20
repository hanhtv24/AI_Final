import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    """
    Test the Encoder consist of ShuffleNet-V2 - FC. Since this model is lightweight, penalty is not applied
    The variable name, as well as file name is because of the writer laziness for experiments
    """
    def __init__(self):
        super(Encoder, self).__init__()        # Backbone Resnet 101
        self.resnet = torchvision.models.shufflenet_v2_x1_0(pretrained=True)  # pretrained ImageNet ShuffleNet-V2
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, 91)

        self.fine_tune()

    # Output is probabilities vector of 91 classes in COCO dataset
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for FC part of the encoder.

        :param fine_tune: Allow?
        """
        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune FC part (this code is a bit hard code)
        for c in list(self.resnet.children())[6:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        for c in list(self.resnet.children())[0:3]:
            for p in c.parameters():
                p.requires_grad = fine_tune