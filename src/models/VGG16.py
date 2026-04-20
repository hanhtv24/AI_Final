import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    """
    Test the Encoder consist of VGG-16. Since this model contains many drop out layers. Further penalization is not needed
    The variable name, as well as file name is because of the writer laziness for experiments
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # Backbone Resnet 101
        self.resnet = torchvision.models.vgg16(pretrained=True)  # pretrained ImageNet VGG-16
        num_features = self.resnet.classifier[-1].out_features
        self.concat_fc = nn.Sequential(
            nn.ReLU(inplace=True),
            torch.nn.Linear(num_features, 91)
        )
        # print(list(self.resnet.children())[9])
        self.fine_tune()

    # Output is probabilities vector of 91 classes in COCO dataset
    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.concat_fc(out)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for the concat fully connected layer

        :param fine_tune: Allow?
        """
        # Freeze all layers of all vgg part except the new fc part
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune after FC
        # for c in list(self.resnet.children())[9:]:
        #     for p in c.parameters():
        #         p.requires_grad = fine_tune
