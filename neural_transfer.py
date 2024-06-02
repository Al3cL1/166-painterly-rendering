import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
import neural_utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentLoss(nn.Module):
    """
    Content loss layer for neural style transfer.
    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
class StyleLoss(nn.Module):
    """
    Style loss layer for neural style transfer.
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        target = neural_utils.gram_matrix(target_feature).detach()
        self.target = target

    def forward(self, input):
        gram = neural_utils.gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input
    
class Normalization(nn.Module):
    """
    Normalization layer for neural style transfer.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


def losses_and_model(backbone: nn.Module, norm_mean: torch.Tensor, norm_std: torch.Tensor, 
                     style_img: torch.Tensor, content_img: torch.Tensor, 
                     content_layers: list, style_layers: list) -> tuple:
    """
    Trims the backbone and attaches content and style losses.

    Args:
        backbone: A pre-trained model backbone.
        norm_mean: Normalization mean.
        norm_std: Normalization standard deviation.
        style_img: The style image tensor.
        content_img: The content image tensor.
        content_layers: List of content layer names, to be used for calculating content loss.
        style_layers: List of style layer names, to be used for calculating style loss.

    Returns:
        tuple: The trimmed model with added loss layers, and lists of content and style losses.
    """
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    conv_counter = 0
    for layer in backbone.children():
        if isinstance(layer, nn.Conv2d):
            conv_counter += 1
            name = f'conv_{conv_counter}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{conv_counter}'
            layer = nn.ReLU(inplace=False) # for some reason, relu doesent work with losses.
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{conv_counter}'
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_counter}'
        else:
            raise RuntimeError(f'Layer Error')

        model.add_module(name, layer)

        if content_layers and name == content_layers[0]:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{conv_counter}', content_loss)
            content_losses.append(content_loss)
            content_layers.pop(0)

        if style_layers and name == style_layers[0]:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{conv_counter}', style_loss)
            style_losses.append(style_loss)
            style_layers.pop(0)

        if not content_layers and not style_layers:
            break

    return model, style_losses, content_losses


def neural_transfer_optimization(backbone: nn.Module, norm_mean: torch.Tensor, norm_std: torch.Tensor, 
                                 style_img: torch.Tensor, content_img: torch.Tensor, input_img: torch.Tensor,
                                 content_layers: list, style_layers: list, num_steps: int = 300, 
                                 style_weight: float = 1e6, content_weight: float = 1) -> torch.Tensor:
    """
    Optimizes the input image to achieve style transfer.

    Args:
        backbone: The pre-trained model backbone.
        norm_mean: Normalization mean.
        norm_std: Normalization standard deviation.
        style_img: The style image tensor.
        content_img: The content image tensor.
        input_img: The input image tensor.
        content_layers: List of content layer names, to be used for calculating content loss.
        style_layers: List of style layer names, to be used for calculating style loss.
        num_steps: Number of optimization steps.
        style_weight: Weight for style loss.
        content_weight: Weight for content loss.

    Returns:
        torch.Tensor: The optimized image tensor.
    """
    model, style_losses, content_losses = losses_and_model(backbone, norm_mean, norm_std, 
                                                           style_img, content_img, content_layers, 
                                                           style_layers)
    model.eval()
    model.requires_grad_(False)

    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("Step {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print("---")

            return style_score + content_score
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def neural_transfer_rendering(content_path: str, style_path: str, content_layers: list,
                    style_layers: list, input_img: torch.Tensor = None, output_path: str = None, 
                    num_steps: int = 300, style_weight: float = 1e6, content_weight: float = 1) -> None:
    """
    Performs neural style transfer and optionally saves the result.

    Args:
        content_path: Path to the content image.
        style_path: Path to the style image.
        content_layers: List of content layer names, to be used for calculating content loss.
        style_layers: List of style layer names, to be used for calculating style loss.
        input_img: The input image tensor (optional).
        output_path: Path to save the output image (optional).
        num_steps: Number of optimization steps.
        style_weight: Weight for style loss.
        content_weight: Weight for content loss.
    """
    vgg_19 = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)

    # As defined per VGG19 architecture
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_img, style_img = neural_utils.image_loaders(content_path, style_path)

    if input_img == None:
        input_img = content_img.clone()

    result = neural_transfer_optimization(vgg_19, norm_mean, norm_std, 
                                          style_img, content_img, input_img, 
                                          content_layers, style_layers, num_steps, 
                                          style_weight, content_weight)
    
    neural_utils.image_out(result, output_path)