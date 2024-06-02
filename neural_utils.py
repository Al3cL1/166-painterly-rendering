import os
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IM_SIZE = 512


def image_loaders(content_path: str, style_path: str) -> tuple:
    """
    Loads and formats content and style images.

    Args:
        content_path: The path to the content image.
        style_path: The path to the style image.

    Returns:
        2-tuple: Transformed content and style images as tensors.
    """
    image_transforms = transforms.Compose([transforms.Resize(IM_SIZE), transforms.ToTensor()]) 

    im_content = Image.open(content_path).convert('RGB')
    im_style = Image.open(style_path).convert('RGB')

    im_style = im_style.resize(im_content.size, Image.BILINEAR)

    image_content = image_transforms(im_content).unsqueeze(0)
    image_style = image_transforms(im_style).unsqueeze(0)
    return image_content.to(device, torch.float), image_style.to(device, torch.float)

def image_out(tensor: torch.Tensor, output_path: str = None) -> None:
    """
    Converts a tensor to an image, and saves it or displays it.

    Args:
        tensor: The input tensor.
        output_path: The path where the image will be saved.
    """
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    np_array = image.numpy()

    if np_array.shape[0] == 3:
        np_array = np.transpose(np_array, (1, 2, 0))

    np_array = (np_array * 255).astype(np.uint8)

    if output_path is not None:
        directory = os.path.dirname(output_path)
        os.makedirs(directory, exist_ok=True)
        plt.imsave(output_path, np_array)
    else:
        plt.imshow(np_array)
        plt.axis('off')
        plt.show()

def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Gram matrix of the input tensor.

    Args:
        input: The input tensor.

    Returns:
        torch.Tensor: The Gram matrix.
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    gram  = torch.mm(features, features.t())
    gram = gram.div(a * b * c * d)
    return gram