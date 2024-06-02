import os
import numpy as np
import random as rnd
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

SIGMA_G = 2.5
RHO = 1.5

def brush(out: np.ndarray, y: int, x: int, color: np.ndarray, texture: np.ndarray) -> None:
    """
    Splatters a texture brush on a specific location of an image. Modifies
    the image in place.

    Args:
        out: The output image where the brush will be applied.
        y: The y-coordinate where the brush will be applied.
        x: The x-coordinate where the brush will be applied.
        color: The color of the brush.
        texture: A 3-channel, greyscale opacity map of the brush.
    """
    tex_h, tex_w, _ = texture.shape
    out_h, out_w, _ = out.shape
    
    half_tex_h = tex_h // 2
    half_tex_w = tex_w // 2
    
    if (y - half_tex_h < 0 or y + (tex_h - half_tex_h) >= out_h or
        x - half_tex_w < 0 or x + (tex_w - half_tex_w) >= out_w):
        return
    
    out_subregion = out[y - half_tex_h:y + (tex_h - half_tex_h), 
                        x - half_tex_w:x + (tex_w - half_tex_w)]

    color_img = np.ones_like(texture) * color
    
    blended = texture * color_img + (1 - texture) * out_subregion
    
    out[y - half_tex_h:y + (tex_h - half_tex_h),
         x - half_tex_w:x + (tex_w - half_tex_w)] = blended
    

def structure_tensor(image: np.ndarray, sigma: float, rho: float, truncate: float = 4.0) -> np.ndarray:
    """
    Extracts the structure tensor of an image.

    Args:
        image: The input image.
        sigma: The standard deviation for Gaussian kernel for the gradients.
        rho: The standard deviation for Gaussian kernel for the structure tensor components.
        truncate: Truncate filter at this many standard deviations. Defaults to 4.0.

    Returns:
        np.ndarray: The structure tensor components as [Sxx, Syy, Sxy].
    """
    Ix = ndimage.gaussian_filter(image, sigma, order=(1, 0), mode="nearest", truncate=truncate)
    Iy = ndimage.gaussian_filter(image, sigma, order=(0, 1), mode="nearest", truncate=truncate)

    Sxx = ndimage.gaussian_filter(Ix * Ix, rho, mode="nearest", truncate=truncate)
    Syy = ndimage.gaussian_filter(Iy * Iy, rho, mode="nearest", truncate=truncate)
    Sxy = ndimage.gaussian_filter(Ix * Iy, rho, mode="nearest", truncate=truncate)

    return np.array([Sxx, Syy, Sxy])


def compute_angles(im: np.ndarray) -> np.ndarray:
    """
    Returns an image where each pixel has been replaced by the angle
    between its local edge orientation and the vertical line.

    Args:
        im: The input image.

    Returns:
        np.ndarray: The angles of the local edge orientations.
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)

    S = structure_tensor(gray, SIGMA_G, RHO)
    angles = np.zeros_like(gray, dtype=np.float32)

    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            J = np.array([[S[0, y, x], S[2, y, x]],
                          [S[2, y, x], S[1, y, x]]])

            eigenvalues, eigenvectors = np.linalg.eigh(J)
            smallest_eigenvector = eigenvectors[:, np.argmin(eigenvalues)]
            angle = np.arctan2(smallest_eigenvector[1], smallest_eigenvector[0])
            angles[y, x] = angle

    return angles


def rotate_image(mat: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image and expands image to avoid cropping.

    Args:
        mat: The input image.
        angle: The rotation angle in radians.

    Returns:
        np.ndarray: The rotated image.
    """
    angle_degrees = np.degrees(angle)
    #the y-axis points downwards, reversing the angle's direction. Rotated to stay horizontal.
    angle_degrees = (angle_degrees + 90) % 360

    height, width = mat.shape[:2] 
    image_center = (width/2, height/2)
    
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.)
    
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def single_scale_paint(
    im: np.ndarray, 
    out: np.ndarray, 
    importance: np.ndarray, 
    texture: np.ndarray, 
    oriented: bool, 
    size: int = 10, 
    N: int = 1000, 
    noise: float = 0.0
) -> None:
    """
    Splatters N textured brushstrokes onto a dark canvas, following an input image.

    Args:
        im: The input image to be paintified.
        out: The output image.
        importance: A 3-channel grayscale image, where pixel values define importance.
        texture: A 3-channel, greyscale opacity map of the brush.
        oriented: Whether to use orientation in painting.
        size: The size of the brush. Defaults to 10.
        N: The number of brush strokes. Defaults to 1000.
        noise: The amount of noise to apply to the brush color. Defaults to 0.
    """
    tex_h, tex_w, _ = texture.shape
    scale = size / max(tex_h, tex_w)
    new_tex_h = int(tex_h * scale)
    new_tex_w = int(tex_w * scale)
    resized_texture = cv2.resize(texture, (new_tex_w, new_tex_h), interpolation=cv2.INTER_LINEAR)
    if oriented:
        angles = compute_angles(im)

    avg_importance = np.mean(importance)
    N = int(N / avg_importance)

    for _ in range(N):
        y = rnd.randint(0, im.shape[0] - 1)
        x = rnd.randint(0, im.shape[1] - 1)

        if rnd.random() > importance[y, x, 0]:  # Reject sample based on importance
            continue
        
        color = im[y, x]
        
        noise_factor = 1 - noise / 2 + noise * np.random.rand(3)
        modulated_color = np.clip(color * noise_factor, 0, 255)
        if oriented:
            rotated_texture = rotate_image(resized_texture, angles[y, x])
        else:
            rotated_texture = resized_texture
        brush(out, y, x, modulated_color, rotated_texture)


def painterly_rendering(
    image_path: str, 
    brush_path: str, 
    out_dir: str, 
    oriented: bool, 
    size: int = 10, 
    N: int = 1000, 
    noise: float = 0.0
) -> None:
    """
    Renders an input image with a painterly effect using a given brush texture, and saves
    it or displays it.

    Args:
        image_path: The path to the input image.
        brush_path: The path to the brush texture.
        out_dir: The output path to save the result. Set to None for displaying and no saving.
        oriented: Whether to use orientation in painting.
        size: The size of the brush. Defaults to 10.
        N: The number of brush strokes. Defaults to 1000.
        noise: The amount of noise to apply to the brush color. Defaults to 0.0.
    """
    im = cv2.imread(image_path)
    texture = cv2.imread(brush_path)
    constant_importance = np.ones_like(im)

    texture = texture / np.max(texture)
    out = np.ones_like(im) * 0
    if oriented:
        single_scale_paint(im, out, constant_importance, texture, True, size, N // 2, noise)
    else:
        single_scale_paint(im, out, constant_importance, texture, False, size, N // 2, noise)

    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_im, cv2.CV_64F)
    sharpness_map = cv2.convertScaleAbs(laplacian)
    sharpness_map = cv2.merge([sharpness_map, sharpness_map, sharpness_map]) / np.max(sharpness_map)

    if oriented:
        single_scale_paint(im, out, sharpness_map, texture, True, size // 4, N // 2, noise)
    else:
        single_scale_paint(im, out, sharpness_map, texture, False, size // 4, N // 2, noise)

    if out_dir is not None:
        directory = os.path.dirname(out_dir)
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(out_dir, out)
    else:
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()