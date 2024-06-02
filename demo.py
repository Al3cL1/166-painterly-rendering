import painterly
import neural_transfer
from os.path import join

image_path = 'assets/images'
brush_path = 'assets/brushes'
out_path = 'assets/outputs'

# Parinterly rendering demo with and without orientation
painterly.painterly_rendering(join(image_path, "villeperdue.png"), join(brush_path, "brush.png"), 
          join(out_path, "villeperdue_unOr.png"), False, size=30, N=100000, noise=0.03)
painterly.painterly_rendering(join(image_path, "china.png"), join(brush_path, "brush.png"), 
          join(out_path, "china_unOr.png"), False, size=30, N=100000, noise=0.03)

painterly.painterly_rendering(join(image_path, "china.png"), join(brush_path, "brush.png"), 
          join(out_path, "china_Or.png"), True, size=30, N=100000, noise=0.03)
painterly.painterly_rendering(join(image_path, "round.png"), join(brush_path, "brush.png"), 
          join(out_path, "round_Or.png"), True, size=30, N=100000, noise=0.03)

# Neural Transfer demo
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_path = join(image_path, "DSC_8394.png")

neural_transfer.neural_transfer_rendering(content_path, join(image_path, "Juan-Gris.jpg"), 
                content_layers, style_layers, None, join(out_path, "Juan-Gris-DSC_8394.png"))