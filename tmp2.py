import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

circle = [(-1, -1), (1, 1)]
triangle_u = [(1, 1), (0, -1), (-1, 1)]
triangle_r = [(-1, 1), (1, 0), (-1, -1)]
triangle_l = [(1, 1), (-1, 0), (1, -1)]
triangle_d = [(-1, -1), (0, 1), (1, -1)]
square = [(1, 1), (1, -1), (-1, -1), (-1, 1)]


def draw(shape, color, scale, pos):
    color = 255 * color // 11
    pos = np.array(pos) * 160
    shape = np.array(shape) * 20 * scale + pos
    shape = [(s[0], s[1]) for s in shape]
    painter.polygon(shape, fill=color, outline=0)


def draw_circle(shape, color, scale, pos):
    color = 255 * color // 11
    pos = np.array(pos) * 160
    shape = np.array(shape) * 20 * scale + pos
    shape = [(s[0], s[1]) for s in shape]
    painter.ellipse(shape, fill=color, outline=0)


image = Image.new("L", (160, 160), color=255)
painter = ImageDraw.Draw(image)

draw(square, 2, 0.9, (0.5, 0.5))
draw(triangle_r, 3, 0.9, (0.75, 0.5))
draw(triangle_l, 4, 0.7, (0.25, 0.5))
draw_circle(circle, 5, 0.9, (0.5, 0.25))

# image.show()
plt.imshow(np.array(image))
plt.show()

