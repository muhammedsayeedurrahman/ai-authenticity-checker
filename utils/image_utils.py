from PIL import Image


def load_rgb_image(path):
    img = Image.open(path)
    return img.convert("RGB")
