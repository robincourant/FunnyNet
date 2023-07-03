from PIL import Image


def augment(img, rot, flip_H, flip_V):
    if flip_V < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if flip_H < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img
