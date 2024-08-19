from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import random
import matplotlib.pyplot as plt

def apply_filters(img):
    filters = [
        ('BLUR', ImageFilter.BLUR),
        ('CONTOUR', ImageFilter.CONTOUR),
        ('DETAIL', ImageFilter.DETAIL),
        ('EDGE_ENHANCE', ImageFilter.EDGE_ENHANCE),
        ('EMBOSS', ImageFilter.EMBOSS),
        ('SHARPEN', ImageFilter.SHARPEN),
        ('SMOOTH', ImageFilter.SMOOTH),
        ('SMOOTH_MORE', ImageFilter.SMOOTH_MORE),
    ]

    filtered_images = {}
    
    for name, filter in filters:
        filtered_img = img.filter(filter)
        filtered_images[name] = filtered_img

    return filtered_images

def enhance_image(img):
    enhancer = ImageEnhance.Color(img)
    img_color = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Contrast(img_color)
    img_contrast = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Brightness(img_contrast)
    img_brightness = enhancer.enhance(1.2)

    return img_brightness

def invert_image(img):
    if img.mode == 'RGBA':
        img_rgb = img.convert('RGB')
        inverted_img_rgb = ImageOps.invert(img_rgb)
        inverted_img = inverted_img_rgb.convert('RGBA')
    else:
        inverted_img = ImageOps.invert(img)
    
    return inverted_img

def save_images(images, base_filename):
    for name, img in images.items():
        img.save(f'{base_filename}_{name}.png')

def main():
    input_image_path = 'image.jpg'
    output_base_filename = 'output_image'

    try:
        img = Image.open(input_image_path)
    except FileNotFoundError:
        print(f"File {input_image_path} not found.")
        return

    filtered_images = apply_filters(img)
    enhanced_image = enhance_image(img)
    inverted_image = invert_image(img)

    save_images(filtered_images, output_base_filename)
    inverted_image.save(f'{output_base_filename}_INVERTED.png')
    enhanced_image.save(f'{output_base_filename}_ENHANCED.png')

    print(f"Images saved with base filename {output_base_filename}")

def create_random_image(size=(100, 100)):
    """Create a random RGB image."""
    data = np.random.rand(size[0], size[1], 3) * 255
    image = Image.fromarray(data.astype('uint8'), 'RGB')
    return image
