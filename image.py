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

def image_statistics(img):
    """Calculate basic statistics for an image."""
    np_img = np.array(img)
    mean = np_img.mean(axis=(0, 1))
    stddev = np_img.std(axis=(0, 1))
    min_val = np_img.min(axis=(0, 1))
    max_val = np_img.max(axis=(0, 1))
    return {
        'mean': mean,
        'stddev': stddev,
        'min': min_val,
        'max': max_val
    }

def plot_image_statistics(stats, output_filename='image_statistics.png'):
    """Plot the statistics of an image."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    categories = ['Mean', 'Stddev', 'Min', 'Max']
    for idx, (key, value) in enumerate(stats.items()):
        ax.plot(value, label=f'{categories[idx]} R', color='red', linestyle='--')
        ax.plot(value, label=f'{categories[idx]} G', color='green', linestyle='--')
        ax.plot(value, label=f'{categories[idx]} B', color='blue', linestyle='--')
    ax.set_title('Image Color Statistics')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Value')
    ax.legend()
    plt.savefig(output_filename)
    plt.close()

def add_noise_to_image(img, noise_level=25):
    """Add random noise to an image."""
    np_img = np.array(img)
    noise = np.random.normal(0, noise_level, np_img.shape).astype('uint8')
    noisy_img = Image.fromarray(np.clip(np_img + noise, 0, 255).astype('uint8'), 'RGB')
    return noisy_img

def apply_sepia_filter(img):
    """Apply a sepia filter to an image."""
    np_img = np.array(img)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = np.dot(np_img[...,:3], sepia_filter.T)
    sepia_img = np.clip(sepia_img, 0, 255).astype('uint8')
    return Image.fromarray(sepia_img, 'RGB')

def apply_grayscale_filter(img):
    """Convert an image to grayscale."""
    return img.convert('L')

def create_pattern_image(size=(100, 100), pattern_type='checkerboard'):
    """Create a pattern image."""
    np_img = np.zeros((size[0], size[1], 3), dtype='uint8')
    if pattern_type == 'checkerboard':
        block_size = 10
        for i in range(0, size[0], block_size):
            for j in range(0, size[1], block_size):
                if (i // block_size) % 2 == (j // block_size) % 2:
                    np_img[i:i+block_size, j:j+block_size] = [255, 255, 255]
    return Image.fromarray(np_img, 'RGB')

def save_histogram(img, filename='histogram.png'):
    """Save histogram of an image."""
    np_img = np.array(img)
    r_hist, g_hist, b_hist = np_img[..., 0].flatten(), np_img[..., 1].flatten(), np_img[..., 2].flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(r_hist, bins=256, color='red', alpha=0.6, label='Red')
    plt.hist(g_hist, bins=256, color='green', alpha=0.6, label='Green')
    plt.hist(b_hist, bins=256, color='blue', alpha=0.6, label='Blue')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(filename)
    plt.close()
