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


def generate_random_images(num_images=10, size=(100, 100)):
    """Generate a number of random images."""
    for i in range(num_images):
        img = create_random_image(size)
        img.save(f'random_image_{i}.png')

def apply_contrast_stretching(img):
    """Apply contrast stretching to an image."""
    np_img = np.array(img)
    min_val = np_img.min()
    max_val = np_img.max()
    stretched_img = ((np_img - min_val) / (max_val - min_val) * 255).astype('uint8')
    return Image.fromarray(stretched_img, 'RGB')

def apply_thresholding(img, threshold=128):
    """Apply a simple thresholding to an image."""
    np_img = np.array(img.convert('L'))
    binary_img = (np_img > threshold) * 255
    return Image.fromarray(binary_img.astype('uint8'), 'L')

def blur_image(img, radius=5):
    """Apply a Gaussian blur to an image."""
    return img.filter(ImageFilter.GaussianBlur(radius))

def rotate_image(img, angle=45):
    """Rotate an image by a given angle."""
    return img.rotate(angle, expand=True)

def resize_image(img, size=(200, 200)):
    """Resize an image to the given size."""
    return img.resize(size)

def crop_image(img, box=(50, 50, 150, 150)):
    """Crop an image using the given box."""
    return img.crop(box)

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
    
    # New functionalities
    random_image = create_random_image(size=(200, 200))
    random_image.save('random_image.png')

    stats = image_statistics(img)
    plot_image_statistics(stats, 'image_statistics.png')


    noisy_image = add_noise_to_image(img)
    noisy_image.save(f'{output_base_filename}_NOISY.png')

    sepia_image = apply_sepia_filter(img)
    sepia_image.save(f'{output_base_filename}_SEPIA.png')

    grayscale_image = apply_grayscale_filter(img)
    grayscale_image.save(f'{output_base_filename}_GRAYSCALE.png')

    pattern_image = create_pattern_image(size=(200, 200))
    pattern_image.save('pattern_image.png')

    save_histogram(img, 'histogram.png')

    generate_random_images(num_images=5, size=(100, 100))

    contrast_stretched_image = apply_contrast_stretching(img)
    contrast_stretched_image.save(f'{output_base_filename}_CONTRAST_STRETCHED.png')

    thresholded_image = apply_thresholding(img)
    thresholded_image.save(f'{output_base_filename}_THRESHOLDED.png')

    blurred_image = blur_image(img)
    blurred_image.save(f'{output_base_filename}_BLURRED.png')

    rotated_image = rotate_image(img)
    rotated_image.save(f'{output_base_filename}_ROTATED.png')

    resized_image = resize_image(img)
    resized_image.save(f'{output_base_filename}_RESIZED.png')

    cropped_image = crop_image(img)
    cropped_image.save(f'{output_base_filename}_CROPPED.png')

    save_images(filtered_images, output_base_filename)
    inverted_image.save(f'{output_base_filename}_INVERTED.png')
    enhanced_image.save(f'{output_base_filename}_ENHANCED.png')

    print(f"Images saved with base filename {output_base_filename}")

    from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageChops

def apply_effect(img, effect_name):
    """Apply the chosen effect to the image."""
    if effect_name == 'BLUR':
        return img.filter(ImageFilter.BLUR)
    elif effect_name == 'CONTOUR':
        return img.filter(ImageFilter.CONTOUR)
    elif effect_name == 'DETAIL':
        return img.filter(ImageFilter.DETAIL)
    elif effect_name == 'EDGE_ENHANCE':
        return img.filter(ImageFilter.EDGE_ENHANCE)
    elif effect_name == 'EMBOSS':
        return img.filter(ImageFilter.EMBOSS)
    elif effect_name == 'SHARPEN':
        return img.filter(ImageFilter.SHARPEN)
    elif effect_name == 'SMOOTH':
        return img.filter(ImageFilter.SMOOTH)
    elif effect_name == 'SMOOTH_MORE':
        return img.filter(ImageFilter.SMOOTH_MORE)
