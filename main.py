import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageUtils:
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load an image from file as a numpy array."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return np.array(Image.open(filepath))

    @staticmethod
    def save_image(image_array: np.ndarray, filepath: str, normalize: bool = False) -> None:
        """Save a numpy array as an image file."""
        if image_array.dtype != np.uint8:
            if normalize:
                image_array = ImageUtils.normalize_image(image_array).astype(np.uint8)
            else:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(filepath)

    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """Normalize image to range [0, 255]."""
        min_val = np.min(image_array)
        max_val = np.max(image_array)

        if max_val > min_val:
            normalized = (image_array - min_val) / (max_val - min_val) * 255
            return normalized
        else:
            return np.zeros_like(image_array)

    @staticmethod
    def paper_normalization(
            image_array: np.ndarray,
            desired_mean: float = 128.0,  # M
            desired_var: float = 5000.0  # Var
    ) -> np.ndarray:
        """
        Implements Equation (17) from the paper with your variable naming:
          - M (desired_mean) is the target mean,
          - Var (desired_var) is the target variance,
          - M0 is the image's current mean,
          - Var0 is the image's current variance.

        f(x,y) = M0 + sqrt( Var0 * ( (I(x,y) - M)^2 / Var ) ) if I(x,y) > M0
               = M0 - sqrt( Var0 * ( (I(x,y) - M)^2 / Var ) ) otherwise
        """
        # Convert input to float32 for consistent math
        img = image_array.astype(np.float32)

        # Image mean and variance
        M0 = np.mean(img)  # current (image) mean
        Var0 = np.var(img)  # current (image) variance

        # Prepare output array
        out = np.zeros_like(img, dtype=np.float32)

        # Apply piecewise normalization
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > desired_mean:
                    out[i, j] = M0 + np.sqrt(
                        Var0 * ((img[i, j] - desired_mean) ** 2 / (desired_var + 1e-12))
                    )
                else:
                    out[i, j] = M0 - np.sqrt(
                        Var0 * ((img[i, j] - desired_mean) ** 2 / (desired_var + 1e-12))
                    )
        return out

class ConvolutionProcessor:
    @staticmethod
    def calculate_padding_size(kernel: np.ndarray) -> Tuple[int, int]:
        """Calculate required padding size for a convolution kernel."""
        if kernel.ndim != 2:
            raise ValueError("Convolution kernel must be 2D")

        pad_width = (kernel.shape[1] - 1) // 2
        pad_height = (kernel.shape[0] - 1) // 2

        return pad_width, pad_height

    @staticmethod
    def apply_padding(image: np.ndarray, pad_width: int, pad_height: int) -> np.ndarray:
        """Apply zero padding to an image."""
        if image.ndim != 2:
            raise ValueError("Image array must be 2D")

        padded_image = np.zeros(
            (image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width),
            dtype=image.dtype
        )
        padded_image[pad_height:-pad_height, pad_width:-pad_width] = image

        return padded_image

    @staticmethod
    def apply_convolution(
            image: np.ndarray,
            kernel: np.ndarray,
            pad_width: Optional[int] = None,
            pad_height: Optional[int] = None
    ) -> np.ndarray:
        """Apply convolution operation to an image using the given kernel."""
        if image.ndim != 2:
            raise ValueError("Image array must be 2D")

        if kernel.ndim != 2:
            raise ValueError("Convolution kernel must be 2D")

        if pad_width is None or pad_height is None:
            pad_width, pad_height = ConvolutionProcessor.calculate_padding_size(kernel)

        new_height = max(1, image.shape[0] - 2 * pad_height)
        new_width = max(1, image.shape[1] - 2 * pad_width)

        result = np.zeros((new_height, new_width), dtype=np.float32)
        k_height, k_width = kernel.shape

        for i in range(new_height):
            for j in range(new_width):
                roi = image[i:i + k_height, j:j + k_width]
                result[i, j] = np.sum(roi * kernel)

        return result

class ImageFilter:
    """Class for different image filtering operations."""

    @staticmethod
    def create_gabor_kernel(size, theta, f, sigma_x, sigma_y):

        x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
        y = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
        x_grid, y_grid = np.meshgrid(x, y)

        # Rotation
        x_theta = x_grid * np.cos(theta) + y_grid * np.sin(theta)
        y_theta = -x_grid * np.sin(theta) + y_grid * np.cos(theta)

        # Gabor filter equation
        gabor = np.exp(
            -(x_theta ** 2 / (2 * sigma_x ** 2) +
            y_theta ** 2 / (2 * sigma_y ** 2))
        ) * np.cos(2 * np.pi * f * x_theta)

        # Normalize the kernel
        gabor = (gabor - np.mean(gabor)) / np.std(gabor)

        return gabor


class Visualizer:
    """Class for visualizing image processing results."""

    @staticmethod
    def visualize_gabor_filters(
            original_image: np.ndarray,
            gabor_results: np.ndarray,
            theta_vector: np.ndarray,
            filepath_to_save: str = None
    ) -> None:
        """ Visualize Gabor filtering results. """
        # Calculate optimal grid layout
        num_filters = len(gabor_results)
        grid_cols = min(3, num_filters + 1)
        grid_rows = int(np.ceil((num_filters + 1) / grid_cols))

        plt.figure(figsize=(20, 5 * grid_rows))

        # Original image
        plt.subplot(grid_rows, grid_cols, 1)
        plt.title('Normalized Original Image', fontsize=12)
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')

        # Filtered images
        for i, (filtered_img, theta) in enumerate(zip(
                gabor_results,
                theta_vector), start=2):
            plt.subplot(grid_rows, grid_cols, i)
            plt.title(f'Filtered (θ = {float(theta):.1f}°)', fontsize=10)
            plt.imshow(filtered_img, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.suptitle('Gabor Filtering Results', fontsize=16, y=1.02)

        if filepath_to_save is not None:
            plt.savefig(filepath_to_save, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    base_path = r'Images'

    image_array = ImageUtils.load_image(
        filepath=os.path.join(base_path, '104_3.tif')
    )
    print(50 * "-","\nImage Array:\n", image_array)

    normalized_image_array = ImageUtils.paper_normalization(
       image_array,
       desired_mean=128.0,
       desired_var=5000.0
    )

    ImageUtils.save_image(
        image_array=normalized_image_array,
        filepath=os.path.join(base_path, 'normalized_original_image.tif'),
        normalize=True
    )

    theta_vector = np.linspace(0, np.pi, 8, endpoint=False)
    theta_vector_deg = np.degrees(theta_vector)
    print(50 * "-" ,"\nTheta Vector:\n", theta_vector_deg)

    gabor_kernel_vector = [
        ImageFilter.create_gabor_kernel(
            size = 33,
            theta = theta_vector[i],
            f = 0.125,
            sigma_x = 4,
            sigma_y = 4,
        ) for i in range(8)
    ]

    for i in range(8):
        print(50 * "-", "\nGabor Kernel:", f"(Theta={theta_vector_deg[i]:0.1f})\n" ,gabor_kernel_vector[i])

    filtered_images = [
        ConvolutionProcessor.apply_convolution(
            normalized_image_array,
            kernel
        ) for kernel in gabor_kernel_vector
    ]

    Visualizer.visualize_gabor_filters(
        original_image=normalized_image_array,
        gabor_results=filtered_images,
        theta_vector=theta_vector_deg,
        filepath_to_save=os.path.join(base_path,'gabor_filter_visualization.png')
    )

    os.makedirs(os.path.join(base_path, 'Filtered Images'), exist_ok=True)
    for i in range(8):
        ImageUtils.save_image(
            image_array=filtered_images[i],
            filepath=os.path.join(base_path, 'Filtered Images', f"{theta_vector_deg[i]:.1f}.png"),
            normalize=True
        )

if __name__ == '__main__':
    main()

