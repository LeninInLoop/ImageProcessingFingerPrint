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

    @staticmethod
    def compute_aad_map(image_array: np.ndarray, block_size: int) -> np.ndarray:
        """
        Splits the image into non-overlapping square blocks of size block_size x block_size.
        For each block, computes the Absolute Average Deviation (AAD) from the mean.
        Returns a 2D 'AAD map' of shape (n_blocks_y, n_blocks_x),
        where each element corresponds to the AAD of one block.
        """
        if image_array.ndim != 2:
            raise ValueError("compute_aad_map expects a 2D grayscale image.")

        height, width = image_array.shape
        n_blocks_y = height // block_size
        n_blocks_x = width // block_size

        # Create an empty 2D array to store the AAD of each block
        aad_map = np.zeros((n_blocks_y, n_blocks_x), dtype=np.float32)

        # Loop over blocks
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                start_y = by * block_size
                end_y = start_y + block_size
                start_x = bx * block_size
                end_x = start_x + block_size

                block = image_array[start_y:end_y, start_x:end_x]
                mean_val = np.mean(block)
                # AAD = average of absolute deviations from mean
                aad_val = np.mean(np.abs(block - mean_val))

                aad_map[by, bx] = aad_val

        return aad_map

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
        """ Visualize Gabor filtering results with 3 rows. """
        plt.figure(figsize=(20, 15))

        # First row - only original image
        plt.subplot(3, 4, (1, 4))
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Fingerprint', fontsize=25, pad=10)
        plt.axis('off')

        # Second row - first 4 filtered images
        for i in range(4):
            plt.subplot(3, 4, 4 + i + 1)
            plt.imshow(gabor_results[i], cmap='gray')
            plt.title(f'Filtered (θ = {float(theta_vector[i]):.1f}°)', fontsize=25, pad=10)
            plt.axis('off')

        # Third row - last 4 filtered images
        for i in range(4):
            plt.subplot(3, 4, 8 + i + 1)
            plt.imshow(gabor_results[i + 4], cmap='gray')
            plt.title(f'Filtered (θ = {float(theta_vector[i + 4]):.1f}°)', fontsize=25, pad=10)
            plt.axis('off')

        plt.tight_layout()

        if filepath_to_save is not None:
            plt.savefig(filepath_to_save, dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def show_aad_maps_grid(
            aad_maps: list,
            titles=None,
            filepath_to_save: str = None
    ) -> None:
        """
        Displays a list of 8 AAD maps in a 2x4 grid on a black background,
        with a shared grayscale colorbar for consistent contrast.
        """
        if len(aad_maps) != 8:
            raise ValueError("Expected exactly 8 AAD maps for a 2×4 grid.")

        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.patch.set_facecolor('white')  # Full black background
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            ax.set_facecolor('black')  # Black frame background
            ax.imshow(aad_maps[i], cmap='gray', aspect='auto')  # Show image in grayscale
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_frame_on(True)  # Ensure there's a black border

            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12, color='black', pad=6)

        plt.subplots_adjust(wspace=0.5, hspace=0.2)  # Adjust spacing

        if filepath_to_save is not None:
            plt.savefig(filepath_to_save, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    base_path = r'Images'

    image_array = ImageUtils.load_image(
        filepath=os.path.join(base_path, 'original_image.tif')
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
            f = 0.100,
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

    block_size = 16
    aad_maps = [ConvolutionProcessor.compute_aad_map(img, block_size=block_size) for img in filtered_images]

    labels = []
    for i in range(8):
        labels.append(f"Theta={theta_vector_deg[i]:0.1f}")
    Visualizer.show_aad_maps_grid(
        aad_maps,
        titles=labels,
        filepath_to_save=os.path.join(base_path,'AAD_MAP_visualization.png')
    )

    os.makedirs(os.path.join(base_path, 'AAD MAP Images'), exist_ok=True)
    for i in range(8):
        ImageUtils.save_image(
            image_array=aad_maps[i],
            filepath=os.path.join(base_path, 'AAD MAP Images', f"{theta_vector_deg[i]:.1f}.png"),
            normalize=True
        )

if __name__ == '__main__':
    main()

