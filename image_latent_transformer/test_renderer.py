import unittest

import numpy as np
import torch

from image_latent_transformer.renderer import deconstruct_images, render_texts


class TestRenderer(unittest.TestCase):

    def test_single_text_has_black_pixels(self):
        """Test that rendering a single text produces black pixels in the image."""
        texts = ["Hello World"]
        image = render_texts(texts, line_height=32, dpi=120, font_size=12)

        # Convert to numpy array
        img_array = np.array(image)

        # Check if there are any black pixels (0, 0, 0) in RGB
        # Since the background is white (255, 255, 255), any text should create non-white pixels
        has_black_pixels = np.any(img_array < 255)

        assert has_black_pixels, "Rendered text should contain black pixels"

    def test_empty_text_no_black_pixels(self):
        """Test that rendering empty text produces no black pixels (all white)."""
        texts = [""]
        image = render_texts(texts, line_height=32, dpi=120, font_size=12)

        # Convert to numpy array
        img_array = np.array(image)

        # Check that all pixels are white (255, 255, 255)
        all_white = np.all(img_array == 255)

        assert all_white, "Empty text should produce an all-white image"

    def test_multiple_different_texts_deconstruction(self):
        """Test that different texts produce different deconstructions."""
        texts = ["a", "b"]
        image = render_texts(texts, line_height=32, dpi=120, font_size=12)

        # Convert to tensor format expected by deconstruct_images
        img_array = np.array(image)
        img_tensor = torch.tensor(img_array)  # [H, W, C]

        # Deconstruct the image
        deconstructed = deconstruct_images(img_tensor, num_words=2)

        # Each text should produce a different deconstruction
        line_a = deconstructed[0]  # First line containing "a"
        line_b = deconstructed[1]  # Second line containing "b"

        # Check that the two lines are different
        are_different = not torch.equal(line_a, line_b)

        assert are_different, "Different texts 'a' and 'b' should produce different deconstructions"

    def test_multiple_identical_texts_deconstruction(self):
        """Test that identical texts produce identical deconstructions."""
        texts = ["a", "a", "a", "a"]
        image = render_texts(texts, line_height=32, dpi=120, font_size=12)

        # Convert to tensor format expected by deconstruct_images
        img_array = np.array(image)
        img_tensor = torch.tensor(img_array)  # [H, W, C]

        # Deconstruct the image
        deconstructed = deconstruct_images(img_tensor, num_words=4)

        # All lines should be identical since they contain the same text
        all_same = True
        first_line = deconstructed[0]

        for i in range(1, 4):
            if not torch.equal(first_line, deconstructed[i]):
                all_same = False
                break

        assert all_same, "Identical texts 'a' should produce identical deconstructions"

    def test_deconstruct_shape(self):
        """Test that deconstruct_images produces the correct output shape."""
        texts = ["line1", "line2", "line3"]
        num_words = len(texts)
        line_height = 32

        image = render_texts(texts, line_height=line_height, dpi=120, font_size=12)

        # Convert to tensor format
        img_array = np.array(image)
        img_tensor = torch.tensor(img_array)  # [H, W, C]

        # Deconstruct the image
        deconstructed = deconstruct_images(img_tensor, num_words=num_words)

        # Check output shape
        expected_shape = (num_words, 3, line_height, img_tensor.shape[1])  # 3 channels (RGB)
        assert deconstructed.shape == expected_shape, f"Expected shape {expected_shape}, got {deconstructed.shape}"

    def test_render_consistency(self):
        """Test that rendering the same text multiple times produces consistent results."""
        texts = ["consistent test"]

        # Render the same text twice
        image1 = render_texts(texts, line_height=32, dpi=120, font_size=12)
        image2 = render_texts(texts, line_height=32, dpi=120, font_size=12)

        # Convert to arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Check that both renderings are identical
        are_identical = np.array_equal(array1, array2)

        assert are_identical, "Rendering the same text should produce identical results"


if __name__ == '__main__':
    unittest.main()
