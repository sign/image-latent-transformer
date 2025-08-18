import unittest

import numpy as np
import torch

from image_latent_transformer.renderer import render_text


class TestRenderer(unittest.TestCase):

    def test_single_text_has_black_pixels(self):
        """Test that rendering a single text produces black pixels in the image."""
        text = "Hello World"
        image = render_text(text, line_height=32, dpi=120, font_size=12)

        # Convert to numpy array
        img_array = np.array(image)

        # Check if there are any black pixels (0, 0, 0) in RGB
        # Since the background is white (255, 255, 255), any text should create non-white pixels
        has_black_pixels = np.any(img_array < 255)

        assert has_black_pixels, "Rendered text should contain black pixels"

    def test_empty_text_no_black_pixels(self):
        """Test that rendering empty text produces no black pixels (all white)."""
        image = render_text("", line_height=32, dpi=120, font_size=12)

        # Convert to numpy array
        img_array = np.array(image)

        # Check that all pixels are white (255, 255, 255)
        all_white = np.all(img_array == 255)

        assert all_white, "Empty text should produce an all-white image"

    def test_multiple_different_texts_are_different(self):
        """Test that different texts produce different deconstructions."""
        texts = ["a", "b"]
        renders = [render_text(text, line_height=32, dpi=120, font_size=12) for text in texts]
        img_array = [np.array(render) for render in renders]
        img_tensor = [torch.tensor(arr) for arr in img_array]

        # Each text should produce a different deconstruction
        line_a = img_tensor[0]  # First line containing "a"
        line_b = img_tensor[1]  # Second line containing "b"

        # Check that the two lines are different
        are_different = not torch.equal(line_a, line_b)

        assert are_different, "Different texts 'a' and 'b' should produce different deconstructions"

    def test_multiple_identical_texts_deconstruction(self):
        """Test that identical texts produce identical deconstructions."""
        texts = ["a", "a", "a", "a"]
        renders = [render_text(text, line_height=32, dpi=120, font_size=12) for text in texts]
        img_array = [np.array(render) for render in renders]
        img_tensor = [torch.tensor(arr) for arr in img_array]

        # All images should be identical since they contain the same text
        all_same = True
        first_line = img_tensor[0]

        for i in range(1, len(texts)):
            if not torch.equal(first_line, img_tensor[i]):
                all_same = False
                break

        assert all_same, "Identical texts 'a' should produce identical deconstructions"

    def test_render_consistency(self):
        """Test that rendering the same text multiple times produces consistent results."""
        text = "consistent test"

        # Render the same text twice
        image1 = render_text(text, line_height=32, dpi=120, font_size=12)
        image2 = render_text(text, line_height=32, dpi=120, font_size=12)

        # Convert to arrays
        array1 = np.array(image1)
        array2 = np.array(image2)

        # Check that both renderings are identical
        are_identical = np.array_equal(array1, array2)

        assert are_identical, "Rendering the same text should produce identical results"

    def test_newline_text_has_black_pixels(self):
        image = render_text("\n", line_height=32, dpi=120, font_size=12)

        # Convert to numpy array
        img_array = np.array(image)

        # Check if there are any black pixels (0, 0, 0) in RGB
        # Since the background is white (255, 255, 255), any text should create non-white pixels
        has_black_pixels = np.any(img_array < 255)

        assert has_black_pixels, "Rendered text should contain black pixels"


if __name__ == '__main__':
    unittest.main()
