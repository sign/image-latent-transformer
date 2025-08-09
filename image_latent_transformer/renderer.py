import cairo
import gi
import numpy as np
import torch
from PIL import Image

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # noqa: E402


def render_texts(texts: list[str], line_height: int = 32, dpi: int = 120, font_size: int = 12) -> Image.Image:
    """
    Renders multiple lines of text in black on white background using PangoCairo.

    Args:
        texts (list[str]): The texts to render, one per line
        line_height (int): Height of each line in pixels (default: 32)
        dpi (int): DPI resolution (default: 120)
        font_size (int): Font size (default: 12)

    Returns:
        PIL.Image: Rendered image with text lines
    """
    # Scale font size by DPI
    scaled_font_size = (dpi / 72) * font_size

    # Create temporary surface to measure text
    temp_surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1, 1)
    temp_context = cairo.Context(temp_surface)
    temp_layout = PangoCairo.create_layout(temp_context)

    # Set font
    font_desc = Pango.font_description_from_string(f"sans {scaled_font_size}px")
    temp_layout.set_font_description(font_desc)

    # Measure all texts to find maximum width
    max_text_width = 0
    text_widths = []
    for text in texts:
        temp_layout.set_text(text, -1)
        text_width, text_height = temp_layout.get_pixel_size()
        text_widths.append(text_width)
        max_text_width = max(max_text_width, text_width)

    # Add padding and round up to nearest multiple of 32
    width = max_text_width + 10
    width = ((width + 31) // 32) * 32

    # Calculate total height
    height = len(texts) * line_height

    # Create final surface
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
    context = cairo.Context(surface)

    # Fill white background
    context.set_source_rgb(1.0, 1.0, 1.0)
    context.rectangle(0, 0, width, height)
    context.fill()

    # Set black text color
    context.set_source_rgb(0.0, 0.0, 0.0)

    # Create layout for final rendering
    layout = PangoCairo.create_layout(context)
    layout.set_font_description(font_desc)

    # Render each text line
    for i, text in enumerate(texts):
        layout.set_text(text, -1)
        text_width, text_height = layout.get_pixel_size()

        # Position text (left-aligned horizontally, vertically within its line)
        x = 5  # Small left padding
        y = i * line_height + (line_height - text_height) // 2
        context.move_to(x, y)

        # Render text
        PangoCairo.show_layout(context, layout)

    # Convert to PIL Image
    data = surface.get_data()
    img_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
    img_array = img_array[:, :, :3]  # Remove alpha channel
    img_array = img_array[:, :, ::-1]  # BGR to RGB

    img = Image.fromarray(img_array)
    img.info['dpi'] = (dpi, dpi)

    return img


def deconstruct_images(image_tensor: torch.Tensor, num_words: int, channels_first: bool = False) -> torch.Tensor:
    if not channels_first:
        # Convert from [H, W, C] to [C, H, W]
        image_tensor = image_tensor.permute(2, 0, 1)

    C, H, W = image_tensor.shape  # noqa: N806
    line_height = H // num_words
    return image_tensor \
        .unflatten(1, (num_words, line_height)) \
        .permute(1, 0, 2, 3)  # [n, C, H//n, W]


if __name__ == "__main__":
    # Example: render multiple lines of text
    texts = ["hello🤗🤗🤗🤗🤗🤗", "world", "multiple lines"]
    image = render_texts(texts, line_height=32, dpi=120, font_size=12)

    # Save the example
    image.save("hello_example.png")
    print(f"Rendered {texts} and saved as 'hello_example.png'")
    print(f"Image size: {image.size}")

    image = render_texts(["a", "b", "c", "d"], line_height=32, dpi=120, font_size=12)
    image_tensor = torch.tensor(np.array(image))
    print(deconstruct_images(image_tensor, num_words=4).shape)
