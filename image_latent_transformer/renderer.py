import re
from functools import cache

import cairo
import gi
import numpy as np
from PIL import Image
from signwriting.formats.swu import is_swu
from signwriting.visualizer.visualize import signwriting_to_image
from tqdm import tqdm
from transformers import AutoImageProcessor

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # noqa: E402


def dim_to_block_size(value: int, block_size: int) -> int:
    return ((value + block_size - 1) // block_size) * block_size


def replace_control_characters(text: str) -> str:
    # Special visual handling for control characters using Control Pictures Unicode block
    # Based on https://unicode.org/charts/nameslist/n_2400.html
    def control_char_to_symbol(match):
        char = match.group(0)
        code = ord(char)
        if code <= 0x1F:  # Control characters 0x00-0x1F map to 0x2400-0x241F
            return chr(0x2400 + code)
        elif code == 0x7F:  # DELETE character maps to 0x2421
            return chr(0x2421)
        return char

    return re.sub(r'[\x00-\x1F\x7F]', control_char_to_symbol, text)


@cache
def cached_font_description(font_name: str, dpi: int, font_size: int) -> Pango.FontDescription:
    # Scale font size by DPI
    scaled_font_size = (dpi / 72) * font_size
    return Pango.font_description_from_string(f"{font_name} {scaled_font_size}px")


def render_text(text: str,
                block_size: int = 16,
                dpi: int = 120,
                font_size: int = 12) -> np.ndarray:
    """
    Renders multiple lines of text in black on white background using PangoCairo.

    Args:
        text (str): The text to render on a single line
        block_size (int): Height of each line in pixels, and width scale (default: 32)
        dpi (int): DPI resolution (default: 120)
        font_size (int): Font size (default: 12)

    Returns:
        PIL.Image: Rendered image with text lines
    """
    if is_swu(text):
        return render_signwriting(text, block_size=block_size)

    text = replace_control_characters(text)

    # Create temporary surface to measure text
    temp_surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1, 1)
    temp_context = cairo.Context(temp_surface)
    layout = PangoCairo.create_layout(temp_context)

    # Set font
    font_desc = cached_font_description("sans", dpi, font_size)
    layout.set_font_description(font_desc)

    # Measure all texts to find maximum width
    layout.set_text(text, -1)
    text_width, text_height = layout.get_pixel_size()

    # Add padding and round up to nearest multiple of 32
    # width = dim_to_block_size(text_width + 10, block_size=block_size)
    # TODO: remove once https://github.com/sign/image-latent-transformer/issues/1 is efficient
    width = font_size * 16  # Assume max 16 characters for now

    line_height = block_size

    # Create final surface
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, line_height)
    context = cairo.Context(surface)

    # Fill white background
    context.set_source_rgb(1.0, 1.0, 1.0)
    context.rectangle(0, 0, width, line_height)
    context.fill()

    # Set black text color
    context.set_source_rgb(0.0, 0.0, 0.0)

    # Position text (left-aligned horizontally, vertically within its line)
    x = 5  # Small left padding
    y = (line_height - text_height) // 2
    context.move_to(x, y)

    # Render text
    PangoCairo.show_layout(context, layout)

    # Extract image data as numpy array
    data = surface.get_data()
    img_array = np.frombuffer(data, dtype=np.uint8).reshape((line_height, width, 4))
    img_array = img_array[:, :, 2::-1]  # Remove alpha channel + convert BGR→RGB

    return img_array.copy()


def render_text_image(text: str, block_size: int = 16, dpi: int = 120, font_size: int = 12) -> Image.Image:
    img_array = render_text(text, block_size=block_size, dpi=dpi, font_size=font_size)
    img = Image.fromarray(img_array)
    img.info['dpi'] = (dpi, dpi)
    return img


def render_signwriting(text: str, block_size: int = 16) -> np.ndarray:
    image = signwriting_to_image(text, trust_box=False)
    width = dim_to_block_size(image.width + 10, block_size=block_size)
    height = dim_to_block_size(image.height + 10, block_size=block_size)
    new_image = Image.new("RGB", (width, height), color=(255, 255, 255))
    padding = (width - image.width) // 2, (height - image.height) // 2
    new_image.paste(image, padding, image)
    return np.array(new_image)


def render_text_torch(text: str, image_processor: AutoImageProcessor, **kwargs):
    image = render_text(text, **kwargs)
    image = image_processor(image, do_center_crop=False, do_resize=False, return_tensors="pt")
    return image.pixel_values[0]


def main():
    # Example: render mixed text with emojis and newlines
    text = "hello🤗\r\n\x02 "
    image = render_text_image(text, block_size=16, dpi=120, font_size=12)

    # Save the example
    image.save("hello_example.png")
    print(f"Rendered {text} and saved as 'hello_example.png'")
    print(f"Image size: {image.size}")

    # Example: render SignWriting
    text = "𝠀񀀒񀀚񋚥񋛩𝠃𝤟𝤩񋛩𝣵𝤐񀀒𝤇𝣤񋚥𝤐𝤆񀀚𝣮𝣭"
    image = render_text_image(text, block_size=32)

    # Save the example
    image.save("swu_example.png")
    print(f"Rendered {text} and saved as 'swu_example.png'")
    print(f"Image size: {image.size}")


if __name__ == "__main__":
    main()
