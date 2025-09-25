import argparse
import logging
import pathlib
import shutil
from enum import StrEnum, unique

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@unique
class FontSetScenario(StrEnum):
    NOTO_SANS = "Noto_Sans"


class FontCollector:
    """
    Collect font files from a local fonts repo based on patterns and copy them to bundles.
    URL: https://github.com/google/fonts
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def find_fonts_by_pattern(self, source_dir: pathlib.Path, pattern: str = "*.ttf") -> list[pathlib.Path]:
        self.logger.info("Searching for fonts in %s with pattern %s", source_dir, pattern)
        font_files = list(source_dir.rglob(pattern))
        if len(font_files) == 0:
            self.logger.critical("No font files found in %s with pattern %s", source_dir, pattern)
            msg = f"No font files found in {source_dir} with pattern {pattern}"
            raise FileNotFoundError(msg)

        self.logger.info("Found %d font files", len(font_files))
        return font_files

    def filter_fonts_by_parent_dir(self, font_paths: list[pathlib.Path], parent_names: list[str]) -> list[pathlib.Path]:
        filtered_fonts = [font for font in font_paths if font.parent.name.lower() in parent_names]
        if not filtered_fonts:
            msg = f"No .ttf files found for parent directories: {parent_names}"
            self.logger.critical(msg)
            raise FileNotFoundError(msg)
        self.logger.info("Filtered fonts to %d files based on parent names", len(filtered_fonts))
        return filtered_fonts

    def copy_fonts(self, font_paths: list[pathlib.Path], destination_dir: pathlib.Path) -> None:
        destination_dir.mkdir(parents=True, exist_ok=True)
        for font_path in font_paths:
            destination_file_path = destination_dir.joinpath(font_path.name)
            shutil.copy2(font_path, destination_file_path)


def collect_noto_sans_fonts(
    fonts_root: pathlib.Path,
    output_dir: pathlib.Path,
) -> tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
    if not isinstance(fonts_root, pathlib.Path):
        fonts_root = pathlib.Path(fonts_root).resolve(strict=True)

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir).resolve()

    pattern = "NotoSans*.ttf"

    collector = FontCollector()
    all_noto_sans_fonts = collector.find_fonts_by_pattern(source_dir=fonts_root, pattern=pattern)

    # by .parent we get the directory containing the font files
    core_scripts = [
        # "notocoloremoji",  # colored emoji (added separately)
        "notosans",  # Latin, Greek, Cyrillic
        "notosansarabic",  # Arabic
        "notosansdevanagari",  # Devanagari
        "notosanssc",  # Simplified Chinese
        "notosanstc",  # Traditional Chinese
        "notosansjp",  # Japanese
        "notosanskr",  # Korean
        "notosanshebrew",  # Hebrew
        "notosansthai",  # Thai
        "notosansmath",  # math symbols
        "notosanssymbols",  # additional symbols
    ]

    extra_scripts = [
        "notosansethiopic",  # Ethiopic
        "notosansarmenian",  # Armenian
        "notosansgeorgian",  # Georgian
        "notosanskhmer",  # Khmer
        "notosanslao",  # Lao (Thai covered, but just in case)
        "notosansmyanmar",  # Myanmar
        "notosanssinhala",  # Sinhala
        "notosanssymbols2",  # additional symbols
    ]

    core_list = collector.filter_fonts_by_parent_dir(font_paths=all_noto_sans_fonts, parent_names=core_scripts)
    extra_list = collector.filter_fonts_by_parent_dir(font_paths=all_noto_sans_fonts, parent_names=extra_scripts)

    emoji_font_name = "NotoColorEmoji-Regular.ttf"
    emoji_font = collector.find_fonts_by_pattern(source_dir=fonts_root, pattern=emoji_font_name)
    assert len(emoji_font) == 1, f"Expected exactly one {emoji_font_name} font"
    core_list.extend(emoji_font)  # add emoji font to core list

    other_list = [font for font in all_noto_sans_fonts if font not in core_list + extra_list]

    dir_name = "Noto_Sans"

    core_path = output_dir.joinpath(dir_name).joinpath("core")
    collector.copy_fonts(
        font_paths=core_list,
        destination_dir=core_path,
    )

    core_and_extra_path = output_dir.joinpath(dir_name).joinpath("core_and_extra")
    collector.copy_fonts(
        font_paths=core_list + extra_list,
        destination_dir=core_and_extra_path,
    )

    other_path = output_dir.joinpath(dir_name).joinpath("other")
    collector.copy_fonts(
        font_paths=core_list + extra_list + other_list,
        destination_dir=other_path,
    )

    # +1 because of the emoji font added to core_list
    assert len(core_list) + len(extra_list) + len(other_list) == len(all_noto_sans_fonts) + 1, (
        "Some fonts are missing in the final lists"
    )

    # emoji font and NotoSans-Italic[wdth,wght].ttf
    assert len(core_scripts) + 2 == len(core_list), (
        f"Some core scripts are missing, expected {len(core_scripts) + 2}, got {len(core_list)}"
    )

    assert len(extra_scripts) == len(extra_list), (
        f"Some extra scripts are missing, expected {len(extra_scripts)}, got {len(extra_list)}"
    )

    assert len(core_list) == len(list(core_path.rglob("*.ttf"))), "Mismatch in core fonts copied"
    assert len(core_list + extra_list) == len(list(core_and_extra_path.rglob("*.ttf"))), (
        "Mismatch in core_and_extra fonts copied"
    )

    return core_list, extra_list, other_list


def main(
    fonts_root: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    scenario: FontSetScenario = FontSetScenario.NOTO_SANS,
) -> None:
    if not isinstance(fonts_root, pathlib.Path):
        fonts_root = pathlib.Path(fonts_root).resolve(strict=True)

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir).resolve()

    match scenario:
        case FontSetScenario.NOTO_SANS:
            fonts_root = fonts_root.joinpath("ofl").resolve(strict=True)
            _ = collect_noto_sans_fonts(
                fonts_root=fonts_root,
                output_dir=output_dir,
            )
        case _:
            msg = f"Scenario {scenario} is not implemented"
            raise NotImplementedError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect/copy fonts into bundles")
    parser.add_argument(
        "--fonts-root",
        dest="fonts_root",
        help="Path to the local Google Fonts repository root (e.g., ~/repos/fonts)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="fonts_collections",
        help="Destination directory for the bundled fonts",
    )
    parser.add_argument(
        "--scenario",
        choices=[e.value for e in FontSetScenario],
        default=FontSetScenario.NOTO_SANS.value,
        help="Font bundling scenario",
    )
    args = parser.parse_args()

    main(
        fonts_root=args.fonts_root,
        output_dir=args.output_dir,
        scenario=FontSetScenario(args.scenario),
    )
