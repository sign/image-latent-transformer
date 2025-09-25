import pathlib
from dataclasses import dataclass, field

from font_configurator.fontconfig_managers import FontconfigMode


@dataclass
class FontConfiguratorArguments:
    mode: FontconfigMode = field(
        default=FontconfigMode.TEMPLATE_MINIMAL,
        metadata={"help": "Fontconfig mode to use. See FontconfigMode enum for details."},
    )
    fontconfig_source_path: pathlib.Path | str | None = field(
        default=None,
        metadata={"help": "Path to existing fontconfig file to use as a template."},
    )
    font_dir: pathlib.Path | str | None = field(
        default="fonts_collections/Noto_Sans/core_and_extra",
        metadata={
            "help": (
                "Directory containing .ttf font files to be registered with fontconfig. "
                "See scripts/fonts/README.md for instructions to download Noto Sans fonts."
            )
        },
    )
    fontconfig_destination_dir: pathlib.Path | str | None = field(
        default="train_fontconfig",
        metadata={"help": "Directory to save the generated fontconfig files."},
    )
    force_reinitialize: bool = field(
        default=True,
        metadata={"help": "Whether to force re-initialization of fontconfig files."},
    )
