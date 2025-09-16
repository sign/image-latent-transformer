# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import platform
from collections.abc import Callable

import cairo
import gi
import pytest

from font_configurator.font_configurator import FontConfigurator
from font_configurator.fontconfig_managers import FontconfigMode
from font_configurator.test_font_intergration.conftest import (
    FONT_CONFIGURATOR_ONLY_MINIMAL_TEST_CASES,
    FontConfiguratorOnlyMinimalTestCase,
)

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # type: ignore # noqa: E402


@pytest.mark.skipif(
    platform.system() not in ("Linux", "Darwin"),
    reason="Fontconfig and Pango/Cairo are primarily used on Linux and macOS",
)
class TestFontConfiguratorIntegration:
    @pytest.mark.parametrize(
        "font_file_name",
        [
            "Honk-Regular-VariableFont_MORF,SHLN.ttf",
            "NotoSansCuneiform-Regular.ttf",
        ],
    )
    def test_minimal_fontconfig_creation(
        self,
        font_file_name: str,
        tmp_path: pathlib.Path,
        font_dir_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Tests that the FontConfigurator can create a minimal fontconfig file."""
        font_dir = font_dir_factory(font_file_name)

        font_configurator = FontConfigurator()

        output_dir = tmp_path.joinpath("test_output_font_configurator_only_minimal")
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.is_dir()

        font_configurator.setup_font(
            mode=FontconfigMode.TEMPLATE_MINIMAL,
            font_dir=font_dir,
            fontconfig_destination_dir=output_dir,
        )

        expected_fontconfig_path = output_dir.joinpath(f"{FontconfigMode.TEMPLATE_MINIMAL}").with_suffix(".conf")
        assert expected_fontconfig_path.exists()
        assert expected_fontconfig_path.is_file()

        assert os.environ["PANGOCAIRO_BACKEND"] == "fontconfig"
        assert os.environ["FONTCONFIG_FILE"] == str(expected_fontconfig_path)

    @pytest.mark.parametrize("test_case", FONT_CONFIGURATOR_ONLY_MINIMAL_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_rendering_with_minimal_fontconfig(
        self,
        test_case: FontConfiguratorOnlyMinimalTestCase,
        tmp_path: pathlib.Path,
        font_dir_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Tests rendering text with a minimal fontconfig setup."""
        font_dir = font_dir_factory(test_case.font_file_name)

        print(f"Using font directory: {font_dir}")

        font_configurator = FontConfigurator()

        output_dir = tmp_path.joinpath(f"test_output_{test_case.font_file_name}_only_minimal")
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.is_dir()

        font_configurator.setup_font(
            mode=FontconfigMode.TEMPLATE_MINIMAL,
            font_dir=font_dir,
            fontconfig_destination_dir=output_dir,
        )

        expected_fontconfig_path = output_dir.joinpath(f"{FontconfigMode.TEMPLATE_MINIMAL}").with_suffix(".conf")
        assert expected_fontconfig_path.exists()
        assert expected_fontconfig_path.is_file()

        assert os.environ["PANGOCAIRO_BACKEND"] == "fontconfig"
        assert os.environ["FONTCONFIG_FILE"] == str(expected_fontconfig_path)

        cairo_surface = cairo.ImageSurface(cairo.Format.RGB24, 1024, 768)
        cairo_context = cairo.Context(cairo_surface)

        # add white background
        cairo_context.set_source_rgb(1, 1, 1)
        cairo_context.paint()

        pango_layout = PangoCairo.create_layout(cairo_context)
        pango_layout.set_text(test_case.text_to_render, -1)

        # font_map = PangoCairo.FontMap.new()
        font_map = PangoCairo.FontMap.get_default()
        families = font_map.list_families()
        font_family_names = [family.get_name() for family in families]

        assert test_case.font_family in font_family_names, (
            f"Font family '{test_case.font_family}' not found in available families: {font_family_names}"
        )

        font_description = Pango.FontDescription()
        font_description.set_family(test_case.font_family)
        font_description.set_size(18 * Pango.SCALE)
        pango_layout.set_font_description(font_description)

        # cairo_context.set_source_rgb(0, 0, 0)  # black text
        PangoCairo.show_layout(cairo_context, pango_layout)

        file_output_path = output_dir.joinpath(f"{test_case.font_file_name}_rendered.png")
        cairo_surface.write_to_png(file_output_path)
        assert file_output_path.exists()
        assert file_output_path.is_file()

        assert font_description.to_string() == f"{test_case.font_family} 18"

        print(file_output_path)

        unknown_glyphs_count = pango_layout.get_unknown_glyphs_count()

        assert test_case.expected_op(unknown_glyphs_count, test_case.unknown_glyphs_count), (
            f"Expected unknown glyphs count to be {test_case.expected_op.__name__} "
            f"{test_case.unknown_glyphs_count}, but got {unknown_glyphs_count}"
        )
