# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

import operator
import pathlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass

import pytest


@pytest.fixture
def font_dir_factory(tmp_path: pathlib.Path) -> Callable[[str], pathlib.Path]:
    """
    A pytest fixture factory to create a temporary directory containing a single specified font.

    This factory reduces boilerplate code for tests that need isolated font environments.

    Args:
        tmp_path: The pytest-provided temporary path fixture.

    Returns:
        A function that takes a font file name (e.g., "Honk-Regular.ttf")
        and returns the path to the temporary directory containing that font.
    """

    def _create_font_dir(font_file_name: str) -> pathlib.Path:
        """The actual factory function returned by the fixture."""

        current_dir = pathlib.Path(__file__).parent
        base_assets_path = (current_dir.parent / "test_assets" / "fonts").resolve()
        src_font_path = base_assets_path.joinpath(font_file_name)

        if not src_font_path.exists():
            pytest.fail(f"Test asset font file not found: {src_font_path}")

        dest_dir = tmp_path.joinpath(src_font_path.stem)
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_font_path, dest_dir)

        return dest_dir

    return _create_font_dir


@dataclass(frozen=True, slots=True)
class FontConfiguratorOnlyMinimalTestCase:
    font_file_name: str
    font_family: str
    text_to_render: str
    expected_op: Callable[[int, int], bool]
    unknown_glyphs_count: int
    test_name: str


FONT_CONFIGURATOR_ONLY_MINIMAL_TEST_CASES = [
    FontConfiguratorOnlyMinimalTestCase(
        font_file_name="Honk-Regular-VariableFont_MORF,SHLN.ttf",
        font_family="Honk",
        text_to_render="Hello, World!",
        expected_op=operator.eq,
        unknown_glyphs_count=0,
        test_name="Honk with basic Latin",
    ),
    FontConfiguratorOnlyMinimalTestCase(
        font_file_name="NotoSansCuneiform-Regular.ttf",
        font_family="Noto Sans Cuneiform",
        text_to_render="""𒀭𒆗𒀳𒀭𒁇𒀀𒈾𒀭𒀜𒋾𒀀𒇉𒈦𒄘𒃼𒀭𒅎𒀭𒀀𒇉𒀀𒌑𒋛𒀪""",
        expected_op=operator.eq,
        unknown_glyphs_count=0,
        test_name="Noto Sans Cuneiform with cuneiform script",
    ),
    FontConfiguratorOnlyMinimalTestCase(
        font_file_name="NotoSansCuneiform-Regular.ttf",
        font_family="Noto Sans Cuneiform",
        text_to_render="Hello, World!",
        expected_op=operator.eq,
        unknown_glyphs_count=0,
        test_name="Noto Sans Cuneiform with basic Latin",
    ),
    FontConfiguratorOnlyMinimalTestCase(
        font_file_name="Honk-Regular-VariableFont_MORF,SHLN.ttf",
        font_family="Honk",
        text_to_render="""𒀭𒆗𒀳𒀭𒁇𒀀𒈾𒀭""",
        expected_op=operator.eq,
        unknown_glyphs_count=8,
        test_name="Honk with cuneiform script",
    ),
]
