# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from lxml import etree

from font_configurator.fontconfig_managers import (
    BaseFontconfigManager,
    DarwinFontconfigManager,
    FontconfigMode,
    LinuxFontconfigManager,
)


class DummyFontconfigManager(BaseFontconfigManager):
    """Minimal implementation for testing abstract methods."""

    def add_font_directory(self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path) -> pathlib.Path:
        result_path = fontconfig_path.parent.joinpath(f"{FontconfigMode.SYSTEM_EXTENDED}.conf")
        _ = font_dir  # unused in dummy
        return result_path

    def add_font_directory_and_remove_system(
        self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path
    ) -> pathlib.Path:
        result_path = fontconfig_path.parent.joinpath(f"{FontconfigMode.SYSTEM_ISOLATED}.conf")
        _ = font_dir  # unused in dummy
        return result_path

    def create_fontconfig_from_template(
        self, font_dir: pathlib.Path, fontconfig_output_dir: pathlib.Path, mode: FontconfigMode
    ) -> pathlib.Path:
        result_path = fontconfig_output_dir.joinpath(f"{mode}.conf")
        _ = font_dir  # unused in dummy
        return result_path


@pytest.fixture
def dummy_manager() -> DummyFontconfigManager:
    """Create a dummy implementation for testing."""
    return DummyFontconfigManager()


@pytest.fixture
def darwin_manager() -> DarwinFontconfigManager:
    """Create a DarwinFontconfigManager instance for testing."""
    return DarwinFontconfigManager()


@pytest.fixture
def linux_manager() -> LinuxFontconfigManager:
    """Create a LinuxFontconfigManager instance for testing."""
    return LinuxFontconfigManager()


@pytest.fixture
def font_dir() -> pathlib.Path:
    """Path to the test fonts directory."""
    current_dir = pathlib.Path(__file__).parent
    return (current_dir.parent / "test_assets" / "fonts").resolve()


@pytest.fixture(scope="session")
def fontconfig_data_path() -> pathlib.Path:
    """Returns the path to the fontconfig test data directory."""
    # assumes tests are run from the project root.
    current_dir = pathlib.Path(__file__).parent
    return (current_dir.parent / "test_assets" / "fontconfigs").resolve()


@pytest.fixture
def fontconfig_file_factory(
    tmp_path: pathlib.Path, fontconfig_data_path: pathlib.Path
) -> Callable[[str], pathlib.Path]:
    """
    Factory to create a temporary fontconfig file from the test_assets.
    It copies a named .conf file from the assets dir to a temp dir.

    Usage in a test:
        path = fontconfig_file_factory("empty")
        path = fontconfig_file_factory("invalid_xml_v1")
    """

    def _create_fontconfig_file(name: str) -> pathlib.Path:
        source_file = fontconfig_data_path.joinpath(name).with_suffix(".conf").resolve()
        if not source_file.exists():
            msg = f"Test data file not found: {source_file}"
            raise FileNotFoundError(msg)

        content = source_file.read_text(encoding="utf-8")
        dest_file = tmp_path.joinpath(name).with_suffix(".conf").resolve()
        dest_file.write_text(content, encoding="utf-8")
        return dest_file

    return _create_fontconfig_file


@pytest.fixture
def fontconfig_path_non_existent_v1(tmp_path: pathlib.Path) -> pathlib.Path:
    """A path to a directory that does not have a file in it."""
    return tmp_path.joinpath("non_existent_dir_v1").resolve()


@pytest.fixture
def fontconfig_path_non_existent_v2(tmp_path: pathlib.Path) -> pathlib.Path:
    """A path to a file that does not exist within a non-existent directory."""
    return tmp_path.joinpath("non_existent_dir_v2").joinpath("fontconfig_non_existent").with_suffix(".conf").resolve()


@dataclass(frozen=True, slots=True)
class ValidFontConfigTestCase:
    test_name: str  # this name maps to the file in test/test_assets/fontconfigs/
    expected_description: str
    expected_dir_count: int


VALID_FONTCONFIG_TEST_CASES = [
    ValidFontConfigTestCase(
        test_name="valid_dummy__no_font_dir",
        expected_description="valid_dummy__no_font_dir",
        expected_dir_count=0,
    ),
    ValidFontConfigTestCase(
        test_name="valid_dummy__single_font_dir",
        expected_description="valid_dummy__single_font_dir",
        expected_dir_count=1,
    ),
    ValidFontConfigTestCase(
        test_name="valid_dummy__multiple_font_dirs",
        expected_description="valid_dummy__multiple_font_dirs",
        expected_dir_count=3,
    ),
    ValidFontConfigTestCase(
        test_name="valid_dummy__no_description",
        expected_description="",
        expected_dir_count=3,
    ),
    ValidFontConfigTestCase(
        test_name="valid_dummy__no_font_dir_no_description",
        expected_description="",
        expected_dir_count=0,
    ),
    ValidFontConfigTestCase(
        test_name="valid_dummy__v1",
        expected_description="",
        expected_dir_count=3,
    ),
    ValidFontConfigTestCase(
        test_name="valid_darwin__v1",
        expected_description="valid_darwin__v1",
        expected_dir_count=11,
    ),
    ValidFontConfigTestCase(
        test_name="valid_darwin__v2",
        expected_description="valid_darwin__v2",
        expected_dir_count=12,
    ),
    ValidFontConfigTestCase(
        test_name="valid_linux__v1",
        expected_description="valid_linux__v1",
        expected_dir_count=5,
    ),
    ValidFontConfigTestCase(
        test_name="valid_linux__v2",
        expected_description="valid_linux__v2",
        expected_dir_count=4,
    ),
]


@dataclass(frozen=True, slots=True)
class InvalidXMLFontConfigTestCase:
    test_name: str
    expected_exception: type[Exception]


INVALID_XML_FONTCONFIG_TEST_CASES = [
    InvalidXMLFontConfigTestCase(test_name="invalid_dummy__invalid_xml_v1", expected_exception=etree.XMLSyntaxError),
    InvalidXMLFontConfigTestCase(test_name="invalid_dummy__invalid_xml_v2", expected_exception=etree.XMLSyntaxError),
]


@dataclass(frozen=True, slots=True)
class NonExistentFontConfigTestCase:
    # this one still needs the fixture name because the paths are generated dynamically
    fontconfig_path_fixture: str
    expected_exception: type[Exception]


NON_EXISTENT_FONTCONFIG_TEST_CASES = [
    NonExistentFontConfigTestCase(
        fontconfig_path_fixture="fontconfig_path_non_existent_v1", expected_exception=OSError
    ),
    NonExistentFontConfigTestCase(
        fontconfig_path_fixture="fontconfig_path_non_existent_v2", expected_exception=OSError
    ),
]


@dataclass(frozen=True, slots=True)
class InvalidFontConfigTestCase:
    test_name: str
    expected_description: str
    expected_dir_count: int


INVALID_FONTCONFIG_TEST_CASES = [
    InvalidFontConfigTestCase(
        test_name="invalid_dummy__invalid_root_v1",
        expected_description="invalid_dummy__invalid_root_v1",
        expected_dir_count=3,
    ),
]

INVALID_EXTENSIONS = [
    pytest.param(".txt", id="txt"),
    pytest.param(".xml", id="xml"),
    pytest.param(".json", id="json"),
]

DARWIN_ONLY_VALID_TEST_CASES = [tc for tc in VALID_FONTCONFIG_TEST_CASES if tc.test_name.startswith("valid_darwin__")]

LINUX_ONLY_VALID_TEST_CASES = [tc for tc in VALID_FONTCONFIG_TEST_CASES if tc.test_name.startswith("valid_linux__")]

SUPPORTED_TEMPLATE_MODES = [
    FontconfigMode.TEMPLATE_MINIMAL,
]
