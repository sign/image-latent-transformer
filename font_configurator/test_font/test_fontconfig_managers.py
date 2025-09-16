# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from lxml import etree

from font_configurator.fontconfig_managers import (
    BaseFontconfigManager,
    DarwinFontconfigManager,
    FontconfigMode,
    LinuxFontconfigManager,
    SupportedPlatforms,
)
from font_configurator.fontconfig_templates import (
    DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS,
    LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS,
)
from font_configurator.test_font.conftest import (
    DARWIN_ONLY_VALID_TEST_CASES,
    INVALID_EXTENSIONS,
    INVALID_FONTCONFIG_TEST_CASES,
    INVALID_XML_FONTCONFIG_TEST_CASES,
    LINUX_ONLY_VALID_TEST_CASES,
    NON_EXISTENT_FONTCONFIG_TEST_CASES,
    SUPPORTED_TEMPLATE_MODES,
    VALID_FONTCONFIG_TEST_CASES,
    DummyFontconfigManager,
    InvalidFontConfigTestCase,
    InvalidXMLFontConfigTestCase,
    NonExistentFontConfigTestCase,
    ValidFontConfigTestCase,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


class TestSupportedPlatforms:
    """Test the SupportedPlatforms enum."""

    def test_supported_platforms_enum(self) -> None:
        assert SupportedPlatforms.DARWIN.value == "Darwin"
        assert SupportedPlatforms.LINUX.value == "Linux"
        assert len(SupportedPlatforms) == 2


class TestFontconfigMode:
    """Test the FontconfigMode enum."""

    def test_all_mode_values(self) -> None:
        assert FontconfigMode.FROM_FILE.value == "from_file"
        assert FontconfigMode.SYSTEM_COPY.value == "system_copy"
        assert FontconfigMode.SYSTEM_EXTENDED.value == "system_extended"
        assert FontconfigMode.SYSTEM_ISOLATED.value == "system_isolated"
        assert FontconfigMode.TEMPLATE_MINIMAL.value == "template_minimal"
        assert len(FontconfigMode) == 5


class TestBaseFontconfigManager:
    """Test the abstract base class functionality."""

    def test_fontconfig_extension_class_variable(self, dummy_manager: DummyFontconfigManager) -> None:
        """Test that the fontconfig extension is set correctly."""
        assert dummy_manager.fontconfig_extension == ".conf"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_parse_xml_config_file(
        self,
        dummy_manager: DummyFontconfigManager,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test parsing with valid XML fontconfig content."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = dummy_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

        assert isinstance(tree, etree._ElementTree)
        assert tree.getroot().tag == "fontconfig"
        assert test_case.expected_description in tree.findtext(".//description", default="")
        assert len(tree.findall(".//dir")) == test_case.expected_dir_count

    @pytest.mark.parametrize("test_case", INVALID_XML_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_parse_xml_config_file_invalid_xml(
        self,
        dummy_manager: DummyFontconfigManager,
        test_case: InvalidXMLFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test parsing with invalid XML fontconfig content."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        with pytest.raises(test_case.expected_exception):
            dummy_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

    @pytest.mark.parametrize("test_case", NON_EXISTENT_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.fontconfig_path_fixture)
    def test_parse_xml_config_file_not_exists(
        self,
        dummy_manager: DummyFontconfigManager,
        test_case: NonExistentFontConfigTestCase,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test parsing with invalid file path."""
        fontconfig_path = request.getfixturevalue(test_case.fontconfig_path_fixture)

        with pytest.raises(test_case.expected_exception):
            dummy_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

    @pytest.mark.parametrize("test_case", INVALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_parse_xml_config_file_invalid_fontconfig(
        self,
        dummy_manager: DummyFontconfigManager,
        test_case: InvalidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test parsing with valid XML but invalid fontconfig structure."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = dummy_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

        assert isinstance(tree, etree._ElementTree)
        assert tree.getroot().tag != "fontconfig"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_write_xml_config_file(
        self,
        dummy_manager: DummyFontconfigManager,
        test_case: ValidFontConfigTestCase,
        tmp_path: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test writing XML config file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)
        original_content = fontconfig_path.read_text(encoding="utf-8")

        tree = dummy_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

        output_file = tmp_path.joinpath("output").with_suffix(".conf").resolve()
        dummy_manager._write_xml_config_file(fontconfig_path=output_file, tree=tree)

        assert output_file.exists()
        assert output_file.is_file()
        assert output_file.read_text(encoding="utf-8") == original_content

        output_file_tree = etree.parse(str(output_file))
        assert isinstance(output_file_tree, etree._ElementTree)
        assert output_file_tree.getroot().tag == "fontconfig"
        assert test_case.expected_description in output_file_tree.findtext(".//description", default="")
        assert len(output_file_tree.findall(".//dir")) == test_case.expected_dir_count

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_validate_conf_file_existence(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test validation with existing .conf file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)
        # this should not raise any exception
        dummy_manager._validate_conf_file_existence(file_path=fontconfig_path)

    @pytest.mark.parametrize("test_case", NON_EXISTENT_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.fontconfig_path_fixture)
    def test_validate_conf_file_existence_not_exists(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: NonExistentFontConfigTestCase,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test validation with non-existent file."""
        fontconfig_path = request.getfixturevalue(test_case.fontconfig_path_fixture)

        with pytest.raises(test_case.expected_exception):
            dummy_manager._validate_conf_file_existence(file_path=fontconfig_path)

    @pytest.mark.parametrize("invalid_extension", INVALID_EXTENSIONS)
    def test_validate_conf_file_existence_wrong_extension(
        self,
        dummy_manager: BaseFontconfigManager,
        tmp_path: pathlib.Path,
        invalid_extension: str,
    ) -> None:
        """Test validation with wrong file extension."""
        wrong_ext = tmp_path.joinpath("dummy_file").with_suffix(invalid_extension).resolve()

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager._validate_conf_file_existence(file_path=wrong_ext)

    def test_validate_conf_file_existence_not_a_file(
        self,
        dummy_manager: BaseFontconfigManager,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test validation when passing a path that is not a file."""
        # tmp_path is a directory, so we can test it directly

        assert tmp_path.exists()
        assert tmp_path.is_dir()
        assert not tmp_path.is_file()

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager._validate_conf_file_existence(file_path=tmp_path)

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_load_fontconfig_file_from_path(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test loading valid fontconfig file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)
        original_content = fontconfig_path.read_text(encoding="utf-8")

        result = dummy_manager.load_fontconfig_file_from_path(fontconfig_path=fontconfig_path)
        assert result == fontconfig_path.resolve()

        tree = etree.parse(str(result))
        assert test_case.expected_description in tree.findtext(".//description", default="")
        assert len(tree.findall(".//dir")) == test_case.expected_dir_count
        assert result.read_text(encoding="utf-8") == original_content

    @pytest.mark.parametrize("test_case", NON_EXISTENT_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.fontconfig_path_fixture)
    def test_load_fontconfig_file_from_path_not_exists(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: NonExistentFontConfigTestCase,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test loading non-existent fontconfig file."""
        fontconfig_path = request.getfixturevalue(test_case.fontconfig_path_fixture)

        with pytest.raises(test_case.expected_exception):
            dummy_manager.load_fontconfig_file_from_path(fontconfig_path=fontconfig_path)

    @pytest.mark.parametrize("test_case", INVALID_XML_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_load_fontconfig_file_from_path_invalid_xml(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: InvalidXMLFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test loading invalid XML fontconfig file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        with pytest.raises(test_case.expected_exception):
            dummy_manager.load_fontconfig_file_from_path(fontconfig_path=fontconfig_path)

    @pytest.mark.parametrize("invalid_extension", INVALID_EXTENSIONS)
    def test_load_fontconfig_file_from_path_wrong_extension(
        self, dummy_manager: BaseFontconfigManager, tmp_path: pathlib.Path, invalid_extension: str
    ) -> None:
        """Test loading file with wrong extension."""
        wrong_ext = tmp_path.joinpath("dummy_file").with_suffix(invalid_extension).resolve()
        wrong_ext.write_text("dummy_content")

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager.load_fontconfig_file_from_path(fontconfig_path=wrong_ext)

    def test_load_fontconfig_file_from_path_not_a_file(
        self, dummy_manager: BaseFontconfigManager, tmp_path: pathlib.Path
    ) -> None:
        """Test loading when passing a path that is not a file."""

        assert tmp_path.exists()
        assert tmp_path.is_dir()
        assert not tmp_path.is_file()

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager.load_fontconfig_file_from_path(fontconfig_path=tmp_path)

    @pytest.mark.parametrize("test_case", INVALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_load_fontconfig_file_from_path_invalid_fontconfig(
        self,
        dummy_manager: BaseFontconfigManager,
        test_case: InvalidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test loading fontconfig file (valid xml but invalid fontconfig)."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        with pytest.raises(ValueError, match="Invalid root element in fontconfig file"):
            dummy_manager.load_fontconfig_file_from_path(fontconfig_path=fontconfig_path)

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_copy_fontconfig_file(
        self,
        dummy_manager: BaseFontconfigManager,
        tmp_path: pathlib.Path,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test copying fontconfig file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        dest_dir = tmp_path.joinpath("copy_destination").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        result = dummy_manager.copy_fontconfig_file(
            fontconfig_source_path=fontconfig_path,
            fontconfig_destination_dir=dest_dir,
        )

        expected_path = dest_dir.joinpath(FontconfigMode.SYSTEM_COPY).with_suffix(".conf").resolve()
        assert result == expected_path
        assert result.exists()
        assert result.read_text() == fontconfig_path.read_text()

    @pytest.mark.parametrize("test_case", NON_EXISTENT_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.fontconfig_path_fixture)
    def test_copy_fontconfig_file_not_exists(
        self,
        dummy_manager: BaseFontconfigManager,
        tmp_path: pathlib.Path,
        test_case: NonExistentFontConfigTestCase,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test copying non-existent fontconfig file."""
        fontconfig_path = request.getfixturevalue(test_case.fontconfig_path_fixture)

        dest_dir = tmp_path.joinpath("copy_destination_not_exists").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(test_case.expected_exception):
            dummy_manager.copy_fontconfig_file(
                fontconfig_source_path=fontconfig_path, fontconfig_destination_dir=dest_dir
            )

    @pytest.mark.parametrize("invalid_extension", INVALID_EXTENSIONS)
    def test_copy_fontconfig_file_wrong_extension(
        self, dummy_manager: BaseFontconfigManager, tmp_path: pathlib.Path, invalid_extension: str
    ) -> None:
        """Test copying file with wrong extension."""
        wrong_ext = tmp_path.joinpath("dummy_file").with_suffix(invalid_extension).resolve()
        wrong_ext.write_text("dummy_content")

        dest_dir = tmp_path.joinpath("copy_destination_wrong_extension").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager.copy_fontconfig_file(fontconfig_source_path=wrong_ext, fontconfig_destination_dir=dest_dir)

    def test_copy_fontconfig_file_not_a_file(
        self, dummy_manager: BaseFontconfigManager, tmp_path: pathlib.Path
    ) -> None:
        """Test copying when passing a path that is not a file."""

        dest_dir = tmp_path.joinpath("copy_destination_not_a_file").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        assert tmp_path.exists()
        assert tmp_path.is_dir()
        assert not tmp_path.is_file()

        with pytest.raises(FileNotFoundError, match=r"No .conf file found"):
            dummy_manager.copy_fontconfig_file(fontconfig_source_path=tmp_path, fontconfig_destination_dir=dest_dir)

    def test_dummy_abstract_methods(
        self,
        dummy_manager: DummyFontconfigManager,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
    ) -> None:
        """Test that dummy implementations of abstract methods work."""
        # use tmp_path as the base directory for operations
        fontconfig_path = tmp_path.joinpath("dummy.conf").resolve()
        fontconfig_path.touch()

        extended_path = dummy_manager.add_font_directory(fontconfig_path=fontconfig_path, font_dir=font_dir)
        assert extended_path.name == f"{FontconfigMode.SYSTEM_EXTENDED}.conf"
        assert extended_path.parent == tmp_path

        isolated_path = dummy_manager.add_font_directory_and_remove_system(
            fontconfig_path=fontconfig_path, font_dir=font_dir
        )
        assert isolated_path.name == f"{FontconfigMode.SYSTEM_ISOLATED}.conf"
        assert isolated_path.parent == tmp_path

        template_path = dummy_manager.create_fontconfig_from_template(
            font_dir=font_dir, fontconfig_output_dir=tmp_path, mode=FontconfigMode.TEMPLATE_MINIMAL
        )
        assert template_path.name == f"{FontconfigMode.TEMPLATE_MINIMAL}.conf"
        assert template_path.parent == tmp_path


class TestDarwinFontconfigManager:
    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_insert_font_dir(
        self,
        darwin_manager: DarwinFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test inserting font directory after description element."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = darwin_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

        updated_tree = darwin_manager._insert_font_dir(tree=tree, font_dir=font_dir)
        dirs = updated_tree.findall(".//dir")
        assert len(dirs) == test_case.expected_dir_count + 1
        assert dirs[0].text == str(font_dir)

        root = updated_tree.getroot()
        elements = list(root)

        description_exists = any(el.tag == "description" for el in elements)
        if description_exists:
            # description + new dir + original dirs
            assert len(elements) == 1 + 1 + test_case.expected_dir_count
            assert elements[0].tag == "description"
            assert elements[1].tag == "dir"
        else:
            # new dir + original dirs
            assert len(elements) == 1 + test_case.expected_dir_count
            assert elements[0].tag == "dir"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_remove_system_fonts(
        self,
        darwin_manager: DarwinFontconfigManager,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test removing system fonts."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = darwin_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)
        initial_dirs_count = len(tree.findall(".//dir"))

        updated_tree = darwin_manager._remove_system_fonts(tree=tree)
        remaining_dirs = updated_tree.findall(".//dir")

        assert len(remaining_dirs) <= initial_dirs_count

        for dir_element in remaining_dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            # verify no system patterns remain
            assert not any(dir_text.startswith(pattern) for pattern in DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
            # verify no xdg prefixes remain
            assert dir_element.get("prefix") != "xdg"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_add_font_directory(
        self,
        darwin_manager: DarwinFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test adding a font directory to an existing fontconfig file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        result_path = darwin_manager.add_font_directory(fontconfig_path=fontconfig_path, font_dir=font_dir)

        expected_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_EXTENDED.value).with_suffix(".conf").resolve()
        )
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()

        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        assert len(dirs) == test_case.expected_dir_count + 1
        assert dirs[0].text == str(font_dir)

    @pytest.mark.parametrize("test_case", DARWIN_ONLY_VALID_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_add_font_directory_and_remove_system(
        self,
        darwin_manager: DarwinFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test adding a font directory and removing system font directories."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        # programmatically determine how many dirs should be removed from the original file
        original_tree = etree.parse(str(fontconfig_path))
        original_dirs = original_tree.findall(".//dir")
        initial_dir_count = len(original_dirs)
        assert initial_dir_count == test_case.expected_dir_count  # sanity check

        dirs_to_be_removed_count = 0
        for dir_element in original_dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            if (
                any(dir_text.startswith(pattern) for pattern in DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
                or dir_element.get("prefix") == "xdg"
            ):
                dirs_to_be_removed_count += 1

        # perform the actual operation
        result_path = darwin_manager.add_font_directory_and_remove_system(
            fontconfig_path=fontconfig_path, font_dir=font_dir
        )

        # verify the output file
        expected_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_ISOLATED.value).with_suffix(".conf").resolve()
        )
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()

        # verify the content of the output file
        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        expected_final_count = (initial_dir_count - dirs_to_be_removed_count) + 1
        assert len(dirs) == expected_final_count
        assert dirs[0].text == str(font_dir)  # new dir should be first

        # check that no system font directories remain
        for dir_element in dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            if dir_text == str(font_dir):
                continue  # skip the directory we just added
            assert not any(dir_text.startswith(pattern) for pattern in DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
            assert dir_element.get("prefix") != "xdg"

    @pytest.mark.parametrize(
        "mode", SUPPORTED_TEMPLATE_MODES, ids=[f"mode_{m.name.lower()}" for m in SUPPORTED_TEMPLATE_MODES]
    )
    def test_create_fontconfig_from_template(
        self,
        darwin_manager: DarwinFontconfigManager,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
        mode: FontconfigMode,
    ) -> None:
        """Test creating a fontconfig from supported templates."""
        # make the output directory unique for each parameterized run to avoid conflicts
        output_dir = tmp_path.joinpath(f"template_output_{mode.value}").resolve()

        result_path = darwin_manager.create_fontconfig_from_template(
            font_dir=font_dir, fontconfig_output_dir=output_dir, mode=mode
        )

        expected_path = output_dir.joinpath(mode.value).with_suffix(".conf").resolve()
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()
        assert output_dir.exists()
        assert output_dir.is_dir()

        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        assert len(dirs) == 1
        assert dirs[0].text == str(font_dir)

    def test_create_fontconfig_from_template_invalid_mode(
        self,
        darwin_manager: DarwinFontconfigManager,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
    ) -> None:
        """Test creating fontconfig from template with an unsupported mode."""
        output_dir = tmp_path.joinpath("template_output_invalid").resolve()
        # use a mode that is not supported by the create_from_template method
        invalid_mode = FontconfigMode.SYSTEM_COPY

        with pytest.raises(ValueError, match=f"Unsupported template mode: {invalid_mode}"):
            darwin_manager.create_fontconfig_from_template(
                font_dir=font_dir, fontconfig_output_dir=output_dir, mode=invalid_mode
            )


class TestLinuxFontconfigManager:
    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_insert_font_dir(
        self,
        linux_manager: LinuxFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test inserting font directory after description element."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = linux_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)

        updated_tree = linux_manager._insert_font_dir(tree=tree, font_dir=font_dir)
        dirs = updated_tree.findall(".//dir")
        assert len(dirs) == test_case.expected_dir_count + 1
        assert dirs[0].text == str(font_dir)

        root = updated_tree.getroot()
        elements = list(root)

        description_exists = any(el.tag == "description" for el in elements)
        if description_exists:
            # description + new dir + original dirs
            assert len(elements) == 1 + 1 + test_case.expected_dir_count
            assert elements[0].tag == "description"
            assert elements[1].tag == "dir"
        else:
            # new dir + original dirs
            assert len(elements) == 1 + test_case.expected_dir_count
            assert elements[0].tag == "dir"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_remove_system_fonts(
        self,
        linux_manager: LinuxFontconfigManager,
        test_case: ValidFontConfigTestCase,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test removing system fonts on Linux."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        tree = linux_manager._parse_xml_config_file(fontconfig_path=fontconfig_path)
        initial_dirs_count = len(tree.findall(".//dir"))

        updated_tree = linux_manager._remove_system_fonts(tree=tree)
        remaining_dirs = updated_tree.findall(".//dir")

        assert len(remaining_dirs) <= initial_dirs_count

        for dir_element in remaining_dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            # verify no system patterns remain
            assert not any(dir_text.startswith(pattern) for pattern in LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
            # verify no xdg prefixes remain
            assert dir_element.get("prefix") != "xdg"

    @pytest.mark.parametrize("test_case", VALID_FONTCONFIG_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_add_font_directory(
        self,
        linux_manager: LinuxFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test adding a font directory to an existing fontconfig file."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        result_path = linux_manager.add_font_directory(fontconfig_path=fontconfig_path, font_dir=font_dir)

        expected_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_EXTENDED.value).with_suffix(".conf").resolve()
        )
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()

        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        assert len(dirs) == test_case.expected_dir_count + 1
        assert dirs[0].text == str(font_dir)

    @pytest.mark.parametrize("test_case", LINUX_ONLY_VALID_TEST_CASES, ids=lambda tc: tc.test_name)
    def test_add_font_directory_and_remove_system(
        self,
        linux_manager: LinuxFontconfigManager,
        test_case: ValidFontConfigTestCase,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test adding a font directory and removing system font directories on Linux."""
        fontconfig_path = fontconfig_file_factory(test_case.test_name)

        # programmatically determine how many dirs should be removed from the original file
        original_tree = etree.parse(str(fontconfig_path))
        original_dirs = original_tree.findall(".//dir")
        initial_dir_count = len(original_dirs)
        assert initial_dir_count == test_case.expected_dir_count  # sanity check

        dirs_to_be_removed_count = 0
        for dir_element in original_dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            if (
                any(dir_text.startswith(pattern) for pattern in LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
                or dir_element.get("prefix") == "xdg"
            ):
                dirs_to_be_removed_count += 1

        # perform the actual operation
        result_path = linux_manager.add_font_directory_and_remove_system(
            fontconfig_path=fontconfig_path, font_dir=font_dir
        )

        # verify the output file
        expected_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_ISOLATED.value).with_suffix(".conf").resolve()
        )
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()

        # verify the content of the output file
        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        expected_final_count = (initial_dir_count - dirs_to_be_removed_count) + 1
        assert len(dirs) == expected_final_count
        assert dirs[0].text == str(font_dir)  # new dir should be first

        # check that no system font directories remain
        for dir_element in dirs:
            dir_text = dir_element.text.strip() if dir_element.text else ""
            if dir_text == str(font_dir):
                continue  # skip the directory we just added
            assert not any(dir_text.startswith(pattern) for pattern in LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
            assert dir_element.get("prefix") != "xdg"

    @pytest.mark.parametrize(
        "mode", SUPPORTED_TEMPLATE_MODES, ids=[f"mode_{m.name.lower()}" for m in SUPPORTED_TEMPLATE_MODES]
    )
    def test_create_fontconfig_from_template(
        self,
        linux_manager: LinuxFontconfigManager,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
        mode: FontconfigMode,
    ) -> None:
        """Test creating a fontconfig from supported templates on Linux."""
        # make the output directory unique for each parameterized run to avoid conflicts
        output_dir = tmp_path.joinpath(f"template_output_{mode.value}").resolve()

        result_path = linux_manager.create_fontconfig_from_template(
            font_dir=font_dir, fontconfig_output_dir=output_dir, mode=mode
        )

        expected_path = output_dir.joinpath(mode.value).with_suffix(".conf").resolve()
        assert result_path == expected_path
        assert result_path.exists()
        assert result_path.is_file()
        assert output_dir.exists()
        assert output_dir.is_dir()

        tree = etree.parse(str(result_path))
        assert tree.getroot().tag == "fontconfig"

        dirs = tree.findall(".//dir")
        assert len(dirs) == 1
        assert dirs[0].text == str(font_dir)

    def test_create_fontconfig_from_template_invalid_mode(
        self,
        linux_manager: LinuxFontconfigManager,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
    ) -> None:
        """Test creating fontconfig from template with an unsupported mode on Linux."""
        output_dir = tmp_path.joinpath("template_output_invalid").resolve()
        # use a mode that is not supported by the create_from_template method
        invalid_mode = FontconfigMode.SYSTEM_COPY

        with pytest.raises(ValueError, match=f"Unsupported template mode: {invalid_mode}"):
            linux_manager.create_fontconfig_from_template(
                font_dir=font_dir, fontconfig_output_dir=output_dir, mode=invalid_mode
            )
