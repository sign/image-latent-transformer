# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from enum import StrEnum, unique
from typing import TYPE_CHECKING, ClassVar

from lxml import etree

from font_configurator.fontconfig_templates import (
    DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS,
    DARWIN_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL,
    LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS,
    LINUX_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL,
)

if TYPE_CHECKING:
    import pathlib


@unique
class SupportedPlatforms(StrEnum):
    DARWIN = "Darwin"  # macOS
    LINUX = "Linux"


@unique
class FontconfigMode(StrEnum):
    FROM_FILE = "from_file"  # use existing fontconfig file without modifications
    SYSTEM_COPY = "system_copy"  # use system fontconfig
    SYSTEM_EXTENDED = "system_extended"  # use system fontconfig with custom font directory
    SYSTEM_ISOLATED = "system_isolated"  # use system fontconfig with custom font directory and no system fonts
    TEMPLATE_MINIMAL = "template_minimal"  # use template (minimal) fontconfig only with custom font directory


class BaseFontconfigManager(ABC):
    """Base class for fontconfig file operations."""

    fontconfig_extension: ClassVar[str] = ".conf"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _parse_xml_config_file(self, fontconfig_path: pathlib.Path) -> etree._ElementTree:
        self.logger.debug("Parsing XML fontconfig file: %s", fontconfig_path)
        return etree.parse(str(fontconfig_path))

    def _write_xml_config_file(self, fontconfig_path: pathlib.Path, tree: etree._ElementTree) -> None:
        self.logger.debug("Writing XML fontconfig file to: %s", fontconfig_path)
        etree.indent(tree, space="\t")
        tree.write(str(fontconfig_path), encoding="utf-8", xml_declaration=True, pretty_print=True)

    def _validate_conf_file_existence(self, file_path: pathlib.Path) -> None:
        if not file_path.is_file() or file_path.suffix != self.fontconfig_extension:
            msg_no_conf_file = f"No {self.fontconfig_extension} file found at {file_path}"
            raise FileNotFoundError(msg_no_conf_file)

    def load_fontconfig_file_from_path(self, fontconfig_path: pathlib.Path) -> pathlib.Path:
        """Load existing fontconfig file from the specified path."""
        result_path = fontconfig_path.resolve()
        self._validate_conf_file_existence(file_path=result_path)

        try:
            tree = self._parse_xml_config_file(fontconfig_path=result_path)
            if tree.getroot().tag != "fontconfig":
                msg_invalid_root = f"Invalid root element in fontconfig file: {tree.getroot().tag}"
                self.logger.error(msg_invalid_root)
                raise ValueError(msg_invalid_root)
        except (etree.XMLSyntaxError, etree.ParseError) as e:
            msg_invalid_xml = f"Invalid XML in fontconfig file: {e}"
            self.logger.exception(msg_invalid_xml)
            raise

        self.logger.debug("Loading fontconfig from path: %s", result_path)
        return result_path

    def copy_fontconfig_file(
        self, fontconfig_source_path: pathlib.Path, fontconfig_destination_dir: pathlib.Path
    ) -> pathlib.Path:
        """Copy fontconfig file from source to output directory."""
        self.logger.debug("Copying fontconfig file from %s to %s", fontconfig_source_path, fontconfig_destination_dir)

        fontconfig_source_path = fontconfig_source_path.resolve()
        self._validate_conf_file_existence(file_path=fontconfig_source_path)

        fontconfig_destination_dir.mkdir(parents=True, exist_ok=True)
        result_path = (
            fontconfig_destination_dir.joinpath(FontconfigMode.SYSTEM_COPY)
            .with_suffix(self.fontconfig_extension)
            .resolve()
        )
        shutil.copy(fontconfig_source_path, result_path)
        self.logger.debug("Fontconfig copied to: %s", result_path)
        return result_path

    @abstractmethod
    def add_font_directory(self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path) -> pathlib.Path:
        """Add custom font directory to the fontconfig file."""
        ...

    @abstractmethod
    def add_font_directory_and_remove_system(
        self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path
    ) -> pathlib.Path:
        """Add custom font directory and remove system fonts."""
        ...

    @abstractmethod
    def create_fontconfig_from_template(
        self, font_dir: pathlib.Path, fontconfig_output_dir: pathlib.Path, mode: FontconfigMode
    ) -> pathlib.Path: ...


class DarwinFontconfigManager(BaseFontconfigManager):
    """Darwin-specific implementation of fontconfig operations."""

    def _insert_font_dir(self, tree: etree._ElementTree, font_dir: pathlib.Path) -> etree._ElementTree:
        root = tree.getroot()

        dir_element = etree.Element("dir")
        dir_element.text = str(font_dir)

        # attempt #1: insert after <description></description>
        description_element = root.find("description")
        if description_element is not None:
            # Insert after the description element
            description_element.addnext(dir_element)
        # attempt #2: insert at the beginning if root tag is <fontconfig>
        elif root.tag == "fontconfig":
            root.insert(0, dir_element)
        else:
            msg_fail = "Failed to insert <dir> element: 1) <description> not found, 2) root tag is not <fontconfig>"
            self.logger.error(msg_fail)
            raise ValueError(msg_fail)
        return tree

    def _remove_system_fonts(self, tree: etree._ElementTree) -> etree._ElementTree:
        root = tree.getroot()

        # Find all dir elements to remove
        dirs_to_remove = []
        for dir_element in root.xpath("//dir"):
            dir_text = dir_element.text.strip() if dir_element.text else ""

            # check text patterns or xdg prefix attribute (like <dir prefix="xdg">fonts</dir>)
            if (
                any(dir_text.startswith(pattern) for pattern in DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
                or dir_element.get("prefix") == "xdg"
            ):
                dirs_to_remove.append(dir_element)

        # Remove the elements
        for dir_element in dirs_to_remove:
            parent = dir_element.getparent()
            if parent is not None:
                parent.remove(dir_element)

        return tree

    def add_font_directory(self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path) -> pathlib.Path:
        fontconfig_path = fontconfig_path.resolve()
        self._validate_conf_file_existence(file_path=fontconfig_path)
        font_dir = font_dir.resolve()

        self.logger.debug("Adding font directory %s to config %s", font_dir, fontconfig_path)

        tree = self._parse_xml_config_file(fontconfig_path=fontconfig_path)
        tree = self._insert_font_dir(tree=tree, font_dir=font_dir)
        result_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_EXTENDED)
            .with_suffix(self.fontconfig_extension)
            .resolve()
        )
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("New fontconfig created at: %s", result_path)
        return result_path

    def add_font_directory_and_remove_system(
        self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path
    ) -> pathlib.Path:
        fontconfig_path = fontconfig_path.resolve()
        self._validate_conf_file_existence(file_path=fontconfig_path)
        font_dir = font_dir.resolve()

        self.logger.debug("Adding font directory %s and removing system fonts from %s", font_dir, fontconfig_path)

        tree = self._parse_xml_config_file(fontconfig_path=fontconfig_path)
        tree = self._remove_system_fonts(tree=tree)
        tree = self._insert_font_dir(tree=tree, font_dir=font_dir)

        result_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_ISOLATED)
            .with_suffix(self.fontconfig_extension)
            .resolve()
        )
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("New fontconfig created at: %s", result_path)
        return result_path

    def create_fontconfig_from_template(
        self, font_dir: pathlib.Path, fontconfig_output_dir: pathlib.Path, mode: FontconfigMode
    ) -> pathlib.Path:
        """Create fontconfig file from a predefined template."""
        font_dir = font_dir.resolve()
        fontconfig_output_dir = fontconfig_output_dir.resolve()

        template_map = {
            FontconfigMode.TEMPLATE_MINIMAL: DARWIN_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL,
        }
        if mode not in template_map:
            msg_error_mode = f"Unsupported template mode: {mode}"
            self.logger.error(msg_error_mode)
            raise ValueError(msg_error_mode)

        self.logger.debug("Creating fontconfig with mode %s and font directory %s", mode, font_dir)

        template = template_map[mode]
        config_content = template.substitute(font_dir=str(font_dir))

        # Parse the string content into an lxml element
        root = etree.fromstring(config_content.encode("utf-8"))
        tree = etree.ElementTree(root)

        fontconfig_output_dir.mkdir(parents=True, exist_ok=True)
        result_path = fontconfig_output_dir.joinpath(mode).with_suffix(self.fontconfig_extension).resolve()
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("Fontconfig created at: %s", result_path)
        return result_path


class LinuxFontconfigManager(BaseFontconfigManager):
    """Linux-specific implementation of fontconfig operations."""

    def _insert_font_dir(self, tree: etree._ElementTree, font_dir: pathlib.Path) -> etree._ElementTree:
        root = tree.getroot()

        dir_element = etree.Element("dir")
        dir_element.text = str(font_dir)

        # attempt #1: insert after <description></description>
        description_element = root.find("description")
        if description_element is not None:
            # Insert after the description element
            description_element.addnext(dir_element)
        # attempt #2: insert at the beginning if root tag is <fontconfig>
        elif root.tag == "fontconfig":
            root.insert(0, dir_element)
        else:
            msg_fail = "Failed to insert <dir> element: 1) <description> not found, 2) root tag is not <fontconfig>"
            self.logger.error(msg_fail)
            raise ValueError(msg_fail)
        return tree

    def _remove_system_fonts(self, tree: etree._ElementTree) -> etree._ElementTree:
        root = tree.getroot()

        # Find all dir elements to remove
        dirs_to_remove = []
        for dir_element in root.xpath("//dir"):
            dir_text = dir_element.text.strip() if dir_element.text else ""

            # check text patterns or xdg prefix attribute (like <dir prefix="xdg">fonts</dir>)
            if (
                any(dir_text.startswith(pattern) for pattern in LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS)
                or dir_element.get("prefix") == "xdg"
            ):
                dirs_to_remove.append(dir_element)

        # Remove the elements
        for dir_element in dirs_to_remove:
            parent = dir_element.getparent()
            if parent is not None:
                parent.remove(dir_element)

        return tree

    def add_font_directory(self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path) -> pathlib.Path:
        fontconfig_path = fontconfig_path.resolve()
        self._validate_conf_file_existence(file_path=fontconfig_path)
        font_dir = font_dir.resolve()

        self.logger.debug("Adding font directory %s to config %s", font_dir, fontconfig_path)

        tree = self._parse_xml_config_file(fontconfig_path=fontconfig_path)
        tree = self._insert_font_dir(tree=tree, font_dir=font_dir)
        result_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_EXTENDED)
            .with_suffix(self.fontconfig_extension)
            .resolve()
        )
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("New fontconfig created at: %s", result_path)
        return result_path

    def add_font_directory_and_remove_system(
        self, fontconfig_path: pathlib.Path, font_dir: pathlib.Path
    ) -> pathlib.Path:
        fontconfig_path = fontconfig_path.resolve()
        self._validate_conf_file_existence(file_path=fontconfig_path)
        font_dir = font_dir.resolve()

        self.logger.debug("Adding font directory %s and removing system fonts from %s", font_dir, fontconfig_path)

        tree = self._parse_xml_config_file(fontconfig_path=fontconfig_path)
        tree = self._remove_system_fonts(tree=tree)
        tree = self._insert_font_dir(tree=tree, font_dir=font_dir)

        result_path = (
            fontconfig_path.parent.joinpath(FontconfigMode.SYSTEM_ISOLATED)
            .with_suffix(self.fontconfig_extension)
            .resolve()
        )
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("New fontconfig created at: %s", result_path)
        return result_path

    def create_fontconfig_from_template(
        self, font_dir: pathlib.Path, fontconfig_output_dir: pathlib.Path, mode: FontconfigMode
    ) -> pathlib.Path:
        """Create fontconfig file from a predefined template."""
        font_dir = font_dir.resolve()
        fontconfig_output_dir = fontconfig_output_dir.resolve()

        template_map = {
            FontconfigMode.TEMPLATE_MINIMAL: LINUX_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL,
        }
        if mode not in template_map:
            msg_error_mode = f"Unsupported template mode: {mode}"
            self.logger.error(msg_error_mode)
            raise ValueError(msg_error_mode)

        self.logger.debug("Creating fontconfig with mode %s and font directory %s", mode, font_dir)

        template = template_map[mode]
        config_content = template.substitute(font_dir=str(font_dir))

        # Parse the string content into an lxml element
        root = etree.fromstring(config_content.encode("utf-8"))
        tree = etree.ElementTree(root)

        fontconfig_output_dir.mkdir(parents=True, exist_ok=True)
        result_path = fontconfig_output_dir.joinpath(mode).with_suffix(self.fontconfig_extension).resolve()
        self._write_xml_config_file(fontconfig_path=result_path, tree=tree)
        self.logger.debug("Fontconfig created at: %s", result_path)
        return result_path
