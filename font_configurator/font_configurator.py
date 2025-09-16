# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import logging
import os
import pathlib
import platform
import sys
from ctypes.util import find_library
from typing import ClassVar

import gi

from font_configurator.fontconfig_managers import (
    BaseFontconfigManager,
    DarwinFontconfigManager,
    FontconfigMode,
    LinuxFontconfigManager,
    SupportedPlatforms,
)

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # type: ignore  # noqa: E402, F401


class FontConfigurator:
    manager_map: ClassVar[dict[SupportedPlatforms, type[BaseFontconfigManager]]] = {
        SupportedPlatforms.DARWIN: DarwinFontconfigManager,  # macOS
        SupportedPlatforms.LINUX: LinuxFontconfigManager,  # Linux
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.detected_system = self._detect_supported_system()
        self.font_manager = self._get_fontconfig_manager()

    def _detect_supported_system(self) -> SupportedPlatforms:
        system = platform.system()
        self.logger.debug("Detected system: %s", system)

        try:
            return SupportedPlatforms(system)
        except ValueError as unsupported_system:
            msg = f"Unsupported platform: '{system}'. Supported: {[p.name for p in SupportedPlatforms]}"
            self.logger.exception(msg)
            raise NotImplementedError(msg) from unsupported_system

    def _get_fontconfig_manager(self) -> BaseFontconfigManager:
        manager_class = self.manager_map.get(self.detected_system)
        if manager_class is None:
            msg = f"No fontconfig manager for platform: {self.detected_system}"
            self.logger.error(msg)
            raise NotImplementedError(msg)

        return manager_class()

    def _load_from_file(self, fontconfig_path: pathlib.Path) -> pathlib.Path:
        return self.font_manager.load_fontconfig_file_from_path(fontconfig_path=fontconfig_path)

    def _copy_system_fontconfig_file(
        self,
        fontconfig_source_path: pathlib.Path,
        fontconfig_destination_dir: pathlib.Path,
    ) -> pathlib.Path:
        return self.font_manager.copy_fontconfig_file(
            fontconfig_source_path=fontconfig_source_path,
            fontconfig_destination_dir=fontconfig_destination_dir,
        )

    def _add_font_directory(
        self,
        fontconfig_source_path: pathlib.Path,
        fontconfig_destination_dir: pathlib.Path,
        font_dir: pathlib.Path,
    ) -> pathlib.Path:
        copy_path = self.font_manager.copy_fontconfig_file(
            fontconfig_source_path=fontconfig_source_path, fontconfig_destination_dir=fontconfig_destination_dir
        )
        return self.font_manager.add_font_directory(fontconfig_path=copy_path, font_dir=font_dir)

    def _add_font_directory_and_remove_system(
        self,
        fontconfig_source_path: pathlib.Path,
        fontconfig_destination_dir: pathlib.Path,
        font_dir: pathlib.Path,
    ) -> pathlib.Path:
        copy_path = self.font_manager.copy_fontconfig_file(
            fontconfig_source_path=fontconfig_source_path, fontconfig_destination_dir=fontconfig_destination_dir
        )
        return self.font_manager.add_font_directory_and_remove_system(fontconfig_path=copy_path, font_dir=font_dir)

    def _create_fontconfig_from_template(
        self,
        fontconfig_destination_dir: pathlib.Path,
        font_dir: pathlib.Path,
        mode: FontconfigMode,
    ) -> pathlib.Path:
        return self.font_manager.create_fontconfig_from_template(
            font_dir=font_dir, fontconfig_output_dir=fontconfig_destination_dir, mode=mode
        )

    def _validate_inputs(
        self,
        mode: FontconfigMode,
        fontconfig_source_path: pathlib.Path | str | None,
        font_dir: pathlib.Path | str | None,
        fontconfig_destination_dir: pathlib.Path | str | None,
    ) -> tuple[pathlib.Path | None, pathlib.Path | None, pathlib.Path | None]:
        # Initialize variables as None
        validated_fontconfig_source_path: pathlib.Path | None = None
        validated_font_dir: pathlib.Path | None = None
        validated_fontconfig_destination_dir: pathlib.Path | None = None

        if mode in (
            FontconfigMode.FROM_FILE,
            FontconfigMode.SYSTEM_COPY,
            FontconfigMode.SYSTEM_EXTENDED,
            FontconfigMode.SYSTEM_ISOLATED,
        ):
            if fontconfig_source_path is None:
                msg = f"{mode} requires 'fontconfig_source_path'"
                self.logger.error(msg)
                raise ValueError(msg)

            validated_fontconfig_source_path = pathlib.Path(fontconfig_source_path).resolve()

        if mode in (
            FontconfigMode.SYSTEM_EXTENDED,
            FontconfigMode.SYSTEM_ISOLATED,
            FontconfigMode.TEMPLATE_MINIMAL,
        ):
            if font_dir is None:
                msg = f"{mode} requires 'font_dir'"
                self.logger.error(msg)
                raise ValueError(msg)

            validated_font_dir = pathlib.Path(font_dir).resolve()

        if mode in (
            FontconfigMode.SYSTEM_COPY,
            FontconfigMode.SYSTEM_EXTENDED,
            FontconfigMode.SYSTEM_ISOLATED,
            FontconfigMode.TEMPLATE_MINIMAL,
        ):
            if fontconfig_destination_dir is None:
                msg = f"{mode} requires 'fontconfig_destination_dir'"
                self.logger.error(msg)
                raise ValueError(msg)

            validated_fontconfig_destination_dir = pathlib.Path(fontconfig_destination_dir).resolve()

        return validated_fontconfig_source_path, validated_font_dir, validated_fontconfig_destination_dir

    def _execute_mode_action(
        self,
        mode: FontconfigMode,
        fontconfig_source_path: pathlib.Path | None,
        font_dir: pathlib.Path | None,
        fontconfig_destination_dir: pathlib.Path | None,
    ) -> pathlib.Path:
        match mode:
            case FontconfigMode.FROM_FILE:
                assert fontconfig_source_path is not None
                return self._load_from_file(fontconfig_path=fontconfig_source_path)

            case FontconfigMode.SYSTEM_COPY:
                assert fontconfig_source_path is not None
                assert fontconfig_destination_dir is not None
                return self._copy_system_fontconfig_file(
                    fontconfig_source_path=fontconfig_source_path,
                    fontconfig_destination_dir=fontconfig_destination_dir,
                )
            case FontconfigMode.SYSTEM_EXTENDED:
                assert fontconfig_source_path is not None
                assert fontconfig_destination_dir is not None
                assert font_dir is not None
                return self._add_font_directory(
                    fontconfig_source_path=fontconfig_source_path,
                    fontconfig_destination_dir=fontconfig_destination_dir,
                    font_dir=font_dir,
                )
            case FontconfigMode.SYSTEM_ISOLATED:
                assert fontconfig_source_path is not None
                assert fontconfig_destination_dir is not None
                assert font_dir is not None
                return self._add_font_directory_and_remove_system(
                    fontconfig_source_path=fontconfig_source_path,
                    fontconfig_destination_dir=fontconfig_destination_dir,
                    font_dir=font_dir,
                )
            case FontconfigMode.TEMPLATE_MINIMAL:
                assert fontconfig_destination_dir is not None
                assert font_dir is not None
                return self._create_fontconfig_from_template(
                    fontconfig_destination_dir=fontconfig_destination_dir,
                    font_dir=font_dir,
                    mode=mode,
                )
            case _:
                msg = f"Unknown font configuration mode: {mode}"
                self.logger.error(msg)
                raise ValueError(msg)

    def _find_and_load_fontconfig(self) -> tuple[ctypes.CDLL, str | None]:  # noqa: C901
        """
        Finds and loads the fontconfig C library using multiple strategies.
        Returns a tuple of the loaded library object and the path it was loaded from.
        Raises RuntimeError if the library cannot be loaded.
        """
        # Strategy 1: Use ctypes.util.find_library
        lib_path = find_library("fontconfig")
        if lib_path:
            try:
                self.logger.debug("Strategy 1: Found fontconfig via find_library: %s. Attempting to load.", lib_path)
                return ctypes.CDLL(lib_path), lib_path
            except (OSError, TypeError) as e:
                self.logger.warning("find_library found '%s' but ctypes could not load it: %s", lib_path, e)

        # Strategy 2: Try common names and let the dynamic linker find them.
        self.logger.debug("Strategy 2: Trying common names for the dynamic linker.")
        darwin_names = ["libfontconfig.1.dylib", "fontconfig.dylib"]
        linux_names = ["libfontconfig.so.1", "libfontconfig.so"]
        generic_names = ["fontconfig"]

        # Order matters: try detected-platform names first, then others as a fallback.
        potential_names: list[str] = []
        if self.detected_system == SupportedPlatforms.LINUX:
            potential_names.extend(linux_names)
            potential_names.extend(darwin_names)
        else:  # Default to Darwin-first order
            potential_names.extend(darwin_names)
            potential_names.extend(linux_names)
        potential_names.extend(generic_names)

        # Remove duplicates while preserving order
        potential_names = list(dict.fromkeys(potential_names))

        for name in potential_names:
            try:
                self.logger.debug("Attempting to load '%s' with dynamic linker", name)
                return ctypes.CDLL(name), name
            except (OSError, TypeError):
                self.logger.debug("Dynamic linker failed to find and load '%s'", name)

        # Strategy 3: Manually search common directories (e.g., Homebrew, venv)
        self.logger.debug("Strategy 3: Manually searching common library paths.")
        search_dirs: list[pathlib.Path] = [
            pathlib.Path("/opt/homebrew/lib"),  # Homebrew on Apple Silicon
            pathlib.Path("/usr/local/lib"),  # Homebrew on Intel, other custom installs
            pathlib.Path("/usr/lib64"),  # Standard Linux
            pathlib.Path("/usr/lib"),  # Standard Linux
            pathlib.Path(sys.prefix) / "lib",  # Python venv/conda env
        ]

        # Use the same ordered list of library names from strategy 2
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            self.logger.debug("Searching in directory: %s", directory)
            for name in potential_names:
                candidate_path = directory / name
                if candidate_path.is_file():
                    self.logger.debug("Found candidate file: %s", candidate_path)
                    try:
                        return ctypes.CDLL(str(candidate_path)), str(candidate_path)
                    except (OSError, TypeError) as e:
                        self.logger.warning("Found candidate '%s' but failed to load: %s", candidate_path, e)

        msg = (
            "Could not find and load the fontconfig C library. Please ensure fontconfig "
            "is installed and accessible. Searched standard paths, Homebrew paths, "
            "and Python virtual environment paths."
        )
        self.logger.error(msg)
        raise RuntimeError(msg)

    def _reinitialize_fontconfig_cache(self) -> None:
        """
        Forces the underlying Fontconfig C-library to re-initialize and
        reread its configuration from environment variables.
        """
        if self.detected_system not in (SupportedPlatforms.DARWIN, SupportedPlatforms.LINUX):
            msg = f"Fontconfig cache re-initialization is not supported on this platform: {self.detected_system}"
            self.logger.error(msg)
            raise NotImplementedError(msg)

        fontconfig, lib_path = self._find_and_load_fontconfig()
        self.logger.debug("Successfully loaded fontconfig library from: %s", lib_path)

        try:
            fontconfig.FcInitReinitialize()
            self.logger.debug("Successfully called FcInitReinitialize.")
        except Exception as e:
            msg = f"Failed to call FcInitReinitialize on library at '{lib_path}': {e}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

    def _configure_environment(self, result_path: pathlib.Path, *, force_reinitialize: bool = True) -> None:
        match self.detected_system:
            case SupportedPlatforms.DARWIN | SupportedPlatforms.LINUX:
                os.environ["PANGOCAIRO_BACKEND"] = "fontconfig"
                os.environ["FONTCONFIG_FILE"] = str(result_path)
                self.logger.debug(
                    "Set environment variables for %s: PANGOCAIRO_BACKEND=%s, FONTCONFIG_FILE=%s",
                    self.detected_system,
                    os.environ["PANGOCAIRO_BACKEND"],
                    os.environ["FONTCONFIG_FILE"],
                )
                # after setting the environment, force the underlying C library to re-read it
                # this is crucial for applications/tests that switch font configs in a single process
                if force_reinitialize:
                    self._reinitialize_fontconfig_cache()

                    new_font_map = PangoCairo.FontMap.new()
                    PangoCairo.FontMap.set_default(new_font_map)  # type: ignore
                    self.logger.debug("Set new PangoCairo default FontMap to clear font cache.")
            case _:
                msg = f"Environment variables for fontconfig are not set for this platform: {self.detected_system}"
                self.logger.error(msg)
                raise NotImplementedError(msg)

    def setup_font(
        self,
        mode: FontconfigMode,
        fontconfig_source_path: pathlib.Path | str | None = None,
        font_dir: pathlib.Path | str | None = None,
        fontconfig_destination_dir: pathlib.Path | str | None = "output_fontconfig",
        *,
        force_reinitialize: bool = True,
    ) -> pathlib.Path:
        # validate inputs
        fontconfig_source_path, font_dir, fontconfig_destination_dir = self._validate_inputs(
            mode=mode,
            fontconfig_source_path=fontconfig_source_path,
            font_dir=font_dir,
            fontconfig_destination_dir=fontconfig_destination_dir,
        )

        # run mode-specific action
        result_path = self._execute_mode_action(
            mode=mode,
            fontconfig_source_path=fontconfig_source_path,
            font_dir=font_dir,
            fontconfig_destination_dir=fontconfig_destination_dir,
        )

        # configure environment variables
        self._configure_environment(result_path=result_path, force_reinitialize=force_reinitialize)

        return result_path
