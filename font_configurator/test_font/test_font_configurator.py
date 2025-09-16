# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from font_configurator.font_configurator import FontConfigurator
from font_configurator.fontconfig_managers import (
    BaseFontconfigManager,
    DarwinFontconfigManager,
    FontconfigMode,
    LinuxFontconfigManager,
    SupportedPlatforms,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


@dataclass
class PlatformTestCase:
    """Encapsulates all platform-specific details for a test run."""

    platform_enum: SupportedPlatforms
    manager_class: type[BaseFontconfigManager]
    expected_env_vars: dict[str, str | None]


PLATFORM_TEST_CASES = [
    pytest.param(
        PlatformTestCase(
            platform_enum=SupportedPlatforms.DARWIN,
            manager_class=DarwinFontconfigManager,
            expected_env_vars={"PANGOCAIRO_BACKEND": "fontconfig", "FONTCONFIG_FILE": "{result_path}"},
        ),
        id="darwin",
    ),
    pytest.param(
        PlatformTestCase(
            platform_enum=SupportedPlatforms.LINUX,
            manager_class=LinuxFontconfigManager,
            expected_env_vars={"PANGOCAIRO_BACKEND": "fontconfig", "FONTCONFIG_FILE": "{result_path}"},
        ),
        id="linux",
    ),
]


def validate_from_file(result_path: pathlib.Path, test_paths: dict[str, Any]) -> None:
    """Validator for FROM_FILE mode."""
    assert result_path == test_paths["source_path"].resolve()
    assert not test_paths["dest_dir"].exists()  # should not create anything in dest_dir


def validate_copy(result_path: pathlib.Path, test_paths: dict[str, Any]) -> None:
    """Validator for SYSTEM_COPY mode."""
    assert result_path.exists()
    assert result_path.parent == test_paths["dest_dir"]
    assert result_path.read_text() == test_paths["source_path"].read_text()


def validate_font_dir_added(result_path: pathlib.Path, test_paths: dict[str, Any]) -> None:
    """Validator for modes that add a font directory."""
    assert result_path.exists()
    assert result_path.parent == test_paths["dest_dir"]
    assert str(test_paths["font_dir"]) in result_path.read_text()


@dataclass
class SetupModeTestCase:
    """Encapsulates all mode-specific details for a setup_font test."""

    mode: FontconfigMode
    # list of keyword arguments required by this mode (from the test fixtures)
    required_args: list[str] = field(default_factory=list)
    # function to validate the output
    validator: Callable[[pathlib.Path, dict[str, Any]], None] | None = None
    # source file to use for the test
    source_fixture: str = "valid_dummy__single_font_dir"


SETUP_MODE_TEST_CASES = [
    pytest.param(
        SetupModeTestCase(
            mode=FontconfigMode.FROM_FILE,
            validator=validate_from_file,
        ),
        id="mode_from_file",
    ),
    pytest.param(
        SetupModeTestCase(
            mode=FontconfigMode.SYSTEM_COPY,
            required_args=["fontconfig_destination_dir"],
            validator=validate_copy,
        ),
        id="mode_system_copy",
    ),
    pytest.param(
        SetupModeTestCase(
            mode=FontconfigMode.SYSTEM_EXTENDED,
            required_args=["fontconfig_destination_dir", "font_dir"],
            validator=validate_font_dir_added,
        ),
        id="mode_system_extended",
    ),
    pytest.param(
        SetupModeTestCase(
            mode=FontconfigMode.SYSTEM_ISOLATED,
            required_args=["fontconfig_destination_dir", "font_dir"],
            validator=validate_font_dir_added,
            source_fixture="valid_darwin__v1",  # this mode benefits from a more realistic source
        ),
        id="mode_system_isolated",
    ),
    pytest.param(
        SetupModeTestCase(
            mode=FontconfigMode.TEMPLATE_MINIMAL,
            required_args=["fontconfig_destination_dir", "font_dir"],
            validator=validate_font_dir_added,
        ),
        id="mode_template_minimal",
    ),
]


class TestFontConfigurator:
    """Tests for the FontConfigurator class."""

    @pytest.fixture(autouse=True)
    def clean_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure fontconfig environment variables are cleaned up after each test."""
        for var in [
            "PANGOCAIRO_BACKEND",
            "FONTCONFIG_FILE",
        ]:
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def mock_platform(self, monkeypatch: pytest.MonkeyPatch, platform_case: PlatformTestCase) -> None:
        """Mocks the OS for a given platform test case."""
        # Since SupportedPlatforms is a StrEnum, we can use it directly.
        monkeypatch.setattr(platform, "system", lambda: platform_case.platform_enum)

    def check_env_vars(self, platform_case: PlatformTestCase, result_path: pathlib.Path) -> None:
        """Helper to dynamically check environment variables."""
        for var, expected_value in platform_case.expected_env_vars.items():
            if expected_value is None:
                assert var not in os.environ, f"Variable '{var}' should not be set for {platform_case.platform_enum}"
            else:
                formatted_value = expected_value.format(result_path=result_path)
                assert os.environ.get(var) == formatted_value, (
                    f"Incorrect value for '{var}' on {platform_case.platform_enum}"
                )

    @pytest.mark.usefixtures("mock_platform")
    @pytest.mark.parametrize("platform_case", PLATFORM_TEST_CASES)
    def test_init_supported_platform(
        self,
        platform_case: PlatformTestCase,
    ) -> None:
        """Test initialization on all supported platforms."""
        configurator = FontConfigurator()
        assert configurator.detected_system == platform_case.platform_enum
        assert isinstance(configurator.font_manager, platform_case.manager_class)

    @pytest.mark.parametrize("unsupported_system", ["Windows", "Java", "", "iOS", "iPadOS", "Android"])
    def test_init_unsupported_platform(self, monkeypatch: pytest.MonkeyPatch, unsupported_system: str) -> None:
        """Test that initialization fails on an unsupported platform."""
        monkeypatch.setattr(platform, "system", lambda: unsupported_system)
        with pytest.raises(NotImplementedError, match=f"Unsupported platform: '{unsupported_system}'"):
            FontConfigurator()

    @pytest.mark.usefixtures("mock_platform")
    @pytest.mark.parametrize("mode_case", SETUP_MODE_TEST_CASES)
    @pytest.mark.parametrize("platform_case", PLATFORM_TEST_CASES)
    def test_setup_font_modes(
        self,
        platform_case: PlatformTestCase,
        mode_case: SetupModeTestCase,
        tmp_path: pathlib.Path,
        font_dir: pathlib.Path,
        fontconfig_file_factory: Callable[[str], pathlib.Path],
    ) -> None:
        """Test all setup_font modes across all supported platforms."""
        configurator = FontConfigurator()

        dest_dir = tmp_path.joinpath("output")
        source_path = fontconfig_file_factory(mode_case.source_fixture)

        # map fixture names to their values
        available_args = {
            "fontconfig_source_path": source_path,
            "fontconfig_destination_dir": dest_dir,
            "font_dir": font_dir,
        }
        # select only the arguments required for the current mode
        call_args = {arg: available_args[arg] for arg in mode_case.required_args}

        result_path = configurator.setup_font(
            mode=mode_case.mode,
            fontconfig_source_path=source_path,
            force_reinitialize=True,
            **call_args,
        )

        # 1. validate the output file itself
        if mode_case.validator:
            test_paths = {"source_path": source_path, "dest_dir": dest_dir, "font_dir": font_dir}
            mode_case.validator(result_path, test_paths)

        # 2. validate the environment variables were set correctly for the platform
        self.check_env_vars(platform_case, result_path)

    @pytest.mark.usefixtures("mock_platform")
    @pytest.mark.parametrize("platform_case", PLATFORM_TEST_CASES)
    @pytest.mark.parametrize(
        ("mode", "missing_args", "expected_error_msg"),
        [
            pytest.param(
                FontconfigMode.FROM_FILE,
                {"fontconfig_source_path": None},
                "requires 'fontconfig_source_path'",
                id="from_file_missing_source",
            ),
            pytest.param(
                FontconfigMode.SYSTEM_EXTENDED,
                {"font_dir": None},
                "requires 'font_dir'",
                id="system_extended_missing_font_dir",
            ),
            pytest.param(
                FontconfigMode.SYSTEM_ISOLATED,
                {"font_dir": None},
                "requires 'font_dir'",
                id="system_isolated_missing_font_dir",
            ),
            pytest.param(
                FontconfigMode.TEMPLATE_MINIMAL,
                {"font_dir": None},
                "requires 'font_dir'",
                id="template_minimal_missing_font_dir",
            ),
            pytest.param(
                FontconfigMode.SYSTEM_COPY,
                {"fontconfig_destination_dir": None},
                "requires 'fontconfig_destination_dir'",
                id="system_copy_missing_dest",
            ),
            pytest.param(
                FontconfigMode.SYSTEM_EXTENDED,
                {"fontconfig_destination_dir": None},
                "requires 'fontconfig_destination_dir'",
                id="system_extended_missing_dest",
            ),
            pytest.param(
                FontconfigMode.SYSTEM_ISOLATED,
                {"fontconfig_destination_dir": None},
                "requires 'fontconfig_destination_dir'",
                id="system_isolated_missing_dest",
            ),
            pytest.param(
                FontconfigMode.TEMPLATE_MINIMAL,
                {"fontconfig_destination_dir": None},
                "requires 'fontconfig_destination_dir'",
                id="template_minimal_missing_dest",
            ),
        ],
    )
    def test_setup_font_missing_arguments(
        self,
        platform_case: PlatformTestCase,  # noqa: ARG002
        mode: FontconfigMode,
        missing_args: dict[str, Any],
        expected_error_msg: str,
    ) -> None:
        """Test that setup_font raises ValueError for missing arguments across all platforms."""
        configurator = FontConfigurator()

        base_args = {
            "fontconfig_source_path": "dummy.conf",
            "font_dir": "dummy_fonts/",
            "fontconfig_destination_dir": "dummy_output/",
        }
        final_args = {**base_args, **missing_args}

        with pytest.raises(ValueError, match=expected_error_msg):
            configurator.setup_font(mode=mode, **final_args)

    @pytest.mark.usefixtures("mock_platform")
    @pytest.mark.parametrize("platform_case", PLATFORM_TEST_CASES)
    def test_setup_font_unknown_mode(
        self,
        platform_case: PlatformTestCase,  # noqa: ARG002
    ) -> None:
        """Test that setup_font raises ValueError for an unknown mode across all platforms."""
        configurator = FontConfigurator()
        with pytest.raises(ValueError, match="Unknown font configuration mode"):
            configurator.setup_font(mode="INVALID_MODE", fontconfig_source_path="dummy.conf")  # type: ignore
