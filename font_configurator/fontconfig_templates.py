# Copyright 2025- Pavel Stepachev
# SPDX-License-Identifier: Apache-2.0

from string import Template

# --- DARWIN (MacOS) ---
DARWIN_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL = Template("""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig>
    <dir>$font_dir</dir>
</fontconfig>
""")

DARWIN_FONTCONFIG_TEMPLATE_INSERT_FONT_DIR = Template("\t<dir>$font_dir</dir>\n")


DARWIN_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS = ("/System/Library/", "/Library", "~/Library", "~/.fonts")

# --- LINUX ---
LINUX_FONTCONFIG_TEMPLATE_CONFIG_MINIMAL = Template("""<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig>
    <dir>$font_dir</dir>
</fontconfig>
""")

LINUX_FONTCONFIG_TEMPLATE_INSERT_FONT_DIR = Template("\t<dir>$font_dir</dir>\n")
LINUX_FONTCONFIG_REMOVE_SYSTEM_FONTS_PATTERNS = ("/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts")
