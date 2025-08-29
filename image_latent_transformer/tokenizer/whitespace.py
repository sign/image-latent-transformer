# White_Space=yes characters (from https://en.wikipedia.org/wiki/Whitespace_character)

WHITE_SPACE_CODES = [
    0x0009,  # TAB
    0x000A,  # LF
    0x000B,  # VT
    0x000C,  # FF
    0x000D,  # CR
    0x0020,  # SPACE
    0x0085,  # NEL
    0x00A0,  # NO-BREAK SPACE
    0x1680,  # OGHAM SPACE MARK
    0x2000,  # EN QUAD
    0x2001,  # EM QUAD
    0x2002,  # EN SPACE
    0x2003,  # EM SPACE
    0x2004,  # THREE-PER-EM SPACE
    0x2005,  # FOUR-PER-EM SPACE
    0x2006,  # SIX-PER-EM SPACE
    0x2007,  # FIGURE SPACE
    0x2008,  # PUNCTUATION SPACE
    0x2009,  # THIN SPACE
    0x200A,  # HAIR SPACE
    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    0x202F,  # NARROW NO-BREAK SPACE
    0x205F,  # MEDIUM MATHEMATICAL SPACE
    0x3000,  # IDEOGRAPHIC SPACE
]

WHITE_SPACE_CHARS = [chr(cp) for cp in WHITE_SPACE_CODES]
