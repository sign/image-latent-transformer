class ControlTokens:
    Null = "\x00"
    StartOfHeading = "\x01"
    StartOfText = "\x02"
    EndOfText = "\x03"
    EndOfTransmission = "\x04"
    Enquiry = "\x05"
    Acknowledge = "\x06"
    Alert = "\x07"
    Backspace = "\x08"
    HorizontalTab = "\x09"  # Whitespace
    LineFeed = "\x0A"  # Whitespace
    VerticalTab = "\x0B"  # Whitespace
    FormFeed = "\x0C"  # Whitespace
    CarriageReturn = "\x0D"  # Whitespace
    ShiftOut = "\x0E"
    ShiftIn = "\x0F"
    DataLinkEscape = "\x10"
    DeviceControl1 = "\x11"
    DeviceControl2 = "\x12"
    DeviceControl3 = "\x13"
    DeviceControl4 = "\x14"
    NegativeAcknowledge = "\x15"
    SynchronousIdle = "\x16"
    EndOfTransmissionBlock = "\x17"
    Cancel = "\x18"
    EndOfMedium = "\x19"
    Substitute = "\x1A"
    Escape = "\x1B"
    FileSeparator = "\x1C"
    GroupSeparator = "\x1D"
    RecordSeparator = "\x1E"
    UnitSeparator = "\x1F"
    Space = "\x20"  # Whitespace
    Delete = "\x7F"


CONTROl_TOKENS_PATTERN = "\x01-\x08\x0E-\x1F\x7F"
