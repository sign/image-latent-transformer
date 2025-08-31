# UTF8Tokenizer

This module includes a **real** byte level tokenizer for text, which encodes text into a sequence of bytes (0-255).
Unlike `ByT5Tokenizer` for example, the tokenizer is implemented from scratch, and is much more efficient.

Other "Byte Level" tokenizers usually include various additional "special tokens" (e.g., `<pad>`, `<unk>`, etc.),
making the encoding and decoding logic more complex, and the token ids larger than 255.

Instead, we rely on C0 Control characters (0-31) as special tokens, which are not used in normal text.

## Special Tokens

We propose a specific usage of the C0 Control characters for "special tokens" in tokenization.
Other tokens can be left for task-specific usage, or future extensions.

- Padding tokens are using the <Null> character (`\x00`).
- Text is surrounded by <StartOfText> and <EndOfText> tokens (`\x02`...`\x03`).
- Messages for instruction tuning are surrounded by <StartOfHeading> and <EndOfTransmission> tokens (`\x01`system ...
  `\x17`).
- Attention blocks (PrefixLM / MAS) are surrounded by <ShiftOut> and <ShiftIn> tokens (`\xOE`...`\x0F`).
- Thinking blocks are surrounded by <Enquiry> and <Acknowledge> tokens (`\x05`...`\x06`).
- Tool calls are surrounded by <Substitute> and <Escape> tokens (`\x1A`...`\x1B`).

Example text for instruction tuning with attention blocks:

> `\x02\x01`system
> `\xOE` You are a helpful assistant `\x0F\x17`
> `\x01`user
> `\xOE` How much is 1+2? `\x0F\x17`
> `\x01`assistant
> First I'll think about it.
> `\x05`The user wants me to calculate, I should call the calculator
> `\x1A`{"type": "calculator", "expression": "1+2"}`\x1B`3`\x06`
> 1 + 2 = 3 `\x17\x03`
> `\x00\x00\x00\x00\x00\x00\x00`

| Dec | Hex | Abbr.    | Name                      | Description                                         | Tokenizer Usage       |
|-----|-----|----------|---------------------------|-----------------------------------------------------|-----------------------|
| 0   | 00  | NUL      | Null                      | Does nothing. Used for padding or blank paper tape. | Padding               |
| 1   | 01  | SOH      | Start of Heading          | First character of the heading of a message.        | Begin message block   |
| 2   | 02  | STX      | Start of Text             | Terminates the header and starts the message text.  | Beginning of String   |
| 3   | 03  | ETX      | End of Text               | Ends the message text, starts a footer.             | End of String         |
| 4   | 04  | EOT      | End of Transmission       | Ends the transmission of one or more messages.      |                       |
| 5   | 05  | ENQ      | Enquiry                   | Trigger a response at the receiving end.            | Begin thinking        |
| 6   | 06  | ACK      | Acknowledge               | Indication of successful receipt.                   | End thinking          |
| 7   | 07  | BEL      | Bell, Alert               | Call for attention from an operator.                |                       |
| 8   | 08  | BS       | Backspace                 | Move one position leftwards.                        |                       |
| 9   | 09  | HT       | Horizontal Tab            | Move right to next tab stop.                        | --Whitespace--        |
| 10  | 0A  | LF       | Line Feed                 | Move down to next line.                             | --Whitespace--        |
| 11  | 0B  | VT       | Vertical Tab              | Move down to next vertical tab stop.                | --Whitespace--        |
| 12  | 0C  | FF       | Form Feed                 | Move down to top of next page.                      | --Whitespace--        |
| 13  | 0D  | CR       | Carriage Return           | Move to column zero.                                | --Whitespace--        |
| 14  | 0E  | SO       | Shift Out                 | Switch to alternative character set.                | Begin attention block |
| 15  | 0F  | SI       | Shift In                  | Return to regular character set.                    | End attention block   |
| 16  | 10  | DLE      | Data Link Escape          | Interpret following characters differently.         |                       |
| 17  | 11  | DC1/XON  | Device Control 1          | Turn devices on/off, used in flow control.          |                       |
| 18  | 12  | DC2      | Device Control 2          | —                                                   |                       |
| 19  | 13  | DC3/XOFF | Device Control 3          | —                                                   |                       |
| 20  | 14  | DC4      | Device Control 4          | —                                                   |                       |
| 21  | 15  | NAK      | Negative Acknowledge      | Negative response (e.g., error).                    |                       |
| 22  | 16  | SYN      | Synchronous Idle          | Used when no other character is transmitted.        |                       |
| 23  | 17  | ETB      | End of Transmission Block | Marks end of block of data.                         | End of message block  |
| 24  | 18  | CAN      | Cancel                    | Data preceding it is invalid/disregarded.           |                       |
| 25  | 19  | EM       | End of Medium             | End of usable tape/media.                           |                       |
| 26  | 1A  | SUB      | Substitute                | Replace invalid/error character.                    | Begin tool calling    |
| 27  | 1B  | ESC      | Escape                    | Alters meaning of following bytes.                  | End tool calling      |
| 28  | 1C  | FS       | File Separator            | Used as delimiters for fields.                      |                       |
| 29  | 1D  | GS       | Group Separator           | —                                                   |                       |
| 30  | 1E  | RS       | Record Separator          | —                                                   |                       |
| 31  | 1F  | US       | Unit Separator            | —                                                   |                       |
| 32  | 20  | SP       | Space                     | Move right one position.                            | --Whitespace--        |
| 127 | 7F  | DEL      | Delete                    | Ignore; used to delete chars on punched tape.       |                       |