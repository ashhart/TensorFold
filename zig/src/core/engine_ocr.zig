//! An engine's OCR reader (``Info.ocr``): its prompt and the parse from generated text to Markdown blocks.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Size = @import("engine_vision.zig").Size;

/// The HTML table subset readers and the server share (ocr_html.zig).
pub const html = @import("ocr_html.zig");

/// TL, TR, BR, BL of the text in its upright reading frame, in original-image pixels (rotated text keeps its order).
pub const Quad = [4][2]f32;

/// What a block holds.
pub const Kind = enum {
    text,
    title,
    section_header,
    list,
    table,
    formula,
    code,
    caption,
    footnote,
    page_header,
    page_footer,
    picture,
};

/// One block of a page in reading order.
pub const OcrBlock = struct {
    content: []const u8, // normalized Markdown (docs/api.md OCR: Reader Markdown); empty only for .picture
    quad: Quad,
    kind: Kind,
};

/// The generated text does not follow the model's grammar.
pub const Error = error{Unreadable};

/// OCR an engine serves: the prompt after the page image, and the reader of the generated text.
pub const Ocr = struct {
    /// Text parts that follow the image in the single user turn (may be empty).
    prompt: []const u8,
    /// Text to blocks in pixels of ``original``; ``frame`` is the tower input size the prepare step used.
    parse: *const fn (a: Allocator, text: []const u8, original: Size, frame: Size) anyerror![]OcrBlock,
};

test {
    _ = @import("ocr_html_test.zig");
}
