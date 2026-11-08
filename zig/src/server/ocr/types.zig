//! The OCR routes' shared types: a page's blocks, its plain-text lines, the request's sources and its one fault.
const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const errors = @import("../errors.zig");

/// One page of a request: its index in the source document, the pixel size its quads refer to, and its blocks.
pub const OcrPage = struct {
    index: u32, // 0-based page index in the source document
    width: u32,
    height: u32,
    dpi: u32, // render DPI of a PDF page; 72 for an image
    blocks: []api.ocr.OcrBlock, // validated, reading order
};

/// One plain-text line of a page, never empty; ``block`` is the index of its source block.
pub const OcrLine = struct {
    text: []const u8,
    quad: ?api.ocr.Quad, // null only when the reader's geometry is .none
    block: u32,
    score: f32 = 1.0, // 0..1; the MVP readers give no scores
};

/// How a failed request ends; each dialect gives every class its own status and body (docs/api.md OCR: Failures).
pub const Class = enum { bad_input, unreadable, capacity, internal };

/// The error every OCR step returns; the details are in the request's ``*Fault``.
pub const Faulted = error{Faulted};

/// Why a request failed; ``message`` and ``loc`` live in the request arena, never in a page arena.
pub const Fault = struct {
    class: Class,
    message: []const u8,
    loc: []const []const u8 = &.{},
    retry_after: ?u32 = null,

    /// Fills ``fault`` with ``class`` and a message formatted into ``a`` (the request arena); returns error.Faulted.
    pub fn fail(fault: *Fault, a: Allocator, class: Class, comptime fmt: []const u8, args: anytype) (Faulted || Allocator.Error) {
        fault.* = .{ .class = class, .message = try std.fmt.allocPrint(a, fmt, args) };
        return error.Faulted;
    }

    /// Copies ``cx``'s refusal into ``fault``, its message duplicated into ``a`` (the request arena).
    pub fn fromCx(a: Allocator, cx: *const errors.Cx, fault: *Fault) (Faulted || Allocator.Error) {
        const class: Class = switch (cx.kind) {
            .request, .context_length, .unimplemented => .bad_input,
            .capacity => .capacity,
            .other, .server => .internal,
        };
        fault.* = .{ .class = class, .message = try a.dupe(u8, cx.message), .retry_after = if (class == .capacity) 1 else null };
        return error.Faulted;
    }
};

/// One document of a request: a page image.
pub const Source = struct {
    name: []const u8,
    bytes: []const u8,
};
