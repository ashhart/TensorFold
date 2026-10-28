//! Honest device tensor views: a byte range of one buffer with a fixed element kind, checked before any shader sees it.
const std = @import("std");
const memory = @import("memory.zig");
const dispatch = @import("dispatch.zig");
const Buffer = memory.Buffer;
const Error = @import("runtime.zig").Error;

/// The element kinds stage 1 moves and computes; bf16 rides storage as raw uint16 bits.
pub const Kind = enum {
    f32,
    bf16,
    u32,
    u16,

    /// Bytes per element; every kind is 2 or 4, so 4-aligned byte offsets are element-aligned.
    pub fn size(kind: Kind) u64 {
        return switch (kind) {
            .f32, .u32 => 4,
            .bf16, .u16 => 2,
        };
    }
};

/// `elements` of `kind` in bytes, refusing overflow.
pub fn extent(kind: Kind, elements: u64) Error!u64 {
    const bytes = elements *| kind.size();
    if (elements != 0 and bytes / kind.size() != elements) return error.TooLarge;
    return bytes;
}

/// A checked `{ byte_offset, elements, kind }` window over one buffer; never a raw device address.
pub const View = struct {
    buffer: *const Buffer,
    byte_offset: u64,
    elements: u64,
    kind: Kind,

    /// A nonempty, 4-byte-aligned, word-padded window inside `buffer`; bf16/u16 windows hold whole words.
    pub fn init(buffer: *const Buffer, kind: Kind, byte_offset: u64, elements: u64) Error!View {
        if (elements == 0) return error.OutOfRange;
        if (byte_offset % 4 != 0) return error.Misaligned;
        const bytes = try extent(kind, elements);
        try memory.span(buffer.len, byte_offset, bytes);
        if (bytes % 4 != 0) return error.Misaligned; // std430 word transport: a ragged tail drops its last element
        return .{ .buffer = buffer, .byte_offset = byte_offset, .elements = elements, .kind = kind };
    }

    pub fn byteLen(v: View) u64 {
        return v.elements * v.kind.size(); // init proved this fits and is word-padded
    }

    /// A subwindow starting `first` elements in; it cannot escape the window it slices, only the buffer.
    pub fn slice(v: View, first: u64, elements: u64) Error!View {
        if (first > v.elements or elements > v.elements - first) return error.OutOfRange;
        return init(v.buffer, v.kind, v.byte_offset + first * v.kind.size(), elements);
    }

    /// The window's byte offset as an element index, as the shaders' push constants carry offsets.
    pub fn elemOffset32(v: View) Error!u32 {
        return std.math.cast(u32, v.byte_offset / v.kind.size()) orelse error.TooLarge;
    }

    pub fn elemCount32(v: View) Error!u32 {
        return std.math.cast(u32, v.elements) orelse error.TooLarge;
    }

    /// The descriptor range this window binds: its own base, so shader element offsets start at zero.
    pub fn bind(v: View) dispatch.Binding {
        return .{ .buffer = v.buffer, .offset = v.byte_offset, .len = v.byteLen() };
    }

    /// Whether two windows overlap (same buffer only); used to refuse aliasing before Pipeline.bind does.
    pub fn overlaps(a: View, b: View) bool {
        return a.buffer == b.buffer and a.byte_offset < b.byte_offset + b.byteLen() and b.byte_offset < a.byte_offset + a.byteLen();
    }
};

const testing = std.testing;

fn fakeBuffer(len: u64) Buffer {
    return .{ .dev = undefined, .handle = null, .memory = null, .len = len, .kind = .device, .mapped = null, .coherent = false };
}

test "views hold their geometry and refuse empty, misaligned and out-of-range windows" {
    const buf = fakeBuffer(1024);
    const v = try View.init(&buf, .f32, 16, 128);
    try testing.expectEqual(@as(u64, 16), v.byte_offset);
    try testing.expectEqual(@as(u64, 512), v.byteLen());
    try testing.expectEqual(@as(u32, 4), try v.elemOffset32());
    try testing.expectEqual(@as(u32, 128), try v.elemCount32());
    try testing.expectError(error.OutOfRange, View.init(&buf, .f32, 0, 0));
    try testing.expectError(error.Misaligned, View.init(&buf, .f32, 2, 8));
    try testing.expectError(error.OutOfRange, View.init(&buf, .f32, 512, 129));
    try testing.expectError(error.OutOfRange, View.init(&buf, .f32, 1024, 1));
    try testing.expectError(error.TooLarge, extent(.bf16, std.math.maxInt(u64) / 2 + 1));
}

test "bf16 and u16 windows must end on a word boundary" {
    const buf = fakeBuffer(64);
    try testing.expectError(error.Misaligned, View.init(&buf, .bf16, 0, 3));
    _ = try View.init(&buf, .bf16, 8, 4);
    try testing.expectError(error.Misaligned, View.init(&buf, .u16, 0, 1));
}

test "slices stay inside their window and keep the base offset" {
    const buf = fakeBuffer(256);
    const v = try View.init(&buf, .u32, 32, 32);
    const s = try v.slice(4, 8);
    try testing.expectEqual(@as(u64, 48), s.byte_offset);
    try testing.expectEqual(@as(u64, 8), s.elements);
    try testing.expectError(error.OutOfRange, v.slice(33, 1));
    try testing.expectError(error.OutOfRange, v.slice(0, 33));
    try testing.expectError(error.OutOfRange, v.slice(31, 2)); // may not escape past the window's end
    try testing.expectError(error.OutOfRange, v.slice(5, std.math.maxInt(u64)));
}

test "element offsets past 32 bits refuse instead of wrapping" {
    const buf = fakeBuffer((1 << 34) + 64);
    const v = try View.init(&buf, .u32, 4, (1 << 32) + 1);
    try testing.expectError(error.TooLarge, v.elemCount32());
    try testing.expectEqual(@as(u32, 1), try v.elemOffset32());
}

test "overlap is same-buffer range intersection" {
    const buf = fakeBuffer(256);
    const other = fakeBuffer(256);
    const a = try View.init(&buf, .u32, 0, 16);
    const b = try View.init(&buf, .u32, 64, 16);
    const c = try View.init(&buf, .u32, 60, 4);
    const d = try View.init(&other, .u32, 0, 16);
    try testing.expect(a.overlaps(c));
    try testing.expect(!a.overlaps(b));
    try testing.expect(!a.overlaps(d));
}
