//! One bounded device-local allocation a caller suballocates into aligned, checked tensor views.
//! An arena must not move once a view exists: views point back at its buffer field.
const std = @import("std");
const memory = @import("memory.zig");
const tensor = @import("tensor.zig");
const Device = @import("device.zig").Device;
const Buffer = memory.Buffer;
const Error = @import("runtime.zig").Error;

pub const Arena = struct {
    buffer: Buffer,
    /// Next free byte; only grows through alloc and shrinks through reset.
    next: u64,
    /// Every allocation's alignment: the device's storage offset alignment, at least a word.
    alignment: u64,

    /// A device-local arena of `bytes`; its size must fit the device's buffer and allocation limits.
    pub fn init(dev: *const Device, bytes: u64) Error!Arena {
        try memory.fits(bytes, dev.v13.maxBufferSize, dev.v11.maxMemoryAllocationSize);
        return .{
            .buffer = try Buffer.init(dev, .device, bytes),
            .next = 0,
            .alignment = @max(dev.properties.limits.minStorageBufferOffsetAlignment, 4),
        };
    }

    /// Frees the arena's buffer; call only after the last submission using any of its views returned success.
    pub fn deinit(a: *Arena) void {
        a.buffer.deinit();
    }

    /// An aligned, validated view of `elements` of `kind`, or the named reason it cannot fit; the
    /// aligned cursor only advances once the whole allocation fits, so a failure wastes nothing.
    pub fn alloc(a: *Arena, kind: tensor.Kind, elements: u64) Error!tensor.View {
        if (elements == 0) return error.Invalid;
        const start = std.mem.alignForward(u64, a.next, a.alignment);
        const v = tensor.View.init(&a.buffer, kind, start, elements) catch |e| return if (e == error.OutOfRange) error.OutOfMemory else e;
        const end = std.math.add(u64, v.byte_offset, v.byteLen()) catch return error.OutOfMemory;
        // Round the committed cursor up so every mark/reset boundary is an allocation boundary.
        a.next = (std.math.add(u64, end, a.alignment - 1) catch return error.OutOfMemory) / a.alignment * a.alignment;
        return v;
    }

    /// The cursor to reset to; taken before the allocations it unwinds.
    pub fn mark(a: *const Arena) u64 {
        return a.next;
    }

    /// Winds the cursor back to an earlier mark; call only after the serial commands using those views completed.
    pub fn reset(a: *Arena, to: u64) Error!void {
        if (to > a.next) return error.OutOfRange;
        if (to % a.alignment != 0) return error.Misaligned;
        a.next = to;
    }

    pub fn used(a: *const Arena) u64 {
        return a.next;
    }

    pub fn capacity(a: *const Arena) u64 {
        return a.buffer.len;
    }
};

const testing = std.testing;
const tensorKind = tensor.Kind;

fn arena(len: u64, alignment: u64) Arena {
    return .{ .buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = len, .kind = .device, .mapped = null, .coherent = false }, .next = 0, .alignment = alignment };
}

test "allocations are aligned, bounded and hold their geometry" {
    var a = arena(1024, 256);
    const f = try a.alloc(.f32, 4); // 16 bytes, cursor committed at the next 256 boundary
    try testing.expectEqual(@as(u64, 0), f.byte_offset);
    const b = try a.alloc(tensorKind.bf16, 130); // 260 bytes from 256, cursor rounds to 768
    try testing.expectEqual(@as(u64, 256), b.byte_offset);
    try testing.expectEqual(@as(u64, 768), a.used());
    try testing.expectError(error.OutOfMemory, a.alloc(.u32, 65)); // 260 bytes past 768 cannot round-trip 1024
    try testing.expectEqual(@as(u64, 768), a.used()); // the failed allocation left the cursor alone
    try testing.expect(f.buffer == &a.buffer);
}

test "the arena never returns empty or ragged views" {
    var a = arena(512, 64);
    try testing.expectError(error.Invalid, a.alloc(.u32, 0));
    try testing.expectError(error.Misaligned, a.alloc(.bf16, 1)); // a two-byte tail drops its element
    try testing.expectEqual(@as(u64, 0), a.used());
    const ok = try a.alloc(.bf16, 2); // whole words only
    try testing.expectEqual(@as(u64, 4), ok.byteLen());
}

test "reset returns to an earlier mark and refuses later or misaligned marks" {
    var a = arena(512, 64);
    const first = try a.alloc(.u32, 8); // bytes 0..32, cursor committed at 64
    try testing.expectEqual(@as(u64, 0), first.byte_offset);
    const m = a.mark();
    try testing.expectEqual(@as(u64, 64), m);
    _ = try a.alloc(.u32, 8); // bytes 64..96
    try a.reset(m);
    try testing.expectEqual(@as(u64, m), a.used());
    const again = try a.alloc(.u32, 8); // lands where the reset-away allocation was
    try testing.expectEqual(@as(u64, 64), again.byte_offset);
    try testing.expectError(error.OutOfRange, a.reset(a.used() + 64));
    try testing.expectError(error.Misaligned, a.reset(17));
}

test "a zero-capacity arena allocates nothing" {
    var a = arena(0, 16);
    try testing.expectError(error.OutOfMemory, a.alloc(.f32, 1));
}
