//! CPU ownership and exact-word copy tests for the shared prompt tap ring; no driver or CUDA work.
const std = @import("std");
const cuda = @import("cuda");
const taps = @import("cuda_prompt_taps.zig");
const t = std.testing;
const rows = 2112;
const width = 4;
const shared_ptr = 0x100000000;
const private_ptr = 0x200000000;
const poison: u16 = 0xa5a5;

const Cache = struct {
    tap: ?cuda.DeviceBuffer,
    len: usize = 0,
    tap_epoch: u64 = 1,
    tap_begin: u64 = 0,
    tap_end: u64 = 0,
};
const Part = struct { cache: *Cache, rows: usize };
const Seg = struct { tap: u64, tap_rows: c_int, len: c_int, rows: c_int };
const Fake = struct {
    n_taps: usize = 2,
    tap_capacity: usize = taps.private_rows,
    stream: struct { handle: cuda.abi.Stream = null } = .{},
    g: struct { hidden: usize = 2 } = .{},
    prompt_taps: taps.State = .{ .buffer = buffer(shared_ptr, rows * width * 2), .rows = rows, .row_bytes = width * 2 },
    shared: [rows * width]u16 = @splat(poison),
    local: [2][taps.private_rows * width]u16 = @splat(@splat(poison)),
    copies: usize = 0,
    fail_copy: bool = false,
};
fn buffer(ptr: u64, bytes: usize) cuda.DeviceBuffer {
    return .{ .d = undefined, .ptr = ptr, .len = bytes, .borrowed = true };
}
fn cache(index: usize) Cache {
    return .{ .tap = buffer(private_ptr + index * 0x100000, taps.private_rows * width * 2) };
}
fn segment(c: Cache, n: usize) Seg {
    return .{ .tap = c.tap.?.ptr, .tap_rows = taps.private_rows, .len = @intCast(c.len), .rows = @intCast(n) };
}
fn word(position: u64, column: usize, seed: u16) u16 {
    return @as(u16, @truncate(position *% 17 +% column *% 511 +% seed *% 99)) ^ 0x7fc0;
}
const Copy = struct {
    pub fn copy(e: *Fake, dst: cuda.DeviceBuffer, dst_at: usize, src: u64, bytes: usize) !void {
        if (e.fail_copy) return error.CopyFailed;
        const which = (dst.ptr - private_ptr) / 0x100000;
        @memcpy(e.local[which][dst_at / 2 ..][0 .. bytes / 2], e.shared[(src - shared_ptr) / 2 ..][0 .. bytes / 2]);
        e.copies += 1;
    }
};
fn write(e: *Fake, sg: Seg, seed: u16) void {
    const n: usize = @intCast(sg.rows);
    const cap: usize = @intCast(sg.tap_rows);
    // Serial absolute-position reference: earlier colliding rows are overwritten, so only the final bytes matter.
    for (0..n) |r| for (0..width) |col| {
        const position: u64 = @as(u64, @intCast(sg.len)) + r;
        const offset = position % cap * width + col;
        if (sg.tap == shared_ptr) e.shared[offset] = word(position, col, seed) else e.local[(sg.tap - private_ptr) / 0x100000][offset] = word(position, col, seed);
    };
}
fn forward(e: *Fake, c: *Cache, n: usize, prompt: bool, seed: u16) !void {
    const parts = [_]Part{.{ .cache = c, .rows = n }};
    var segs = [_]Seg{segment(c.*, n)};
    try taps.begin(e, &parts, &segs, prompt);
    errdefer taps.abort(e, &parts);
    write(e, segs[0], seed);
    try taps.publishWith(e, &parts, prompt, Copy);
    c.len += n;
}

const Allocation = struct {
    var fault: enum { none, allocate, zero } = .none;
    var freed: usize = 0;
    pub fn allocate(_: *Fake, bytes: usize) !cuda.DeviceBuffer {
        if (fault == .allocate) return error.OutOfMemory;
        return buffer(shared_ptr, bytes);
    }
    pub fn zero(_: *Fake, _: cuda.DeviceBuffer) !void {
        if (fault == .zero) return error.ZeroFailed;
    }
    pub fn discard(_: *Fake, _: *cuda.DeviceBuffer) void {
        freed += 1;
    }
};

test "Qwen shared prompt taps: setup is atomic on allocation and zero failure, disabled state untouched" {
    for ([_]@TypeOf(Allocation.fault){ .allocate, .zero, .none }) |fault| {
        var e: Fake = .{ .tap_capacity = rows, .prompt_taps = .{} };
        Allocation.fault = fault;
        Allocation.freed = 0;
        const result = taps.configureWith(&e, Allocation);
        if (fault == .none) {
            try result;
            try t.expectEqual(@as(usize, 65), e.tap_capacity);
            try t.expectEqual(@as(usize, rows * width * 2), e.prompt_taps.buffer.?.len);
            try t.expectError(error.BadPromptTapState, taps.configureWith(&e, Allocation));
        } else {
            if (fault == .allocate) try t.expectError(error.OutOfMemory, result) else try t.expectError(error.ZeroFailed, result);
            try t.expectEqual(@as(usize, rows), e.tap_capacity);
            try t.expect(e.prompt_taps.buffer == null);
        }
        try t.expectEqual(@as(usize, @intFromBool(fault == .zero)), Allocation.freed);
    }
    var disabled: Fake = .{ .n_taps = 0, .tap_capacity = 2112, .prompt_taps = undefined };
    var c = cache(0);
    const parts = [_]Part{.{ .cache = &c, .rows = 1 }};
    var segs = [_]Seg{segment(c, 1)};
    try taps.begin(&disabled, &parts, &segs, true);
    try taps.publish(&disabled, &parts, true);
    taps.abort(&disabled, &parts);
    try t.expectEqual(@as(?u64, null), try taps.read(&disabled, &c, 0, 1, 2048));
}

test "Qwen shared prompt taps: prompt65 retains all 64 rows before the excluded final row" {
    var e: Fake = .{};
    var c = cache(0);
    try forward(&e, &c, 65, true, 1);
    try t.expectEqual(@as(?u64, c.tap.?.ptr), try taps.read(&e, &c, 0, 64, 2048));
    for (0..65) |r| for (0..width) |col| try t.expectEqual(word(r, col, 1), e.local[0][r * width + col]);
    try t.expectEqual(@as(usize, 65), taps.modulus(&e, c.tap.?.ptr));
    try t.expectEqual(@as(usize, 2112), taps.modulus(&e, shared_ptr));
}

test "Qwen shared prompt taps: large prompt and interleaved lanes preserve exact private words" {
    var e: Fake = .{};
    var a = cache(0);
    var b = cache(1);
    try forward(&e, &a, 8192, true, 3);
    try t.expectEqual(@as(?u64, shared_ptr), try taps.read(&e, &a, 0, 8192, 2048));
    try t.expectEqual(@as(?u64, a.tap.?.ptr), try taps.read(&e, &a, 8191, 1, 2048));
    try t.expectError(error.RowsNotHeld, taps.read(&e, &a, 0, 128, 2048));
    const saved = e.local[0];
    try forward(&e, &b, 8192, true, 9);
    try t.expectEqualDeep(saved, e.local[0]);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &a, 0, 8192, 2048));
    try t.expectEqual(@as(?u64, a.tap.?.ptr), try taps.read(&e, &a, 8191, 1, 2048));
    for (8127..8192) |r| for (0..width) |col| {
        try t.expectEqual(word(r, col, 3), e.local[0][r % 65 * width + col]);
        try t.expectEqual(word(r, col, 9), e.local[1][r % 65 * width + col]);
    };
    taps.invalidate(&e, &b);
    try t.expect(e.prompt_taps.held == null);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &b, 8191, 1, 2048));
}

test "Qwen shared prompt taps: epoch, rollback and incomplete writes refuse stale intervals" {
    var e: Fake = .{};
    var c = cache(0);
    try forward(&e, &c, 8192, true, 3);
    c.tap_epoch += 1;
    try t.expectError(error.RowsNotHeld, taps.read(&e, &c, 0, 8192, 2048));
    taps.reset(&c);
    try forward(&e, &c, 64, false, 5);
    try t.expectEqual(@as(?u64, c.tap.?.ptr), try taps.read(&e, &c, 8192, 64, 2048));
    c.len = 8193; // only the first row of the previous verify was accepted
    try forward(&e, &c, 1, false, 7);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &c, 8130, 64, 2048));
    try t.expectError(error.RowsNotHeld, taps.read(&e, &c, 8192, 1, 2048));
    try t.expectEqual(@as(?u64, c.tap.?.ptr), try taps.read(&e, &c, 8193, 1, 2048));
    const parts = [_]Part{.{ .cache = &c, .rows = 8192 }};
    var segs = [_]Seg{segment(c, 8192)};
    try taps.begin(&e, &parts, &segs, true);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &c, 8193, 1, 2048));
    e.fail_copy = true;
    try t.expectError(error.CopyFailed, taps.publishWith(&e, &parts, true, Copy));
    taps.abort(&e, &parts);
    try t.expect(!e.prompt_taps.active and e.prompt_taps.held == null);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &c, 8193, 1, 2048));
}

test "Qwen shared prompt taps: malformed and aliased buffers fail before beginning a write" {
    var e: Fake = .{};
    var c = cache(0);
    const parts = [_]Part{.{ .cache = &c, .rows = 64 }};
    var segs = [_]Seg{segment(c, 64)};
    c.tap.?.len = 1;
    try t.expectError(error.BadPromptTapBuffer, taps.begin(&e, &parts, &segs, true));
    c = cache(0);
    c.tap.?.ptr = shared_ptr;
    try t.expectError(error.BadPromptTapBuffer, taps.begin(&e, &parts, &segs, true));
    try t.expect(!e.prompt_taps.active);
    c = cache(0);
    const twice = [_]Part{ parts[0], parts[0] };
    var both = [_]Seg{ segs[0], segs[0] };
    try t.expectError(error.BadPromptTapState, taps.begin(&e, &twice, &both, false));
    try t.expect(!e.prompt_taps.active);
}

test "Qwen shared prompt taps: an error after publish invalidates only participating caches" {
    var e: Fake = .{};
    var a = cache(0);
    var b = cache(1);
    try forward(&e, &a, 8192, true, 3);
    const parts = [_]Part{.{ .cache = &a, .rows = 8192 }};
    try t.expect(!e.prompt_taps.active);
    // Engine.forward's later head/readback failed after the taps had already been published.
    taps.abort(&e, &parts);
    try t.expect(e.prompt_taps.held == null);
    try t.expectError(error.RowsNotHeld, taps.read(&e, &a, 8191, 1, 2048));
    try t.expectError(error.RowsNotHeld, taps.read(&e, &a, 0, 8192, 2048));
    try forward(&e, &b, 8192, true, 5);
    try forward(&e, &a, 64, false, 7);
    taps.abort(&e, &parts);
    try t.expectEqual(@as(?u64, shared_ptr), try taps.read(&e, &b, 0, 8192, 2048));
    try t.expectError(error.RowsNotHeld, taps.read(&e, &a, 8192, 64, 2048));
}
