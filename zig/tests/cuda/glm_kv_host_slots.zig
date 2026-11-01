//! glm-kv-host-slots (GPU): TF_GLM_KV_HOST's gather, past its slots and within them, through the engine's check.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const kv_host = @import("glm_cuda").kv_host;

const arena_rows = 1024;
const row_bytes = 64;
const cap = 64; // slots: the first call picks 256 distinct rows, the second 48
const rows = 32; // a call past the row cache's 16, so mark, copy, remap and reset
const width = 8;

/// Arena row `r`'s bytes: words naming the row and the word, so a row in the wrong slot shows.
fn word(r: usize, i: usize) u32 {
    return @intCast(r * 1000 + i);
}

/// One gather of `picks` (rows x width, every count full); returns the token lists it remapped to slots.
fn call(gpu: check.Gpu, ops: cuda.Module, s: cuda.Stream, lc: u64, picks: []const i32) ![]i32 {
    var tokens = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer tokens.free();
    const full: [rows]i32 = @splat(width);
    var counts = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&full));
    defer counts.free();
    try gpu.d.synchronize(); // the uploads, before the non-blocking stream
    _ = try kv_host.active.?.gather(ops, s, 1, lc, 0, 0, 0, tokens.ptr, counts.ptr, rows, width);
    try s.synchronize();
    const out = try gpu.gpa.alloc(i32, picks.len);
    try tokens.download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// No arena row left marked: every slot_of entry back at -1.
fn unmarked(gpu: check.Gpu) !bool {
    const bytes = try check.download(gpu, kv_host.active.?.slot_of);
    defer gpu.gpa.free(bytes);
    return std.mem.allEqual(u8, bytes, 0xff);
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    var host: [arena_rows * row_bytes / 4]u32 = undefined;
    for (&host, 0..) |*w, i| w.* = word(i / (row_bytes / 4), i % (row_bytes / 4));
    var lc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&host));
    defer lc.free();
    kv_host.active = try kv_host.Gather.init(gpu.d, arena_rows, cap, row_bytes);
    defer {
        kv_host.active.?.deinit();
        kv_host.active = null;
    }

    var past: [rows * width]i32 = undefined;
    for (&past, 0..) |*t, i| t.* = @intCast(i * 37 % arena_rows); // 256 distinct rows
    gpu.gpa.free(try call(gpu, ops, s, lc.ptr, &past));
    kv_host.checkActive() catch |err| {
        try check.expect(err == error.GlmKvHostSlots, "kv host slots: past the slots gave {s}", .{@errorName(err)});
        try check.expect(try unmarked(gpu), "kv host slots: a row past the slots was left marked", .{});
        try kv_host.checkActive(); // the flag is cleared: the next call is judged alone
        check.pass("kv host slots: 256 distinct rows for {d} slots refused as error.{s}, no row left marked", .{ cap, @errorName(err) });
        return within(gpu, ops, s, lc.ptr, &host);
    };
    try check.expect(false, "kv host slots: 256 distinct rows for {d} slots were not refused", .{cap});
}

/// A call within the slots after the refused one: accepted, and each token's slot holds its row's bytes.
fn within(gpu: check.Gpu, ops: cuda.Module, s: cuda.Stream, lc: u64, host: []const u32) !void {
    var picks: [rows * width]i32 = undefined;
    for (&picks, 0..) |*t, i| t.* = @intCast(i % 48 * 5); // 48 distinct rows, each picked more than once
    const slots_of = try call(gpu, ops, s, lc, &picks);
    defer gpu.gpa.free(slots_of);
    try kv_host.checkActive();
    try check.expect(try unmarked(gpu), "kv host slots: a call within the slots left a row marked", .{});
    const slots = try check.download(gpu, kv_host.active.?.slots);
    defer gpu.gpa.free(slots);
    const arena = std.mem.sliceAsBytes(host);
    for (picks, slots_of) |r, slot| {
        try check.expect(slot >= 0 and slot < 48, "kv host slots: row {d} got slot {d}", .{ r, slot });
        const at: usize = @intCast(slot);
        const row: usize = @intCast(r);
        try check.sameBytes("kv host slots: a slot's row", slots[at * row_bytes ..][0..row_bytes], arena[row * row_bytes ..][0..row_bytes]);
    }
    check.pass("kv host slots: then 48 distinct rows for {d} slots accepted, each token's slot holding its row", .{cap});
}
