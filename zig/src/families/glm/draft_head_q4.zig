//! Opt-in 4-bit/64 MTP draft head quantized in memory at load from an 8-bit/64 GLM lm_head.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const affine = @import("affine.zig");
const affine4 = @import("../../core/affine4.zig");

const max_workers = 16;

/// Parse `GLM_DRAFT_HEAD_Q4`: unset, blank or `0` is off; `1` enables in-memory 4-bit draft head synthesis.
pub fn enabled(raw: ?[]const u8) !bool {
    return platform.environment.value(bool, "GLM_DRAFT_HEAD_Q4", raw, false);
}

/// Read `GLM_DRAFT_HEAD_Q4` from the process environment (default off).
pub fn fromEnvironment() !bool {
    return platform.environment.read(bool, "GLM_DRAFT_HEAD_Q4", false);
}

/// Read `GLM_DRAFT_HEAD_Q4` first so malformed text fails closed, then synthesize `w.draft_head` when active.
pub fn maybeMakeFromEnv(gpa: std.mem.Allocator, device: mtl.Device, w: *wts.Weights, c: *const cfg.Config, external_drafter: bool, used: usize, limit: usize) !void {
    const on = try fromEnvironment();
    return maybeMake(gpa, device, w, c, on, external_drafter, used, limit);
}

/// Aligned shared-buffer byte size required to pack an [n, k] matrix at 4-bit/64.
pub fn bytesNeeded(n: u32, k: u32) usize {
    const fmt: affine.Width = .{ .bits = 4, .group = 64 };
    const wb = affine.wBytes(n, k, fmt);
    const sb = affine.sBytes(n, k, fmt);
    const s_off = std.mem.alignForward(usize, wb, 256);
    const b_off = std.mem.alignForward(usize, s_off + sb, 256);
    return std.mem.alignForward(usize, b_off + sb, 256);
}

/// Build a 4-bit/64 `w.draft_head` from an 8-bit/64 `w.head` when opted in, `--drafter` is absent, and memory fits.
pub fn maybeMake(gpa: std.mem.Allocator, device: mtl.Device, w: *wts.Weights, c: *const cfg.Config, on: bool, external_drafter: bool, used: usize, limit: usize) !void {
    if (!on or external_drafter or c.tp != 1 or w.mtp == null or w.draft_head != null) return;
    const h = w.head;
    if (h.more != 0 or h.bits != 8 or h.group != 64 or h.n == 0 or h.k == 0 or h.k % 64 != 0) return;
    const fmt: affine.Width = .{ .bits = 4, .group = 64 };
    const wb = affine.wBytes(h.n, h.k, fmt);
    const sb = affine.sBytes(h.n, h.k, fmt);
    const s_off = std.mem.alignForward(usize, wb, 256);
    const b_off = std.mem.alignForward(usize, s_off + sb, 256);
    const need = bytesNeeded(h.n, h.k);
    if (used + need > limit) return;
    const buf = try device.buffer(need, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    w.buffers.append(gpa, buf) catch |err| {
        buf.deinit();
        return err;
    };
    w.bytes += need;
    const ng = @as(usize, h.k) / 64;
    const groups = @as(usize, h.n) * ng;
    const dst_w: [*]u32 = @ptrCast(@alignCast(buf.contents()));
    const dst_s: [*]u16 = @ptrCast(@alignCast(buf.contents() + s_off));
    const dst_b: [*]u16 = @ptrCast(@alignCast(buf.contents() + b_off));
    const src_w: [*]const u8 = @ptrCast(h.w.addr());
    const src_s: [*]const u16 = @ptrCast(@alignCast(h.s.addr()));
    const src_b: [*]const u16 = @ptrCast(@alignCast(h.b.addr()));
    try quantizeParallel(src_w, src_s, src_b, dst_w, dst_s, dst_b, groups);
    w.draft_head = .{
        .w = .{ .buf = buf, .off = 0 },
        .s = .{ .buf = buf, .off = s_off },
        .b = .{ .buf = buf, .off = b_off },
        .n = h.n,
        .k = h.k,
        .bits = 4,
        .group = 64,
    };
}

fn quantizeParallel(src_w: [*]const u8, src_s: [*]const u16, src_b: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, groups: usize) error{ NonFiniteAffineInput, NonFiniteAffineRange }!void {
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, groups / 4096));
    const per = (groups + workers - 1) / workers;
    var results: [max_workers]error{ NonFiniteAffineInput, NonFiniteAffineRange }!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |t| threads[t] = std.Thread.spawn(.{}, runRange, .{ &results[t], src_w, src_s, src_b, dst_w, dst_s, dst_b, t * per, @min(groups, (t + 1) * per) }) catch null;
    runRange(&results[0], src_w, src_s, src_b, dst_w, dst_s, dst_b, 0, @min(groups, per));
    for (1..workers) |t| if (threads[t]) |th| th.join() else runRange(&results[t], src_w, src_s, src_b, dst_w, dst_s, dst_b, t * per, @min(groups, (t + 1) * per));
    for (results[0..workers]) |r| try r;
}

fn runRange(out: *error{ NonFiniteAffineInput, NonFiniteAffineRange }!void, src_w: [*]const u8, src_s: [*]const u16, src_b: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, first: usize, last: usize) void {
    out.* = quantizeGroups(src_w, src_s, src_b, dst_w, dst_s, dst_b, first, last);
}

/// Dequantize each 8/64 group to fp32 on stack and repack to exact 4/64 affine words and scales.
fn quantizeGroups(src_w: [*]const u8, src_s: [*]const u16, src_b: [*]const u16, dst_w: [*]u32, dst_s: [*]u16, dst_b: [*]u16, first: usize, last: usize) error{ NonFiniteAffineInput, NonFiniteAffineRange }!void {
    for (first..last) |g| {
        const s8 = affine4.asFloat(src_s[g]);
        const b8 = affine4.asFloat(src_b[g]);
        if (!std.math.isFinite(s8) or !std.math.isFinite(b8)) return error.NonFiniteAffineInput;
        var vals: [64]f32 = undefined;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = -std.math.inf(f32);
        const q8 = src_w + g * 64;
        for (0..64) |i| {
            const v = s8 * @as(f32, @floatFromInt(q8[i])) + b8;
            vals[i] = v;
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        dst_s[g] = affine4.asBf16(scale);
        dst_b[g] = affine4.asBf16(bias);
        const w = dst_w[g * 8 ..][0..8];
        @memset(w, 0);
        for (vals, 0..) |v, idx| {
            const code: u32 = @intFromFloat(@min(@as(f32, 15), @max(@as(f32, 0), @round((v - bias) / scale))));
            w[idx / 8] |= code << @as(u5, @intCast(4 * (idx % 8)));
        }
    }
}

test "quantizeGroups packs 4-bit codes within 8-bit group range and parallel matches serial" {
    const a = std.testing.allocator;
    const ng = 2;
    const rows = 4096;
    const groups = rows * ng;
    const sw = try a.alloc(u8, groups * 64);
    defer a.free(sw);
    const ss = try a.alloc(u16, groups);
    defer a.free(ss);
    const sb = try a.alloc(u16, groups);
    defer a.free(sb);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..groups) |g| {
        ss[g] = affine4.asBf16(0.01);
        sb[g] = affine4.asBf16(-1.27);
        for (0..64) |i| sw[g * 64 + i] = prng.random().int(u8);
    }
    const dw0 = try a.alloc(u32, groups * 8);
    defer a.free(dw0);
    const ds0 = try a.alloc(u16, groups);
    defer a.free(ds0);
    const db0 = try a.alloc(u16, groups);
    defer a.free(db0);
    const dw1 = try a.alloc(u32, groups * 8);
    defer a.free(dw1);
    const ds1 = try a.alloc(u16, groups);
    defer a.free(ds1);
    const db1 = try a.alloc(u16, groups);
    defer a.free(db1);
    try quantizeGroups(sw.ptr, ss.ptr, sb.ptr, dw0.ptr, ds0.ptr, db0.ptr, 0, groups);
    try quantizeParallel(sw.ptr, ss.ptr, sb.ptr, dw1.ptr, ds1.ptr, db1.ptr, groups);
    try std.testing.expectEqualSlices(u32, dw0, dw1);
    try std.testing.expectEqualSlices(u16, ds0, ds1);
    try std.testing.expectEqualSlices(u16, db0, db1);
    for (0..64) |i| {
        const want = affine4.asFloat(ss[0]) * @as(f32, @floatFromInt(sw[i])) + affine4.asFloat(sb[0]);
        const code = (dw0[i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 0xf;
        const got = affine4.asFloat(ds0[0]) * @as(f32, @floatFromInt(code)) + affine4.asFloat(db0[0]);
        try std.testing.expect(@abs(want - got) < 0.15);
    }
    try std.testing.expectEqual(@as(usize, 356843520), bytesNeeded(154880, 4096));
    var w: wts.Weights = .{ .gpa = a };
    w.head = .{ .w = undefined, .s = undefined, .b = undefined, .n = 154880, .k = 4096, .bits = 8, .group = 64 };
    var c: cfg.Config = .{};
    const mtp_stub: wts.Mtp = .{ .eh_proj = w.head, .enorm = undefined, .hnorm = undefined, .norm = undefined };
    try maybeMake(a, undefined, &w, &c, true, false, 0, 1 << 40);
    try std.testing.expect(w.draft_head == null);
    w.mtp = mtp_stub;
    try maybeMake(a, undefined, &w, &c, true, false, 1000, 1000 + bytesNeeded(154880, 4096) - 1);
    try std.testing.expect(w.draft_head == null);
    w.head.bits = 4;
    try maybeMake(a, undefined, &w, &c, true, false, 0, 1 << 40);
    try std.testing.expect(w.draft_head == null);
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const n: u32 = 16;
    const k: u32 = 128;
    const b_w = try dev.buffer(n * k, mtl.ResourceOptions.shared);
    const b_s = try dev.buffer((n * k / 64) * 2, mtl.ResourceOptions.shared);
    const b_b = try dev.buffer((n * k / 64) * 2, mtl.ResourceOptions.shared);
    var wm: wts.Weights = .{ .gpa = a };
    try wm.buffers.appendSlice(a, &.{ b_w, b_s, b_b });
    defer wm.deinit();
    wm.mtp = mtp_stub;
    wm.head = .{ .w = .{ .buf = b_w }, .s = .{ .buf = b_s }, .b = .{ .buf = b_b }, .n = n, .k = k, .bits = 8, .group = 64 };
    const need = bytesNeeded(n, k);
    try maybeMake(a, dev, &wm, &c, false, false, 100, 100 + need);
    try std.testing.expect(wm.draft_head == null and wm.bytes == 0 and wm.buffers.items.len == 3);
    try maybeMake(a, dev, &wm, &c, true, true, 100, 100 + need);
    try std.testing.expect(wm.draft_head == null and wm.bytes == 0 and wm.buffers.items.len == 3);
    try maybeMake(a, dev, &wm, &c, true, false, 100, 100 + need);
    try std.testing.expect(wm.draft_head != null);
    try std.testing.expectEqual(@as(u8, 4), wm.draft_head.?.bits);
    try std.testing.expectEqual(@as(usize, need), wm.bytes);
    try std.testing.expectEqual(@as(usize, 4), wm.buffers.items.len);
}

test "GLM_DRAFT_HEAD_Q4 defaults off, refuses bad text, and skips synthesis when --drafter is set" {
    const a = std.testing.allocator;
    for ([_]?[]const u8{ null, "", "   ", "0" }) |raw| try std.testing.expect(!(try enabled(raw)));
    try std.testing.expect(try enabled("1"));
    try std.testing.expectError(error.InvalidEnvironment, enabled("yes"));
    try std.testing.expectError(error.InvalidEnvironment, enabled("2"));
    const why = try platform.environment.describe(a, "GLM_DRAFT_HEAD_Q4", "yes");
    defer a.free(why);
    try std.testing.expect(std.mem.startsWith(u8, why, "GLM_DRAFT_HEAD_Q4=\"yes\":"));
    defer _ = platform.env.unsetenv("GLM_DRAFT_HEAD_Q4");
    var w0: wts.Weights = .{ .gpa = a };
    var c: cfg.Config = .{};
    _ = platform.env.setenv("GLM_DRAFT_HEAD_Q4", "yes", 1);
    try std.testing.expectError(error.InvalidEnvironment, maybeMakeFromEnv(a, undefined, &w0, &c, false, 0, 1 << 40));
    _ = platform.env.unsetenv("GLM_DRAFT_HEAD_Q4");
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const n: u32 = 16;
    const k: u32 = 128;
    const b_w = try dev.buffer(n * k, mtl.ResourceOptions.shared);
    const b_s = try dev.buffer((n * k / 64) * 2, mtl.ResourceOptions.shared);
    const b_b = try dev.buffer((n * k / 64) * 2, mtl.ResourceOptions.shared);
    var wm: wts.Weights = .{ .gpa = a };
    try wm.buffers.appendSlice(a, &.{ b_w, b_s, b_b });
    defer wm.deinit();
    wm.head = .{ .w = .{ .buf = b_w }, .s = .{ .buf = b_s }, .b = .{ .buf = b_b }, .n = n, .k = k, .bits = 8, .group = 64 };
    wm.mtp = .{ .eh_proj = wm.head, .enorm = undefined, .hnorm = undefined, .norm = undefined };
    const need = bytesNeeded(n, k);
    try maybeMakeFromEnv(a, dev, &wm, &c, false, 0, need);
    try std.testing.expect(wm.draft_head == null and wm.bytes == 0 and wm.buffers.items.len == 3);
    _ = platform.env.setenv("GLM_DRAFT_HEAD_Q4", "0", 1);
    try maybeMakeFromEnv(a, dev, &wm, &c, false, 0, need);
    try std.testing.expect(wm.draft_head == null and wm.bytes == 0 and wm.buffers.items.len == 3);
    _ = platform.env.setenv("GLM_DRAFT_HEAD_Q4", "1", 1);
    try maybeMakeFromEnv(a, dev, &wm, &c, true, 0, need);
    try std.testing.expect(wm.draft_head == null and wm.bytes == 0 and wm.buffers.items.len == 3);
    try maybeMakeFromEnv(a, dev, &wm, &c, false, 0, need);
    try std.testing.expect(wm.draft_head != null and wm.draft_head.?.bits == 4 and wm.bytes == need and wm.buffers.items.len == 4);
}
