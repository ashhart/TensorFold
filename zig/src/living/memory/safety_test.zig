//! The bank's error paths: every allocation failure around an ownership transfer frees each buffer once, leaks none.
const std = @import("std");
const quant = @import("quant.zig");
const whitener = @import("whitener.zig");
const bank_mod = @import("bank.zig");
const Bank = bank_mod.Bank;
const ArchSpec = bank_mod.ArchSpec;
const RawState = bank_mod.RawState;
const SnapHandle = bank_mod.SnapHandle;
const t = std.testing;

/// A deterministic backend: per-token hidden directions and states, snaps that only count.
pub const Mock = struct {
    gpa: std.mem.Allocator,
    arch: ArchSpec,
    snaps_alive: usize = 0,
    composed: ?*std.ArrayList(u8) = null, // when set, each composed state's bytes are appended here

    pub const spec = ArchSpec{
        .family = .qwen3_5,
        .hidden = 64,
        .num_layers = 4,
        .layer_lex = 1,
        .layer_sem = 2,
        .num_attn = 2,
        .kv_heads = 2,
        .head_dim = 32,
        .rotary_dim = 16,
        .rope_theta = 10_000_000.0,
        .num_rec = 2,
        .rec_heads = 2,
        .rec_rows = 16,
        .rec_cols = 16,
        .conv_elems = 12,
    };

    pub fn backend(m: *Mock) bank_mod.Backend {
        return .{ .ctx = m, .arch = m.arch, .extractHidden = extractHidden, .compileState = compileState, .buildSnap = buildSnap, .dropSnap = dropSnap };
    }

    fn extractHidden(ctx: *anyopaque, ids: []const u32, _: usize, _: usize, out_lex: []f32, out_sem: []f32) !void {
        const self: *Mock = @ptrCast(@alignCast(ctx));
        const d = self.arch.hidden;
        @memset(out_lex, 0);
        @memset(out_sem, 0);
        for (ids, 0..) |id, k| {
            out_lex[k * d + 0] = 4.0;
            out_lex[k * d + 1] = -3.0;
            out_lex[k * d + (4 + (id % (d - 4)))] = 5.0;
            out_sem[k * d + 0] = 3.0;
            out_sem[k * d + 2] = -2.0;
            out_sem[k * d + (4 + (id % (d - 4)))] = 5.0;
        }
    }

    fn compileState(ctx: *anyopaque, base: ?*const RawState, ids: []const u32, out: *RawState) !void {
        const self: *Mock = @ptrCast(@alignCast(ctx));
        const off: usize = if (base) |b| b.len else 0;
        for (0..self.arch.num_attn) |a| {
            for (0..self.arch.kv_heads) |h| for (0..ids.len) |k| {
                const at = (h * ids.len + k) * self.arch.head_dim;
                for (0..self.arch.head_dim) |c| {
                    out.kv_k[a][at + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[k] + c) % 17)) * 0.1);
                    out.kv_v[a][at + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[k] + c + 3) % 19)) * 0.1);
                }
            };
            quant.ropeRotateKvBf16(out.kv_k[a], self.arch.kv_heads, ids.len, self.arch.head_dim, off, self.arch.rotary_dim, self.arch.rope_theta, false);
        }
        for (0..self.arch.num_rec) |r| {
            if (base) |b| @memcpy(out.rec_state[r], b.rec_state[r]);
            for (out.rec_state[r], 0..) |*v, idx| v.* += @as(f32, @floatFromInt((ids[0] + idx) % 7)) * 0.05;
            for (out.rec_conv[r], 0..) |*v, idx| v.* = quant.f32ToBf16(@as(f32, @floatFromInt((ids[0] + idx) % 5)) * 0.1);
        }
    }

    fn buildSnap(ctx: *anyopaque, composed: *const RawState, suffix_ids: []const u32) !SnapHandle {
        const self: *Mock = @ptrCast(@alignCast(ctx));
        if (self.composed) |out| {
            for (composed.kv_k) |k| try out.appendSlice(self.gpa, std.mem.sliceAsBytes(k));
            for (composed.kv_v) |v| try out.appendSlice(self.gpa, std.mem.sliceAsBytes(v));
            for (composed.rec_state) |r| try out.appendSlice(self.gpa, std.mem.sliceAsBytes(r));
            for (composed.rec_conv) |c| try out.appendSlice(self.gpa, std.mem.sliceAsBytes(c));
        }
        const s = try self.gpa.create(u32);
        s.* = composed.len + @as(u32, @intCast(suffix_ids.len));
        self.snaps_alive += 1;
        return .{ .saved = s, .at = s.* };
    }

    fn dropSnap(ctx: *anyopaque, ptr: *anyopaque) void {
        const self: *Mock = @ptrCast(@alignCast(ctx));
        self.gpa.destroy(@as(*u32, @ptrCast(@alignCast(ptr))));
        self.snaps_alive -= 1;
    }
};

const cal_ids = [_][]const u32{ &.{ 0, 1, 2, 3, 4, 5 }, &.{ 6, 7, 8, 9, 10, 11 }, &.{ 12, 13, 14, 15, 16, 17 } };
const pref_ids = [_]u32{ 1, 2, 3, 4 };
const suf_ids = [_]u32{ 5, 6 };
pub const fact_ids = [_]u32{ 30, 31, 32, 33, 34 };

/// A calibrated bank in `dir` with one taught fact (parent 7) and, when `saved`, its file written.
pub fn taught(gpa: std.mem.Allocator, dir: []const u8, mock: *Mock, saved: bool) !*Bank {
    const b = try Bank.init(gpa, t.io, dir, mock.backend());
    errdefer b.deinit();
    try b.ensureCalibrated(&cal_ids, &pref_ids, &suf_ids);
    _ = try b.teachChunk(.{ .parent_id = 7, .body_ids = &fact_ids, .gate_ids = &fact_ids, .is_small_fact = true });
    if (saved) _ = try b.save();
    return b;
}

/// Fits with the `k`-th allocation failing, for every `k` until a fit succeeds: each failure must free all it took.
fn fitAtEveryFailure(raw: []const f32, n: usize, dim: usize, k_pcs: usize) !void {
    var k: usize = 0;
    while (true) : (k += 1) {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = k });
        const a = failing.allocator();
        if (whitener.LayerWhitener.fit(a, raw, n, dim, k_pcs, 8)) |fitted| {
            var w = fitted;
            w.deinit(a);
            return;
        } else |e| try t.expectEqual(error.OutOfMemory, e);
    }
}

test "bank safety: the whitener's fit frees its basis once on every allocation failure, shrunk or not" {
    const dim = 16;
    var full: [12 * dim]f32 = undefined;
    for (&full, 0..) |*v, i| v.* = @floatFromInt((i * 7919 + i / dim * 31) % 23);
    try fitAtEveryFailure(&full, 12, dim, 4);
    // two distinct rows repeated: rank 2 against 4 asked, so the basis is shrunk to the vectors found
    var low: [12 * dim]f32 = undefined;
    for (&low, 0..) |*v, i| v.* = if ((i / dim) % 2 == 0) @floatFromInt(i % dim) else @floatFromInt((i * 3) % 5);
    try fitAtEveryFailure(&low, 12, dim, 4);
}

test "bank safety: a teach that fails at any allocation, the index rebuild included, frees its unit once" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: Mock = .{ .gpa = t.allocator, .arch = Mock.spec };
    var failing = t.FailingAllocator.init(t.allocator, .{});
    const b = try taught(failing.allocator(), dir, &mock, false);
    defer b.deinit();
    const ids = [_]u32{ 40, 41, 42, 43 };
    var k: usize = 0;
    while (true) : (k += 1) {
        failing.fail_index = failing.alloc_index + k;
        const before = b.units.items.len;
        if (b.teachChunk(.{ .parent_id = 8, .body_ids = &ids, .gate_ids = &ids, .is_small_fact = true })) |_| break else |e| {
            try t.expectEqual(error.OutOfMemory, e);
            try t.expectEqual(before, b.units.items.len); // a failed teach keeps no unit
        }
    }
    failing.fail_index = std.math.maxInt(usize);
    try t.expectEqual(@as(usize, 2), b.units.items.len);
}

test "bank safety: a load that fails at any allocation frees what it read once and keeps nothing" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    var mock: Mock = .{ .gpa = t.allocator, .arch = Mock.spec };
    const written = try taught(t.allocator, dir, &mock, true);
    written.deinit();
    var k: usize = 0;
    while (true) : (k += 1) {
        var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = k });
        const b = Bank.init(failing.allocator(), t.io, dir, mock.backend()) catch continue;
        defer b.deinit();
        if (b.loadIfPresent()) |found| {
            try t.expect(found);
            try t.expectEqual(@as(usize, 1), b.units.items.len);
            break;
        } else |e| try t.expectEqual(error.OutOfMemory, e);
    }
}
