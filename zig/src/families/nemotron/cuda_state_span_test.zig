//! The owner A4+A5 receipt battery (PR 5102): A4 — a context re-opened from the pool comes
//! up at the SAME fresh-open state create() gives (resetStream is the one reset both open
//! paths call), and a shorter second prompt after a release cannot see the first stream's
//! state; A5 — the span gate and the frame's row-0 identity are DERIVED (spanOpen reads the
//! prompt-phase base absorbRows records; row0Token reads the config's declaration), never a
//! prompt-length literal. Zero-GPU rig: recording fakes on the driver seams the touched
//! paths reach (launch/memset/DtoH), no allocations, no GPU.

const std = @import("std");
const cuda = @import("cuda");
const lc = @import("cuda_lilicorr.zig");
const absorbz = @import("cuda_lilicorr_absorb.zig");
const openz = @import("cuda_lilicorr_open.zig");
const wts = @import("cuda_lilicorr_weights.zig");
const lanes = @import("lanes");
const Stream = lanes.Stream;

// -- the zero-GPU driver fakes --

const Set = struct {
    var dst: u64 = 0;
    var n: usize = 0;
    var val: u8 = 0;
    var fires: usize = 0;

    fn arm() void {
        dst = 0;
        n = 0;
        val = 0;
        fires = 0;
    }

    fn record(d: cuda.abi.DevicePtr, v: u8, count: usize) cuda.abi.Result {
        dst = d;
        n = count;
        val = v;
        fires += 1;
        return cuda.abi.success;
    }

    fn memset(d: cuda.abi.DevicePtr, v: u8, count: usize) callconv(.c) cuda.abi.Result {
        return record(d, v, count);
    }

    fn memsetAsync(d: cuda.abi.DevicePtr, v: u8, count: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return record(d, v, count);
    }
};

const Fake = struct {
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn memcpyH(_: ?*anyopaque, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn launchEx(_: *const cuda.abi.LaunchConfig, _: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

/// The rig the absorb-guard harness pins (fake affine/norms/fns faces), plus the memset
/// recorder so a reset's ring re-fill grades as an executed device call covering the ring.
const Rig = struct {
    driver: cuda.Driver = undefined,
    ctx_dev: cuda.Context = undefined,
    e: lc.E = undefined,
    d: lc.Draft = undefined,
    ring_buf: cuda.DeviceBuffer = undefined,
    staged_b: [8192]u8 align(16) = @splat(0),
    lz: [1]wts.Layer = undefined,

    fn arm(self: *Rig) *lc.Draft {
        const e = &self.e;
        const d = &self.d;
        self.driver.api.cuLaunchKernel = Fake.launch;
        self.driver.api.cuLaunchKernelEx = Fake.launchEx;
        self.driver.api.cuMemcpyDtoHAsync_v2 = Fake.memcpyH;
        self.driver.api.cuMemsetD8_v2 = Set.memset;
        self.driver.api.cuMemsetD8Async = Set.memsetAsync;
        self.ctx_dev = .{ .d = &self.driver, .device = 0, .handle = null };
        e.ctx = &self.ctx_dev;
        e.stream = .{ .d = &self.driver, .handle = null };
        e.b = undefined;
        e.k = undefined;
        e.k.wide = true;
        e.k.modelopt = null;
        e.k.ck = null;
        e.k.ggml = null;
        const fade: cuda.Function = .{ .d = &self.driver, .handle = @ptrFromInt(0x9100) };
        e.k.affine = .{
            .group_fn = fade,
            .mlp128_fn = fade,
            .gemv_fn = fade,
            .verify_fn = fade,
            .verify64_fn = fade,
            .verify128_fn = fade,
            .split_fn = fade,
            .seg_fn = fade,
            .many_fn = fade,
            .prefill_fn = fade,
            .prefill22_fn = fade,
            .prefill_many_fn = fade,
            .words_fn = fade,
            .scales_fn = fade,
            .gemv_blocks = 1,
            .split = null,
            .pdl = true,
            .wide_verify = false,
        };
        e.k.norms = undefined;
        e.k.norms.tail_fns = .{ fade, fade };
        d.e = e;
        d.c = .{
            .hidden = 128,
            .intermediate = 128,
            .layers = 1,
            .vocab = 16,
            .heads = 2,
            .kv_heads = 1,
            .head_dim = 32,
            .block = 4,
            .mask = 0,
            .window = 4096,
            .group = 8,
            .rank = 4,
            .topk = 2,
            .eps = 1e-5,
            .theta = 10000,
            .taps = .{ 1, 2, 3, 4, 5, 6, 7, 8 },
            .n_taps = 4,
            .parallel = 0,
            .parallel_declared = false,
        };
        d.gpa = std.testing.allocator;
        d.tap_rows = 512;
        d.taps_in = 0x1a0000;
        d.th = 0x1b0000;
        d.thn = 0x1c0000;
        d.kabs = 0x1d0000;
        d.vabs = 0x1e0000;
        d.kn = 0x1f0000;
        d.xs = 0x200000;
        d.anchor = 0x210000;
        d.staged = .{ .d = &self.driver, .bytes = self.staged_b[0..] };
        d.w = undefined;
        d.w.fc = .{ .w = 0x30000, .s = 0x31000, .b = 0x32000, .n = 64, .k = 1024, .npad = 128 };
        d.w.hidden_norm = 0x220000;
        const wq: cuda.qlinear.Weight = .{ .affine4 = .{ .w = 0x33000, .s = 0x34000, .b = 0x35000, .n = 64, .k = 1024, .npad = 128 } };
        self.lz = .{.{ .input_norm = 0x40000, .post_norm = 0x40040, .q = wq, .k = wq, .v = wq, .o = wq, .q_norm = 0x40080, .k_norm = 0x400c0, .sink = 0x40100, .gate = wq, .up = wq, .down = wq, .attn_conv = wq, .attn_base = 0x40140, .mlp_conv = wq, .mlp_base = 0x40180 }};
        d.w.layers = self.lz[0..];
        d.fns = .{
            .gather_taps = fade,
            .gather_rows = fade,
            .norm_rope = fade,
            .ring_put = fade,
            .sums = fade,
            .norm_rope_batch = fade,
            .attn_partial = fade,
            .attn_merge = fade,
            .mlp_act = fade,
            .mask_rows = fade,
            .dyn_conv_batch = fade,
            .token_proj = fade,
            .lat_attn = fade,
            .factor = fade,
            .edge_linear = fade,
            .l2norm = fade,
            .dots = fade,
            .rms = fade,
            .lin_f32 = fade,
            .silu_bf16 = fade,
            .add_bf16 = fade,
            .lattice = fade,
            .rms_f32 = fade,
            .widen = fade,
            .narrow = fade,
        };
        d.affine = e.k.affine;
        self.ring_buf = .{ .d = self.ctx_dev.d, .ptr = 0x5a000000, .len = 1 << 24 };
        return d;
    }
};

fn expectFresh(ctx: *const lc.Ctx, ring: *const cuda.DeviceBuffer) !void {
    const t = std.testing;
    try t.expectEqual(@as(u64, 0), ctx.begin);
    try t.expectEqual(@as(u64, 0), ctx.end);
    try t.expectEqual(@as(usize, 0), ctx.held_n);
    try t.expect(!ctx.anchor_valid);
    try t.expect(ctx.span_base == null);
    for (ctx.held) |h| try t.expectEqual(@as(u32, 0), h); // held drafts' bytes too
    // the executor ran the reset's ring re-fill over the WHOLE ring: no stale row can leak
    try t.expect(Set.fires > 0);
    try t.expectEqual(ring.ptr, Set.dst);
    try t.expectEqual(ring.len, Set.n);
    try t.expectEqual(@as(u8, 0), Set.val);
}

test "a pooled context re-opens at the fresh-open state (owner A4)" {
    const t = std.testing;
    var rig: Rig = .{};
    const d = rig.arm();
    defer d.pool.deinit(d.gpa);
    defer d.streams.deinit(d.gpa);
    const ring = rig.ring_buf;
    Set.arm();

    // the prior stream's state: positions, held drafts, anchor gate, span base, dirty ring
    var used: lc.Ctx = .{ .ring = ring };
    used.begin = 7;
    used.end = 313;
    used.held[0] = 9;
    used.held[1] = 41;
    used.held_n = 2;
    used.anchor_valid = true;
    used.span_base = 36;
    try d.pool.append(d.gpa, &used);

    var s: Stream = undefined;
    try openz.openStream(d, &s);
    const ctx = d.streams.get(&s).?;
    try t.expect(ctx == &used); // the re-opened context IS the pooled one (the defect's path)
    try expectFresh(ctx, &ring);
}

test "a shorter second prompt matches a fresh open's own absorb history (owner A4)" {
    const t = std.testing;
    var rig: Rig = .{};
    const d = rig.arm();
    defer d.pool.deinit(d.gpa);
    defer d.streams.deinit(d.gpa);
    const ring = rig.ring_buf;
    Set.arm();

    var used: lc.Ctx = .{ .ring = ring };
    used.end = 200; // a first stream's longer session
    used.anchor_valid = true;
    try d.pool.append(d.gpa, &used);
    var s: Stream = undefined;
    try openz.openStream(d, &s);
    const reused = d.streams.get(&s).?;
    try expectFresh(reused, &ring); // the pool path ran the same reset create() runs

    var fresh: lc.Ctx = .{ .ring = ring };
    // a SHORTER second prompt (5 rows) absorbed on both: the ring positions and the span
    // base agree with the fresh-open context's own history, position for position
    try absorbz.absorbRows(d, reused, 0x6a000000, 0, 5, null, true);
    Set.arm(); // clear the recorder between the twin absorbs
    try absorbz.absorbRows(d, &fresh, 0x6a000000, 0, 5, null, true);
    try t.expectEqual(fresh.begin, reused.begin);
    try t.expectEqual(fresh.end, reused.end);
    try t.expectEqual(fresh.span_base, reused.span_base);
    try t.expectEqual(fresh.anchor_valid, reused.anchor_valid);
    try t.expectEqual(@as(u64, 5), reused.span_base.?);
}

test "the span opens at 2 accepted rows beyond the prompt phase at every prompt length (owner A5)" {
    const t = std.testing;
    var rig: Rig = .{};
    const d = rig.arm();
    for ([_]usize{ 5, 35, 200 }) |prompt_len| {
        var ctx: lc.Ctx = .{ .ring = rig.ring_buf };
        try absorbz.absorbRows(d, &ctx, 0x6a000000, 0, prompt_len, null, true);
        try t.expectEqual(@as(u64, prompt_len), ctx.span_base.?); // base = prompt phase END, derived
        try t.expect(!lc.spanOpen(&ctx)); // 0 accepted rows: closed (the 11/24 oracle's frames 0-1)
        try absorbz.absorbRows(d, &ctx, 0x6a000000, prompt_len, 1, null, false); // 1 live row
        try t.expect(!lc.spanOpen(&ctx)); // 1 accepted: still closed
        try absorbz.absorbRows(d, &ctx, 0x6a000000, prompt_len + 1, 1, null, false); // the 2nd live row
        try t.expect(lc.spanOpen(&ctx)); // >=2 accepted rows beyond prompt+absorb: OPEN
        try absorbz.absorbRows(d, &ctx, 0x6a000000, prompt_len + 2, 5, null, false);
        try t.expect(lc.spanOpen(&ctx)); // monotone once open
    }
}

test "no prompt-length literal survives in the span or row-0 rule (owner A5, structural)" {
    const t = std.testing;
    const round_src = @embedFile("cuda_lilicorr_round.zig");
    try t.expectEqual(@as(usize, 0), std.mem.count(u8, round_src, ">= 38")); // the literal is gone
    try t.expect(std.mem.indexOf(u8, round_src, "lc.spanOpen(cx)") != null);
    try t.expect(std.mem.indexOf(u8, round_src, "lc.row0Token(&c, it.pending)") != null);
    const absorb_src = @embedFile("cuda_lilicorr_absorb.zig");
    try t.expect(std.mem.indexOf(u8, absorb_src, "if (prompt) ctx.span_base = ctx.end;") != null);
    // ONE reset, BOTH open paths: the decl plus the create() and openStream() call sites
    const open_src = @embedFile("cuda_lilicorr_open.zig");
    try t.expectEqual(@as(usize, 3), std.mem.count(u8, open_src, "resetStream("));
}
