//! The keyed sampler on the GPU forward: rank 0 sends each call's sampling on the plan link; followers run the steps.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const kops = @import("dsv41_kernels").ops;
const pipe = @import("sampling_pipe.zig");
const Gpu = @import("sampling_dev.zig").Gpu;

pub const Sampling = pipe.Sampling;
pub const Seg = pipe.Pipe.Seg;
pub const sampled = pipe.sampled;

/// TF_DSV41_SAMPLING=1 (every rank alike): model.zig makes the sampler; unset, sampled requests are refused as before.
pub fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_SAMPLING") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One rank's sampler over its forward (model.zig makes one on every rank; it hooks the follower side itself).
pub const GpuSampler = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    gpu: Gpu,
    p: pipe.Pipe,
    /// the call in flight (rows, sampling, first draw position) and the device choice's pinned copy, event and tokens
    cur: ?Cur = null,
    pinned: ?cuda.HostBuffer = null,
    done: ?cuda.Event = null,
    chosen: ?cuda.DeviceBuffer = null,
    /// TF_DSV41_SAMP_BATCH=1 (rank 0 decides): a forward's keyed segments in one call, one gather a forward
    batch: bool = false,
    /// the batched call's per-segment device-choice start (null: none), for its hook
    seg_pos0: []const ?u64 = &.{},
    /// keyed paths already logged (`note`)
    seen: u8 = 0,

    const Cur = struct { row0: u32, s: Sampling, pos0: ?u64 };

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward) !*GpuSampler {
        const x = try gpa.create(GpuSampler);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .f = f, .gpu = try Gpu.init(f.runner.d, f.runner.stream, f.runner.kernels), .p = undefined };
        errdefer x.gpu.deinit();
        x.p = try pipe.Pipe.initVocabAt(gpa, f.comm, f.opts.rank, f.cfg.vocab, try pipe.nucleusEnv());
        f.sampler = .{ .ptr = x, .follow = follow };
        x.p.after_gather = .{ .ctx = x, .run = afterGather };
        x.batch = try batchEnv();
        return x;
    }

    /// TF_DSV41_SAMP_BATCH: 0 (default) | 1.
    pub fn batchEnv() !bool {
        const v = std.mem.span(std.c.getenv("TF_DSV41_SAMP_BATCH") orelse return false);
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
        if (std.mem.eql(u8, v, "1")) return true;
        return error.BadSampBatch;
    }

    /// Whether `chooseSegments` takes a segment under `s` (sampling_pipe.Pipe.batchable, with the knob on).
    pub fn batchable(x: *const GpuSampler, s: Sampling) bool {
        return x.batch and x.p.batchable(s);
    }

    /// The most segments one op_sample_many carries.
    pub const max_segments = 16;
    const seg_words = 8;

    /// Rank 0: several keyed segments of "w.logits" in one call (op 13 first), each choosing from the one gather.
    pub fn chooseSegments(x: *GpuSampler, segs: []const pipe.Pipe.Seg) !void {
        if (segs.len == 0 or segs.len > max_segments) return error.BadSegments;
        var msg: [2 + seg_words * max_segments]i64 = undefined;
        var pos0: [max_segments]?u64 = @splat(null);
        msg[0] = fwd.Forward.op_sample_many;
        msg[1] = @intCast(segs.len);
        for (segs, 0..) |g, i| {
            if (x.f.after_sample != null) if (g.positions) |ps| if (consecutive(ps[0..g.n])) {
                pos0[i] = ps[0];
            };
            const o = opOf(g.row0, g.n, g.s);
            @memcpy(msg[2 + seg_words * i ..][0..6], o[1..7]);
            msg[2 + seg_words * i + 6] = @bitCast(g.s.seed);
            msg[2 + seg_words * i + 7] = if (pos0[i]) |p0| @intCast(p0) else -1;
        }
        try x.f.sendSample(msg[0 .. 2 + seg_words * segs.len]);
        try x.runSegments(segs, pos0[0..segs.len]);
    }

    fn runSegments(x: *GpuSampler, segs: []const pipe.Pipe.Seg, pos0: []const ?u64) !void {
        const f = x.f;
        const lg = f.runner.addressOf("w.logits") orelse return error.Unbound;
        if (f.grammar) |g| for (segs) |sg| try g.apply(g.ptr, sg.row0, sg.n);
        x.cur = null; // the single call's hook stays off: the segments' own below
        x.seg_pos0 = pos0;
        defer x.seg_pos0 = &.{};
        for (segs) |sg| x.note(sg.s, true);
        try x.p.chooseSegments(x.gpu.device(), lg, segs, .{ .ctx = x, .run = segGather });
        if (x.p.whole_rows > 0) std.log.scoped(.dsv41).debug("sampling: {d} rows taken whole ({d} segments)", .{ x.p.whole_rows, segs.len });
    }

    /// A follower: op_sample_many's segments (tokens dropped; device choices as the leader's).
    fn followSegments(x: *GpuSampler, msg: []const i64) !void {
        if (msg.len < 2) return error.BadPlan;
        const ns: usize = @intCast(msg[1]);
        if (ns == 0 or ns > max_segments or msg.len != 2 + seg_words * ns) return error.BadPlan;
        var segs: [max_segments]pipe.Pipe.Seg = undefined;
        var pos0: [max_segments]?u64 = undefined;
        var outs: [64]u32 = undefined;
        var used: usize = 0;
        for (0..ns) |i| {
            const w = msg[2 + seg_words * i ..][0..seg_words];
            const n: u32 = @intCast(w[1]);
            if (n == 0 or used + n > outs.len) return error.BadPlan;
            segs[i] = .{ .row0 = @intCast(w[0]), .n = n, .s = .{ .seed = @bitCast(w[6]), .temperature = @bitCast(w[2]), .top_k = @intCast(w[3]), .top_p = @bitCast(w[4]), .min_p = @bitCast(w[5]) }, .out = outs[used..][0..n] };
            pos0[i] = if (w[7] >= 0) @intCast(w[7]) else null;
            used += n;
        }
        try x.runSegments(segs[0..ns], pos0[0..ns]);
    }

    /// chooseSegments' hook (every rank): device choices from the shared gather, then wait for the copy; false: none.
    fn segGather(ctx: *anyopaque, gathered: u64, n: usize, k: usize, segs: []const pipe.Pipe.Seg, host: []u8) anyerror!bool {
        const x: *GpuSampler = @ptrCast(@alignCast(ctx));
        const h = x.f.after_sample orelse return false;
        const r = x.f.runner;
        const ops = r.kernels.others(r.stream);
        const W: usize = x.f.comm.world();
        if (ops.f.vs_choose == null or W * k > kops.glue.vs_max) return false;
        const r0 = segs[0].row0;
        var any = false;
        for (segs, x.seg_pos0) |g, p0| any = any or (p0 != null and !isNucleusRow(g.s));
        if (!any) return false;
        try x.ensureDevice(host.len, n);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.pinned.?.bytes.ptr, gathered, host.len, r.stream.handle), "cuMemcpyDtoHAsync");
        try x.done.?.record(r.stream);
        const vocab: u32 = x.f.cfg.vocab;
        for (segs, x.seg_pos0) |g, p0| {
            const p = p0 orelse continue;
            const a = g.row0 - r0;
            if (isNucleusRow(g.s)) continue;
            const count = pipe.countOf(g.s, vocab, x.p.nucleus);
            try ops.vsChoose(gathered, W, n, k, count, g.s.seed, p -% a, g.s.top_k, g.s.temperature, g.s.top_p, g.s.min_p, x.chosen.?.ptr);
            try h.run(h.ctx, g.row0, g.n, x.chosen.?.ptr + 8 * @as(u64, a));
        }
        try x.done.?.synchronize();
        @memcpy(host, x.pinned.?.bytes[0..host.len]);
        return true;
    }

    /// The pinned copy, its event and the device choices, grown to `bytes` / `n` rows.
    fn ensureDevice(x: *GpuSampler, bytes: usize, n: usize) !void {
        const r = x.f.runner;
        if (x.pinned == null or x.pinned.?.bytes.len < bytes) {
            if (x.pinned) |*b| b.free();
            x.pinned = null;
            x.pinned = try cuda.HostBuffer.alloc(r.d, @max(bytes, 1 << 16));
        }
        if (x.chosen == null or x.chosen.?.len < 8 * n) {
            if (x.chosen) |*b| b.free();
            x.chosen = null;
            x.chosen = try cuda.DeviceBuffer.alloc(r.d, @max(8 * n, 1024));
        }
        if (x.done == null) x.done = try cuda.Event.init(r.d, false);
    }

    /// Rank 0, once a path: which keyed path serving takes (the server's default sampling decides it).
    fn note(x: *GpuSampler, s: Sampling, batched: bool) void {
        if (x.f.comm.rank() != 0) return;
        const V: usize = x.p.V;
        const count = pipe.countOf(s, x.p.vocab, x.p.nucleus);
        const kind: u3 = if (@min(count, V) > pipe.max_k or V > pipe.max_cols) 0 else if (isNucleusRow(s)) 1 else 2;
        const bit = @as(u8, 1) << (kind + @as(u3, if (batched) 3 else 0));
        if (x.seen & bit != 0) return;
        x.seen |= bit;
        const what = switch (kind) {
            0 => "whole rows (top_k 0 without the nucleus path, or past the device top-k)",
            1 => "nucleus candidates + row statistics",
            else => "top_k candidates",
        };
        const start = x.f.after_sample != null and kind == 2;
        std.log.scoped(.dsv41).info("sampling: first keyed call ({s}): {s}, {d} candidates a rank (top_k {d}, top_p {d}, T {d}), device choice {s}", .{ if (batched) "one call a forward" else "one call a segment", what, @min(count, V), s.top_k, s.top_p, s.temperature, if (start) "on" else "off" });
    }

    pub fn deinit(x: *GpuSampler) void {
        x.f.sampler = null;
        if (x.pinned) |*b| b.free();
        if (x.done) |*e| e.deinit();
        if (x.chosen) |*b| b.free();
        x.gpu.deinit();
        x.p.deinit();
        x.gpa.destroy(x);
    }

    /// Rank 0: the keyed choices of "w.logits" rows [row0, row0 + n) under `s` (T > 0), row i drawn at `draws[i]`.
    pub fn choose(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: []const u64, out: []u32) !void {
        // a device start (Forward.after_sample) needs the seed and consecutive draws on every rank: op 12 carries them
        const pos0: ?u64 = if (x.f.after_sample != null and consecutive(draws)) draws[0] else null;
        if (pos0) |p0| try x.f.sendSample(&opDevice(row0, n, s, p0)) else try x.f.sendSample(&opOf(row0, n, s));
        try x.run(row0, n, s, draws, out, pos0);
    }

    fn consecutive(draws: []const u64) bool {
        if (draws.len == 0) return false;
        for (draws, 0..) |d, i| if (d != draws[0] + i) return false;
        return true;
    }

    fn run(x: *GpuSampler, row0: u32, n: u32, s: Sampling, draws: ?[]const u64, out: []u32, pos0: ?u64) !void {
        const f = x.f;
        const V: u64 = x.p.V;
        const lg = (f.runner.addressOf("w.logits") orelse return error.Unbound) + 4 * V * row0;
        if (f.grammar) |g| try g.apply(g.ptr, row0, n);
        x.cur = .{ .row0 = row0, .s = s, .pos0 = pos0 };
        defer x.cur = null;
        x.note(s, false);
        try x.p.choose(x.gpu.device(), lg, n, s, draws, out);
        if (x.p.whole_rows > 0) std.log.scoped(.dsv41).debug("sampling: {d} of {d} rows taken whole", .{ x.p.whole_rows, n });
    }

    /// The plan-link message: [op, row0, n, T, top_k, top_p, min_p] (floats as their bits; the seed stays on rank 0).
    pub fn opOf(row0: u32, n: u32, s: Sampling) [7]i64 {
        return .{ fwd.Forward.op_sample, row0, n, @bitCast(s.temperature), s.top_k, @bitCast(s.top_p), @bitCast(s.min_p) };
    }

    /// The message with the device start's seed and first draw position: [op 12 ..., seed, pos0] (9 words).
    pub fn opDevice(row0: u32, n: u32, s: Sampling, pos0: u64) [9]i64 {
        const o = opOf(row0, n, s);
        return o ++ [2]i64{ @bitCast(s.seed), @intCast(pos0) };
    }

    /// A follower: the leader's call from its message (tokens dropped; with 9 words its device choice too).
    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *GpuSampler = @ptrCast(@alignCast(ptr));
        if (msg[0] == fwd.Forward.op_sample_many) return x.followSegments(msg);
        if (msg.len != 7 and msg.len != 9) return error.BadPlan;
        const n: u32 = @intCast(msg[2]);
        const dev = msg.len == 9;
        const s: Sampling = .{ .seed = if (dev) @bitCast(msg[7]) else 0, .temperature = @bitCast(msg[3]), .top_k = @intCast(msg[4]), .top_p = @bitCast(msg[5]), .min_p = @bitCast(msg[6]) };
        var out: [64]u32 = undefined;
        if (n == 0 or n > out.len) return error.BadPlan;
        try x.run(@intCast(msg[1]), n, s, null, out[0..n], if (dev) @intCast(msg[8]) else null);
    }

    /// The download hook (every rank): device tokens before the pass when a start is wanted, then wait; false: none.
    fn afterGather(ctx: *anyopaque, gathered: u64, n: usize, k: usize, count: usize, host: []u8) anyerror!bool {
        const x: *GpuSampler = @ptrCast(@alignCast(ctx));
        const c = x.cur orelse return false;
        const h = x.f.after_sample orelse return false;
        const p0 = c.pos0 orelse return false;
        const r = x.f.runner;
        const ops = r.kernels.others(r.stream);
        const W: usize = x.f.comm.world();
        if (ops.f.vs_choose == null or pipe.sampled(c.s) == false or isNucleusRow(c.s) or W * k > kops.glue.vs_max) return false;
        try x.ensureDevice(host.len, n);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.pinned.?.bytes.ptr, gathered, host.len, r.stream.handle), "cuMemcpyDtoHAsync");
        try x.done.?.record(r.stream);
        try ops.vsChoose(gathered, W, n, k, count, c.s.seed, p0, c.s.top_k, c.s.temperature, c.s.top_p, c.s.min_p, x.chosen.?.ptr);
        try h.run(h.ctx, c.row0, @intCast(n), x.chosen.?.ptr);
        try x.done.?.synchronize();
        @memcpy(host, x.pinned.?.bytes[0..host.len]);
        return true;
    }

    fn isNucleusRow(s: Sampling) bool {
        return s.temperature > 0 and s.top_k == 0 and s.top_p > 0 and s.top_p < 1;
    }
};
