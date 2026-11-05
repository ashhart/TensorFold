//! One Qwen3 dense model on one CUDA device for prompt-only heads: weights, kernels and scratch sized to a window.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("../config.zig");
const lm = @import("lm.zig");
const store = @import("store.zig");
const weights = @import("weights.zig");

const Allocator = std.mem.Allocator;
const Model = weights.Model;

/// Inputs one pass pools at most (the embedding lane's default batch).
pub const max_inputs: usize = 128;
/// The least window the budget may shrink a load to.
pub const least_window: usize = 2048;

pub const Engine = struct {
    gpa: Allocator,
    d: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,
    m: Model,
    l: lm.Lm,
    /// Device bytes the weights took.
    weights: usize = 0,
    /// Tokens a pass takes at most, over all its inputs; set by `size`.
    window: usize = 0,
    /// `window` rounded up to whole attention chunks: the positions the RoPE table and the caches hold.
    cap: usize = 0,
    scratch: cuda.DeviceBuffer = undefined,
    a: lm.Addrs = .{},
    ids_host: cuda.HostBuffer = undefined,
    rows_host: cuda.HostBuffer = undefined,
    sized: bool = false,

    /// The model in `dir` on device `device`, its scratch not yet sized.
    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, device: c_int) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .d = undefined, .ctx = undefined, .s = undefined, .m = undefined, .l = undefined };
        e.d = try cuda.Driver.open();
        errdefer e.d.close();
        e.ctx = try cuda.Context.init(&e.d, device);
        errdefer e.ctx.deinit();
        e.s = try cuda.Stream.init(&e.d, true);
        errdefer e.s.deinit();
        const before = cuda.usage(false).device;
        e.m = try Model.load(gpa, io, &e.d, dir);
        errdefer e.m.deinit();
        e.weights = @intCast(cuda.usage(false).device - before);
        e.l = try lm.Lm.init(gpa, &e.d, e.s, e.m.spec);
        errdefer e.l.deinit();
        try e.l.prepare(e.m.spec); // in place: a plan points at e.l's library
        return e;
    }

    /// Device bytes `size(window)` will hold beside the weights.
    pub fn scratchBytes(e: *Engine, window: usize) !usize {
        return lm.Scratch.of(e.m.spec, window, max_inputs).total;
    }

    /// Allocates the scratch for passes of `window` tokens; refuses when the bytes held differ from the count.
    pub fn size(e: *Engine, window: usize) !void {
        if (e.sized) return error.AlreadySized;
        if (window == 0 or window > e.m.spec.max_positions) {
            std.log.err("qwen3: a {d}-token window is not one this model takes (1 to {d})", .{ window, e.m.spec.max_positions });
            return error.BadWindow;
        }
        const sc = lm.Scratch.of(e.m.spec, window, max_inputs);
        const before = cuda.usage(false).device;
        e.scratch = try cuda.DeviceBuffer.alloc(&e.d, sc.total);
        errdefer e.scratch.free();
        const cap = std.mem.alignForward(usize, window, lm.chunk_keys);
        e.a = sc.at(e.scratch.ptr);
        try e.positions(cap);
        try e.rope(cap);
        e.ids_host = try cuda.HostBuffer.alloc(&e.d, 4 * window);
        errdefer e.ids_host.free();
        e.rows_host = try cuda.HostBuffer.alloc(&e.d, 2 * max_inputs * e.m.spec.hidden);
        errdefer e.rows_host.free();
        const held = cuda.usage(false).device - before;
        if (held != sc.total) {
            std.log.err("qwen3: sizing held {d} device bytes, the count says {d}", .{ held, sc.total });
            return error.SizeMismatch;
        }
        // every plan's algorithm checked at one row and at the window now, not on a request
        try e.l.g.checkRows(1);
        try e.l.g.checkRows(window);
        e.cap = cap;
        e.window = window;
        e.sized = true;
    }

    /// Each input's final-normed last row as fp32 into `out`: after a shared `trunk` of tokens, inputs end at `ends`.
    pub fn run(e: *Engine, tokens: []const u32, trunk: u32, ends: []const u32, out: []f32) !void {
        const s = e.m.spec;
        if (!e.sized or tokens.len == 0 or tokens.len > e.window or ends.len == 0 or ends.len > max_inputs or
            ends[ends.len - 1] != tokens.len or out.len != ends.len * s.hidden) return error.InvalidBatch;
        var begin: u32 = trunk;
        for (ends) |x| {
            if (x <= begin) return error.InvalidBatch;
            begin = x;
        }
        for (tokens) |t| if (t >= s.vocab) return error.InvalidBatch;
        const ids = e.ids_host.slice(u32)[0..tokens.len];
        @memcpy(ids, tokens);
        try store.view(&e.d, e.a.ids, 4 * tokens.len).uploadAsync(0, std.mem.sliceAsBytes(ids), e.s.handle);
        try lm.Lm.forward(&e.l, &e.m, e.a, e.cap, trunk, ends);
        const rows = e.rows_host.slice(u16)[0 .. ends.len * s.hidden];
        try store.view(&e.d, e.a.rows, 2 * rows.len).downloadAsync(0, std.mem.sliceAsBytes(rows), e.s.handle);
        try e.s.synchronize();
        // the pass pools BF16 values: each row is the widened bits, as the HIP engine gives them
        for (out, rows) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    }

    pub fn deinit(e: *Engine) void {
        e.s.synchronize() catch {};
        if (e.sized) {
            e.rows_host.free();
            e.ids_host.free();
            e.scratch.free();
        }
        e.l.deinit();
        e.m.deinit();
        e.s.deinit();
        e.ctx.deinit();
        e.d.close();
        e.gpa.destroy(e);
    }

    /// Positions 0 to cap - 1: each row's rotation angle, and the first position a row block's chunks read.
    fn positions(e: *Engine, cap: usize) !void {
        const pos = try e.gpa.alloc(i32, cap);
        defer e.gpa.free(pos);
        for (pos, 0..) |*x, i| x.* = @intCast(i);
        try store.view(&e.d, e.a.pos, cap * 4).upload(0, std.mem.sliceAsBytes(pos));
    }

    /// The RoPE table [cap, head_dim] of fp32 cosines and sines, each half of a head rotating by the same angle.
    fn rope(e: *Engine, cap: usize) !void {
        const hd = e.m.spec.head_dim;
        const theta: f64 = e.m.spec.theta;
        const cs = try e.gpa.alloc(f32, cap * hd);
        defer e.gpa.free(cs);
        const sn = try e.gpa.alloc(f32, cap * hd);
        defer e.gpa.free(sn);
        var i: usize = 0;
        while (i < hd / 2) : (i += 1) {
            const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd)));
            var p: usize = 0;
            while (p < cap) : (p += 1) {
                const angle = @as(f64, @floatFromInt(p)) * inv;
                const c: f32 = @floatCast(@cos(angle));
                const s: f32 = @floatCast(@sin(angle));
                for ([_]usize{ i, i + hd / 2 }) |col| {
                    cs[p * hd + col] = c;
                    sn[p * hd + col] = s;
                }
            }
        }
        try store.view(&e.d, e.a.cos, cs.len * 4).upload(0, std.mem.sliceAsBytes(cs));
        try store.view(&e.d, e.a.sin, sn.len * 4).upload(0, std.mem.sliceAsBytes(sn));
    }
};

test "the scratch the count reports is the block size, and its sections never share a unit" {
    const spec: config.Spec = .{ .layers = 1, .hidden = 128, .heads = 4, .kv_heads = 2, .head_dim = 128, .inter = 256, .vocab = 32, .eps = 1e-6, .theta = 1e6, .max_positions = 4096, .tied = false };
    const sc = lm.Scratch.of(spec, 2048, 4);
    try std.testing.expect(sc.total >= sc.rows + 2 * 4 * spec.hidden);
    try std.testing.expect(sc.po < sc.pm and sc.pm < sc.pl and sc.pl < sc.xs and sc.xs < sc.rows and sc.rows < sc.total);
    try std.testing.expectEqual(@as(usize, 0), sc.ids % store.unit);
    // two windows: the sections only grow
    const wide = lm.Scratch.of(spec, 4096, 4);
    try std.testing.expect(wide.total > sc.total and wide.x > sc.x and wide.po > sc.po);
}
