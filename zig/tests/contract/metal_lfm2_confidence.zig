//! LFM2's DSpark confidence logit (lfm2_dspark_confidence): a dot of a row and a W1 row with the head's 2,304 weights.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const hidden = 2048;
const rank = 256;
pub const name = "metal lfm2 dspark confidence";

/// rows: one stream's drafts (row i reads the block's row i and the W1 row of the token before draft i).
const shapes = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = hidden + rank }, .{ .rows = 4, .n = 1, .k = hidden + rank }, .{ .rows = 9, .n = 1, .k = hidden + rank } };

/// A thread's chain over every 256th input, a 32-way simd_sum tree, then the eight simdgroups' partials in order.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

pub fn spec() c.Spec {
    return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .served = &shapes };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    mutation: controls.Mutation,
    r: ?rows.Rows = null,
    args: [6]rig_.Ref = undefined, // x, W1 rows, ids, picks, the head's weights, the shape
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "lfm2_dspark_confidence"), .mutation = mutation };
    }

    pub fn deinit(p: *Probe) void {
        if (p.r) |r| r.deinit();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec();
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    /// Row i's first 2,048 inputs as the block's row i, its last 256 as W1 row i: the token before draft i is i.
    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        if (s.k != hidden + rank or s.rows == 0 or s.rows > 16) return false;
        if (p.r) |r| r.deinit();
        p.r = null;
        p.rig.reset();
        const r = try rows.Rows.init(p.a, .dot, s);
        p.r = r;
        r.fill(f, seed);
        const x = try p.a.alloc(u16, s.rows * hidden);
        defer p.a.free(x);
        const w1 = try p.a.alloc(u16, s.rows * rank);
        defer p.a.free(w1);
        for (0..s.rows) |i| {
            @memcpy(x[i * hidden ..][0..hidden], r.x[i * s.k ..][0..hidden]);
            @memcpy(w1[i * rank ..][0..rank], r.x[i * s.k + hidden ..][0..rank]);
        }
        // the block's pending id is token 0; the pick before draft i (i > 0) is token i, a flag word after each
        var picks: [32]u32 = @splat(0xdead);
        for (1..s.rows) |i| picks[2 * (i - 1)] = @intCast(i);
        const ids = [1]u32{0};
        const args = [5]u32{ @intCast(s.rows), 1, @intCast(s.rows), rank, hidden };
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(x));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(w1));
        p.args[2] = try p.rig.input(std.mem.asBytes(&ids));
        p.args[3] = try p.rig.input(std.mem.asBytes(&picks));
        p.args[4] = try p.rig.input(std.mem.sliceAsBytes(r.w));
        p.args[5] = try p.rig.input(std.mem.asBytes(&args));
        p.out = try p.rig.output(s.rows * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const n = p.r.?.shape.rows;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.args, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(p.out.buf, p.out.off, 6);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.r.?.outputs();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.r.?.reference(want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.r.?.simulate(acc, order, out);
    }
};
