//! rows.metal's row windows (tf_hy4_rows_r2_q4 and _q8) against MLX's affine formula in float64 at Hy4's widths.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

/// metal_linears.zig's Rows (rps 2): 2 columns a simdgroup and 4 simdgroups a group, 8 columns and 128 threads.
const columns = 8;
const threads = 128;
/// The checkpoint's MLX groups.
const group = 64;

/// The core row projection's bound (16-input pieces, a lane's in turn, simd_sum), whose tf_rp_rows every block calls.
fn depth(s: c.Shape) usize {
    return 16 + 3 * ((s.k + 511) / 512) + 8;
}

/// Every 8-row block's tails (3 to 15 rows) at the indexer's token keys, past the served passes' own counts.
const tails = [_]usize{ 3, 5, 10, 12, 14, 15 };
const shape_count = 6 + tails.len;

/// Where project() takes these windows (Rows.choose: 2 rows or more, portable 4- or 8-bit weights in groups of 64).
fn served(vocab: usize) [shape_count]c.Shape {
    const H = h.hidden;
    var out: [shape_count]c.Shape = undefined;
    // on the M5, the indexer's projections (always portable): token keys at DFly2's window, the edges' base
    out[0] = .{ .rows = h.window_rows, .n = 128, .k = H };
    out[1] = .{ .rows = 2, .n = 16, .k = H }; // head weights, a 2-row pass
    out[2] = .{ .rows = 4, .n = 2048, .k = 1024 }; // queries from the 1,024-wide query rank, a 4-row pass
    out[3] = .{ .rows = h.prompt_rows, .n = 2048, .k = 1024 };
    out[4] = .{ .rows = h.prompt_rows, .n = 128, .k = H };
    for (tails, 5..) |r, i| out[i] = .{ .rows = r, .n = 128, .k = H };
    // the vocabulary head, the widest projection a chip without tensor units sends here
    out[shape_count - 1] = .{ .rows = h.window_rows, .n = vocab, .k = H };
    return out;
}

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    bits: u8,
    fault: ?text.Fault,
    shapes: [shape_count]c.Shape,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    f: ?Affine = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn function(bits: u8) []const u8 {
        return if (bits == 4) "tf_hy4_rows_r2_q4" else "tf_hy4_rows_r2_q8";
    }
    pub fn base(bits: u8) []const u8 {
        return if (bits == 4) "metal hy4 rows tf_hy4_rows_r2_q4" else "metal hy4 rows tf_hy4_rows_r2_q8";
    }

    /// The pipeline Rows.load builds (on the wide core source on every chip), or a fault's edit of it.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, fault: ?text.Fault, vocab: usize) !Rows {
        const src = try text.rows(a, fault);
        defer a.free(src);
        const pipe = try text.pipeline(device, src, function(bits));
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .bits = bits, .fault = fault, .shapes = served(vocab) };
    }
    pub fn deinit(t: *Rows) void {
        t.drop();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn drop(t: *Rows) void {
        if (t.f) |f| f.deinit();
        t.f = null;
        t.rig.reset();
    }

    pub fn kernel(t: *Rows) g.Kernel {
        t.name = h.fl.named(base(t.bits), if (t.fault) |f| f.drops else null, &t.label);
        const s: c.Spec = .{
            .name = t.name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            // rows: the 8-row blocks and their 6/4, 2 and 1 tails; columns: a group's 8; K: a 512-wide pass, its tail
            .tiles = .{ .{ .size = 8, .min = 2 }, .{ .size = columns, .step = columns, .min = columns }, .{ .size = 512, .step = group, .min = group } },
            .served = &t.shapes,
        };
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.fault != null) .shapes else null };
    }
    const vt = blk: {
        var v = g.hooks(Rows);
        v.release = release;
        break :blk v;
    };
    fn release(ctx: *anyopaque) void {
        const t: *Rows = @ptrCast(@alignCast(ctx));
        t.drop();
    }

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        // Rows.choose and Weights.validate: 2 rows or more, whole column groups, whole groups of 64
        if (shape.rows < 2 or shape.n == 0 or shape.n % columns != 0 or shape.k == 0 or shape.k % group != 0) return false;
        t.f = try Affine.init(t.a, shape, t.bits, group);
        const f = t.f.?;
        try f.fill(fill, seed);
        t.x = try t.rig.input(std.mem.sliceAsBytes(f.x));
        // MLX's little-endian bit stream: at 8 bits each byte is a code, at 4 bits a byte holds two (the first low)
        if (t.bits == 8) t.w = try t.rig.input(f.codes) else {
            const words = try t.a.alloc(u8, f.codes.len / 2);
            defer t.a.free(words);
            for (words, 0..) |*v, i| v.* = f.codes[2 * i] | f.codes[2 * i + 1] << 4;
            t.w = try t.rig.input(words);
        }
        t.s = try t.rig.input(std.mem.sliceAsBytes(f.scales));
        t.b = try t.rig.input(std.mem.sliceAsBytes(f.biases));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    /// metal_dense.zig's project(): core affine.call's dims, n / 8 groups of 128 threads.
    pub fn run(t: *Rows) !void {
        const s = t.f.?.shape;
        const dims = [6]i32{ @intCast(s.k), @intCast(s.n), group, @intCast(s.rows), 1, 1 };
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for ([_]rig_.Ref{ t.x, t.w, t.s, t.b }, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(dims, 4);
        p.e.setBuffer(t.out.buf, t.out.off, 5);
        p.e.dispatchGroups(mtl.Size.of(s.n / columns, 1, 1), mtl.Size.of(threads, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return t.f.?.outputs();
    }
    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        return t.f.?.reference(want, mag);
    }
    pub fn simulate(t: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.f.?.simulate(acc, order, out);
    }
};
