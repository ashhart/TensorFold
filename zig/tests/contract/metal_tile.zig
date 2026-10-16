//! core/tile_qmm (#821) under the contract on Metal: the 32x32 prompt tiles and their decode twin, at 4, 6 and 8 bits.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const tile = tf.tile_qmm;

pub const Kind = enum { tiles, twin };

/// A group's 64 products in MMA steps, two fmas a group (its scale, its bias on the group's input sum), groups in turn.
fn tileDepth(s: c.Shape) usize {
    return 64 + 2 * (s.k / 64) + 8;
}

/// LFM2.5-24B-A2B's expert projections, which #822 will run on the tiles: gate or up, 1,536 from 2,048, and down.
const prompt_rows = [_]c.Shape{ .{ .rows = 64, .n = 1536, .k = 2048 }, .{ .rows = 33, .n = 2048, .k = 1536 } };
const decode_rows = [_]c.Shape{ .{ .rows = 32, .n = 1536, .k = 2048 }, .{ .rows = 5, .n = 2048, .k = 1536 } };

pub fn spec(kind: Kind, name: []const u8) c.Spec {
    const k_tile = c.Tile{ .size = 128, .step = 64, .min = 64 };
    return switch (kind) {
        .tiles => .{ .name = name, .accum = .f32, .depth = tileDepth, .out = .bf16, .tiles = .{ .{ .size = 32 }, .{ .size = 32, .step = 32, .min = 32 }, k_tile }, .served = &prompt_rows },
        .twin => .{ .name = name, .accum = .f32, .depth = tileDepth, .out = .bf16, .replaces = "the tiles", .tiles = .{ .{ .size = 4 }, .{ .size = 32, .step = 32, .min = 32 }, k_tile }, .served = &decode_rows },
    };
}

pub const Tile = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    bits: u8,
    name: []const u8,
    p: *const tile.Pipelines,
    f: ?Affine = null,
    /// The tiles' group sums, outside the rig: encodeSums writes them from the buffer's start.
    sums: ?mtl.Buffer = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, p: *const tile.Pipelines, kind: Kind, bits: u8, name: []const u8) Tile {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .bits = bits, .name = name, .p = p };
    }

    pub fn deinit(t: *Tile) void {
        t.drop();
        t.rig.deinit();
    }

    fn drop(t: *Tile) void {
        if (t.f) |f| f.deinit();
        t.f = null;
        if (t.sums) |b| b.deinit();
        t.sums = null;
        t.rig.reset();
    }

    pub fn kernel(t: *Tile) g.Kernel {
        return .{ .spec = spec(t.kind, t.name), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Tile);

    pub fn prepare(t: *Tile, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        if (shape.rows == 0 or shape.n % tile.tile != 0 or shape.n == 0 or shape.k % 64 != 0 or shape.k == 0) return false;
        if (shape.k > 64 * 192 or (t.kind == .twin and shape.rows > tile.twin_rows)) return false;
        var f = try Affine.init(t.a, shape, t.bits, 64);
        t.f = f;
        try f.fill(fill, seed);
        const words = try t.a.alloc(u32, shape.n * shape.k * t.bits / 32);
        defer t.a.free(words);
        f.words(words);
        t.x = try t.rig.input(std.mem.sliceAsBytes(f.x));
        t.w = try t.rig.input(std.mem.sliceAsBytes(words));
        t.s = try t.rig.input(std.mem.sliceAsBytes(f.scales));
        t.b = try t.rig.input(std.mem.sliceAsBytes(f.biases));
        t.sums = try t.rig.device.buffer(shape.rows * (shape.k / 64) * 4, mtl.ResourceOptions.shared);
        if (t.kind == .twin) t.base = try t.rig.scratch(f.outputs() * 2);
        t.out = try t.rig.output(f.outputs() * 2);
        return true;
    }

    fn matrix(t: *const Tile) tile.Matrix {
        const f = t.f.?;
        return .{ .w = t.w.buf, .w_off = t.w.off, .scales = t.s.buf, .s_off = t.s.off, .biases = t.b.buf, .b_off = t.b.off, .n = f.shape.n, .k = f.shape.k, .bits = t.bits };
    }

    /// The tiles on the prepared rows into `y`, their group sums first.
    fn tiles(t: *Tile, y: rig_.Ref) !void {
        const shape = t.f.?.shape;
        const pass = t.rig.begin();
        tile.encodeSums(pass.e, t.p, shape.k, t.x.buf, t.x.off, t.sums.?, shape.rows);
        try tile.encode(pass.e, t.p, t.matrix(), t.x.buf, t.x.off, t.sums.?, 0, y.buf, y.off, shape.rows);
        try t.rig.finish(pass);
    }

    pub fn run(t: *Tile) !void {
        if (t.kind == .tiles) return t.tiles(t.out);
        const pass = t.rig.begin();
        try tile.encodeTwin(pass.e, t.p, t.matrix(), t.x.buf, t.x.off, t.out.buf, t.out.off, t.f.?.shape.rows);
        try t.rig.finish(pass);
    }

    /// The tiles' bytes on the same inputs: the twin must equal them bit for bit.
    pub fn replaced(t: *Tile) ![]const u8 {
        if (t.kind != .twin) return error.NotAReplacement;
        const len = t.f.?.outputs() * 2;
        @memset(t.rig.bytes(t.base, len), 0xff);
        try t.tiles(t.base);
        return t.rig.bytes(t.base, len);
    }

    pub fn count(t: *Tile) usize {
        return t.f.?.outputs();
    }

    pub fn reference(t: *Tile, want: []f64, mag: []f64) !void {
        return t.f.?.reference(want, mag);
    }

    pub fn simulate(t: *Tile, acc: c.Accum, order: g.Order, out: []f64) !void {
        return t.f.?.simulate(acc, order, out);
    }
};
