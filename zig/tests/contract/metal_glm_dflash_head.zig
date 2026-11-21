//! GLM's DFlash2 drafter head under the contract: a block's rows through the target head on affine_mm's 32-row tile.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const mm = tf.affine_mm;

/// A planted fault in affine_mm.metal's text, which must turn gate 3 red at the head's row counts.
pub const Mutation = enum {
    none,
    tail,

    fn edit(m: Mutation) struct { from: []const u8, to: []const u8 } {
        return switch (m) {
            .none => unreachable,
            .tail => .{ .from = "rows = min(BM, simd_shuffle(last, ushort(owner)) - row);", .to = "rows = min(BM, simd_shuffle(last, ushort(owner)) - row - 1);" },
        };
    }
};

/// One registered head: the target head's bits (groups of 64) and a planted fault.
pub const Variant = struct { bits: u8, mutation: Mutation = .none };

/// GLM-5.3-Flash heads at 8 bits (oQ4e) and 4 bits (uniform packs), and one control.
pub const registered = [_]Variant{
    .{ .bits = 8 },
    .{ .bits = 4 },
    .{ .bits = 8, .mutation = .tail },
};

pub fn name(v: Variant) []const u8 {
    return switch (v.mutation) {
        .none => if (v.bits == 8) "metal glm dflash2 head q8 g64 (affine_mm gather, one matrix)" else "metal glm dflash2 head q4 g64 (affine_mm gather, one matrix)",
        .tail => "control: metal glm dflash2 head q8 g64, the block's last row dropped",
    };
}

/// A group's products in the tile op, its scale and bias, the groups in turn (affine_mm's own bound).
fn groupDepth64(s: c.Shape) usize {
    return 64 + s.k / 32 + 8;
}

const offset_inf = c.Known{ .gate = .nonfinite, .finding = "an infinite input gives NaN where MLX's s*sum(q x) + b*sum(x) keeps the infinity (the +128 code offset moves the bias to b - 128 s)" };

pub fn spec(v: Variant) c.Spec {
    const known = if (v.bits == 4 and v.mutation == .none) offset_inf else null;
    // a 16-row block heads its 15 mask rows (the 32-row tile), three streams' blocks 48 rows (the 64-row tile)
    const served = &[_]c.Shape{ .{ .rows = 15, .n = 154880, .k = 4096 }, .{ .rows = 48, .n = 154880, .k = 4096 } };
    return .{ .name = name(v), .accum = .f32, .depth = groupDepth64, .out = .bf16, .known = known, .tiles = .{ .{ .size = 32 }, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 128, .step = 64, .min = 64 } }, .served = served };
}

pub const Head = struct {
    rig: rig_.Rig,
    v: Variant,
    pipes: mm.Pipes,
    part: ?Affine = null,
    shape: c.Shape = undefined,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    sums: rig_.Ref = undefined,
    offsets: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Head {
        const plain = try mm.source(device, a, .{ .bits = v.bits, .group = 64 });
        defer a.free(plain);
        const src = if (v.mutation == .none) try a.dupe(u8, plain) else blk: {
            const e = v.mutation.edit();
            if (std.mem.indexOf(u8, plain, e.from) == null) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, plain, e.from, e.to);
        };
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: mm.Pipes = undefined;
        for (mm.names, 0..) |n, i| pipes[i] = try mtl.Pipeline.init(device, lib, n, false);
        return .{ .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .pipes = pipes };
    }

    pub fn deinit(p: *Head) void {
        p.drop();
        for (p.pipes) |x| x.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Head) void {
        if (p.part) |x| x.deinit();
        p.part = null;
        p.rig.reset();
    }

    pub fn kernel(p: *Head) g.Kernel {
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = if (p.v.mutation == .none) null else .shapes };
    }

    const vt = g.hooks(Head);

    pub fn prepare(p: *Head, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 64 != 0 or shape.k % 64 != 0) return false;
        p.shape = shape;
        const a = p.rig.a;
        var f = try Affine.init(a, shape, p.v.bits, 64);
        if (p.v.bits == 4) f.off = 128;
        p.part = f;
        try f.fill(fill, seed);
        const w = try a.alloc(u32, shape.n * shape.k * p.v.bits / 32);
        defer a.free(w);
        f.words(w);
        const offsets = [1]i32{0};
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(w));
        p.s = try p.rig.input(std.mem.sliceAsBytes(f.scales));
        p.b = try p.rig.input(std.mem.sliceAsBytes(f.biases));
        p.offsets = try p.rig.input(std.mem.sliceAsBytes(&offsets));
        p.sums = try p.rig.scratch(shape.rows * (shape.k / 64) * 4);
        p.out = try p.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    /// The backend's launch: row sums, then headGather (one matrix whose rows start at 0, a 32- or 64-row tile).
    pub fn run(p: *Head) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const x = .{ .buf = p.x.buf, .off = p.x.off };
        const xs = .{ .buf = p.sums.buf, .off = p.sums.off };
        mm.rowSums(e, p.pipes, 64, x, xs, @intCast(p.shape.rows), @intCast(p.shape.k));
        e.barrier();
        const R = tf.glm.metal_dflash_backend.Ref;
        const head = tf.glm.weights.Affine{ .w = .{ .buf = p.w.buf, .off = p.w.off }, .s = .{ .buf = p.s.buf, .off = p.s.off }, .b = .{ .buf = p.b.buf, .off = p.b.off }, .n = @intCast(p.shape.n), .k = @intCast(p.shape.k), .bits = p.v.bits, .group = 64 };
        tf.glm.metal_dflash_backend.headGather(e, p.pipes, R{ .buf = p.x.buf, .off = p.x.off }, R{ .buf = p.sums.buf, .off = p.sums.off }, head, R{ .buf = p.offsets.buf, .off = p.offsets.off }, R{ .buf = p.out.buf, .off = p.out.off }, @intCast(p.shape.rows));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Head) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Head, want: []f64, mag: []f64) !void {
        return p.part.?.reference(want, mag);
    }

    pub fn simulate(p: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.part.?.simulate(acc, order, out);
    }

    pub fn replaced(_: *Head) ![]const u8 {
        return error.NotAReplacement;
    }
};

/// The heads and the control into `kernels`, on tensor-unit Macs (the backend's head takes the row kernels elsewhere).
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    heads: std.ArrayList(*Head) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (registered) |v| {
            if (only) |text| if (std.mem.indexOf(u8, name(v), text) == null) continue;
            const p = try s.a.create(Head);
            errdefer s.a.destroy(p);
            p.* = try Head.init(s.a, s.device, s.queue, v);
            try s.heads.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.heads.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.heads.deinit(s.a);
    }
};
