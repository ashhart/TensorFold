//! Independent row-pair scheduling keeps the shared projection's bytes and independent FP64 gates.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const tf = @import("tensorfold");
const affine = @import("metal_affine.zig");
const rig_ = @import("rig.zig");
const pairs = tf.lfm3.row_projection;
const c = ct.spec;
const g = ct.gates;
pub const Fault = enum { none, tail, rows };
pub const Proof = enum { fp64, bytes };

pub fn label(a: std.mem.Allocator, bits: u8, group: usize, proof: Proof, fault: Fault) ![]u8 {
    return std.fmt.allocPrint(a, "{s}LFM3 rowpairs q{d} g{d} {t} {t}", .{ if (fault == .none) "metal " else "control: ", bits, group, proof, fault });
}

pub const Probe = struct {
    inner: affine.Proj,
    pipelines: pairs.Pipelines,
    proof: Proof,
    fault: Fault,
    label: []u8,
    previous: rig_.Ref = undefined,
    served: [3]c.Shape = .{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 3, .n = 512, .k = 2048 }, .{ .rows = 65, .n = 64, .k = 7680 } },
    const vt = blk: {
        var hooks = g.hooks(affine.Proj);
        hooks.prepare = prepare;
        hooks.run = run;
        hooks.replaced = replaced;
        break :blk hooks;
    };
    fn parent(ctx: *anyopaque) *Probe {
        const inner: *affine.Proj = @ptrCast(@alignCast(ctx));
        return @fieldParentPtr("inner", inner);
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, group: usize, proof: Proof, fault: Fault) !Probe {
        const name = try label(a, bits, group, proof, fault);
        errdefer a.free(name);
        var inner = try affine.Proj.init(a, device, queue, .{ .kind = .row, .bits = bits, .group = group }, name);
        errdefer inner.deinit();
        const text = try pairs.source(a);
        defer a.free(text);
        const source = try std.mem.concat(a, u8, &.{ switch (fault) {
            .none => "",
            .tail => "#define L3_PROJECTION_BAD_TAIL 1\n",
            .rows => "#define L3_PROJECTION_BAD_ROWS 1\n",
        }, text });
        defer a.free(source);
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        var pipelines: pairs.Pipelines = undefined;
        var loaded: usize = 0;
        errdefer for (pipelines.plain[0..loaded]) |pipeline| pipeline.deinit();
        for (pairs.names, 0..) |symbol, i| {
            pipelines.plain[i] = try mtl.Pipeline.init(device, library, symbol, false);
            loaded += 1;
        }
        return .{ .inner = inner, .pipelines = pipelines, .proof = proof, .fault = fault, .label = name };
    }
    pub fn deinit(p: *Probe) void {
        p.pipelines.deinit();
        p.inner.deinit();
        p.inner.a.free(p.label);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var spec = affine.spec(p.inner.v, p.label);
        spec.served = &p.served;
        spec.replaces = if (p.proof == .bytes) "the unchanged shared MB2 row projection" else null;
        return .{ .ctx = &p.inner, .vt = &vt, .spec = spec, .control = if (p.fault == .none) null else .shapes };
    }
    fn prepare(ctx: *anyopaque, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const p = parent(ctx);
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        p.previous = try p.inner.rig.scratch(p.inner.f.?.outputs() * 2);
        return true;
    }
    fn run(ctx: *anyopaque) !void {
        const p = parent(ctx);
        const base = &p.inner;
        const shape = base.f.?.shape;
        const weights = tf.row_projection.Weights{ .w = base.w.buf, .w_off = base.w.off, .scales = base.s.buf, .s_off = base.s.off, .biases = base.b.buf, .b_off = base.b.off, .n = shape.n, .k = shape.k, .group = base.v.group, .bits = base.v.bits };
        const pass = base.rig.begin();
        try pairs.encode(pass.e, &p.pipelines, weights, base.x.buf, base.x.off, base.out.buf, base.out.off, shape.rows);
        try base.rig.finish(pass);
    }
    fn replaced(ctx: *anyopaque) ![]const u8 {
        const p = parent(ctx);
        const actual = p.inner.out;
        p.inner.out = p.previous;
        defer p.inner.out = actual;
        try p.inner.run();
        return p.inner.rig.bytes(p.previous, p.inner.f.?.outputs() * 2);
    }
};
