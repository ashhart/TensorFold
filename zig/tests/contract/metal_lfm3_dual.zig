//! Two independently packed projections share one dispatch and must retain both serial outputs byte for byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const affine = @import("metal_affine.zig");
const rig_ = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
const dual = tf.lfm3.dual_projection;
pub const Fault = enum { none, second, tail };
pub fn label(a: std.mem.Allocator, bits: u8, group: usize, fault: Fault) ![]u8 {
    return std.fmt.allocPrint(a, "{s}LFM3 dual q{d} g{d} {t}", .{ if (fault == .none) "metal " else "control: ", bits, group, fault });
}
const Combined = struct {
    a: std.mem.Allocator,
    first: *rig_.Rig = undefined,
    second: *rig_.Rig = undefined,
    outs: std.ArrayList(u8) = .empty,
    ins: std.ArrayList(u8) = .empty,
    pub fn outputBytes(p: *Combined) ![]const u8 {
        p.outs.clearRetainingCapacity();
        try p.outs.appendSlice(p.a, try p.first.outputBytes());
        try p.outs.appendSlice(p.a, try p.second.outputBytes());
        return p.outs.items;
    }
    pub fn inputBytes(p: *Combined) ![]const u8 {
        p.ins.clearRetainingCapacity();
        try p.ins.appendSlice(p.a, try p.first.inputBytes());
        try p.ins.appendSlice(p.a, try p.second.inputBytes());
        return p.ins.items;
    }
    pub fn intact(p: *Combined) !bool {
        return try p.first.intact() and try p.second.intact();
    }
    pub fn overlap(p: *Combined) bool {
        return p.first.overlap() or p.second.overlap();
    }
};
pub const Probe = struct {
    first: affine.Proj,
    second: affine.Proj,
    rig: Combined,
    pipelines: dual.Pipelines,
    name: []u8,
    fault: Fault,
    previous: [2]rig_.Ref = undefined,
    base: std.ArrayList(u8) = .empty,
    served: [3]c.Shape = .{ .{ .rows = 1, .n = 7680, .k = 2048 }, .{ .rows = 3, .n = 512, .k = 2048 }, .{ .rows = 32, .n = 64, .k = 7680 } },
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, group: usize, fault: Fault) !Probe {
        const name = try label(a, bits, group, fault);
        errdefer a.free(name);
        var first = try affine.Proj.init(a, device, queue, .{ .kind = .row, .bits = bits, .group = group }, name);
        errdefer first.deinit();
        var second = try affine.Proj.init(a, device, queue, first.v, name);
        errdefer second.deinit();
        const text = try dual.source(a);
        defer a.free(text);
        const source = try std.mem.concat(a, u8, &.{ switch (fault) {
            .none => "",
            .second => "#define L3_DUAL_BAD_SECOND 1\n",
            .tail => "#define L3_DUAL_BAD_TAIL 1\n",
        }, text });
        defer a.free(source);
        const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: dual.Pipelines = undefined;
        var made: usize = 0;
        errdefer for (pipes.plain[0..made]) |p| p.deinit();
        for (dual.names, 0..) |symbol, i| {
            pipes.plain[i] = try mtl.Pipeline.init(device, lib, symbol, false);
            made += 1;
        }
        return .{ .first = first, .second = second, .rig = .{ .a = a }, .pipelines = pipes, .name = name, .fault = fault };
    }
    pub fn deinit(p: *Probe) void {
        p.first.deinit();
        p.second.deinit();
        p.pipelines.deinit();
        p.rig.outs.deinit(p.rig.a);
        p.rig.ins.deinit(p.rig.a);
        p.base.deinit(p.rig.a);
        p.rig.a.free(p.name);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var spec = affine.spec(p.first.v, p.name);
        spec.served = &p.served;
        spec.replaces = "two separate shared row projections on identical input rows";
        return .{ .ctx = p, .vt = &vt, .spec = spec, .control = if (p.fault == .none) null else .shapes };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try p.first.prepare(shape, fill, seed) or !try p.second.prepare(shape, fill, seed +% 101)) return false;
        p.rig.first = &p.first.rig;
        p.rig.second = &p.second.rig;
        @memcpy(p.second.f.?.x, p.first.f.?.x);
        @memcpy(p.second.rig.bytes(p.second.x, shape.rows * shape.k * 2), p.first.rig.bytes(p.first.x, shape.rows * shape.k * 2));
        const bytes = p.first.f.?.outputs() * 2;
        p.previous = .{ try p.first.rig.scratch(bytes), try p.second.rig.scratch(bytes) };
        return true;
    }
    fn weights(p: *affine.Proj) tf.row_projection.Weights {
        const s = p.f.?.shape;
        return .{ .w = p.w.buf, .w_off = p.w.off, .scales = p.s.buf, .s_off = p.s.off, .biases = p.b.buf, .b_off = p.b.off, .n = s.n, .k = s.k, .group = p.v.group, .bits = p.v.bits };
    }
    pub fn run(p: *Probe) !void {
        const pass = p.first.rig.begin();
        try dual.encode(pass.e, &p.pipelines, weights(&p.first), weights(&p.second), p.first.x.buf, p.first.x.off, p.first.out.buf, p.first.out.off, p.second.out.buf, p.second.out.off, p.first.f.?.shape.rows);
        try p.first.rig.finish(pass);
    }
    pub fn count(p: *Probe) usize {
        return p.first.count() * 2;
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const n = p.first.count();
        try p.first.reference(want[0..n], mag[0..n]);
        try p.second.reference(want[n..], mag[n..]);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = p.first.count();
        try p.first.simulate(acc, order, out[0..n]);
        try p.second.simulate(acc, order, out[n..]);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        p.base.clearRetainingCapacity();
        const outputs = [_]rig_.Ref{ p.first.out, p.second.out };
        p.first.out = p.previous[0];
        p.second.out = p.previous[1];
        defer {
            p.first.out = outputs[0];
            p.second.out = outputs[1];
        }
        try p.first.run();
        try p.second.run();
        for ([_]*affine.Proj{ &p.first, &p.second }, p.previous) |proj, ref| try p.base.appendSlice(p.rig.a, proj.rig.bytes(ref, proj.count() * 2));
        return p.base.items;
    }
};
