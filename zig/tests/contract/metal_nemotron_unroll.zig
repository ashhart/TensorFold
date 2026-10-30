//! The decode partials also retain their pre-unroll bytes; their original independent FP64 suite stays registered.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const old = @import("metal_nemotron_attention.zig");
const r = @import("rig.zig");
const g = contract.gates;
const c = contract.spec;
const Probe = struct {
    inner: old.Probe,
    name: []const u8,
    rig: *r.Rig,
    baseline: mtl.Pipeline,
    base: r.Ref = undefined,
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try p.inner.prepare(shape, fill, seed)) return false;
        p.base = try p.rig.scratch(p.inner.out_bytes);
        return true;
    }
    pub fn run(p: *Probe) !void {
        try p.inner.run();
    }
    pub fn count(p: *Probe) usize {
        return p.inner.count();
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.inner.reference(want, mag);
    }
    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        try p.inner.bounds(want, mag, allow);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.inner.simulate(acc, order, out);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        const pipeline = p.inner.pipes[0];
        const out = p.inner.out;
        defer {
            p.inner.pipes[0] = pipeline;
            p.inner.out = out;
        }
        p.inner.pipes[0] = p.baseline;
        p.inner.out = p.base;
        try p.inner.run();
        return p.rig.bytes(p.base, p.inner.out_bytes);
    }
};
const vt = g.hooks(Probe);
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (s.device.generation() != 15) return;
        for (1..17) |tiles| {
            const name = try std.fmt.allocPrint(s.a, "metal nemotron attention partial_{d} unroll bytes", .{tiles});
            if (only) |o| if (std.mem.indexOf(u8, name, o) == null) {
                s.a.free(name);
                continue;
            };
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            var key: [32]u8 = undefined;
            const wanted = try std.fmt.bufPrint(&key, "attn_partial_{d}", .{tiles});
            const k = for (sources.nemotron.all) |k| {
                if (std.mem.eql(u8, k.key, wanted)) break k;
            } else return error.UnknownSource;
            const text = try tf.nemotron.simd_attention.layout(s.a, k.source);
            defer s.a.free(text);
            p.* = .{ .name = name, .inner = try old.Probe.init(s.a, s.device, s.queue, .decode, tiles, false), .rig = undefined, .baseline = try r.pipeline(s.device, text, k.function) };
            p.rig = &p.inner.rig;
            try s.probes.append(s.a, p);
            var spec = old.spec(.decode, tiles, false);
            spec.name = name;
            spec.replaces = "pre-unroll SIMD partials";
            try s.kernels.append(s.a, .{ .spec = spec, .ctx = p, .vt = &vt });
        }
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.inner.deinit();
            p.baseline.deinit();
            s.a.free(p.name);
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
