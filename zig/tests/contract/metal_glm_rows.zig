//! Real GLM row/gather kernels under gates 1-4, with frozen predecessor and independent preactivation oracles.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const cases = @import("glm_rows_cases.zig");
const Fixture = @import("glm_rows_fixture.zig").Fixture;
const c = contract.spec;
const g = contract.gates;
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const gather = wide.gather;
pub const predecessor_source = @embedFile("baselines/glm_row_projection.metal.txt") ++ wide.glue_math ++ @embedFile("baselines/glm_rows.metal.txt") ++ @embedFile("baselines/glm_gather.metal.txt");
const Weights = struct { w: rig_.Ref, s: rig_.Ref, b: rig_.Ref };
fn ref(v: rig_.Ref) affine.Ref {
    return .{ .buf = v.buf, .off = v.off };
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: cases.Entry,
    name: []const u8,
    served: [23]c.Shape,
    pipe: mtl.Pipeline,
    base_pipe: ?mtl.Pipeline,
    f: ?Fixture = null,
    x: rig_.Ref = undefined,
    w: Weights = undefined,
    up: Weights = undefined,
    order: rig_.Ref = undefined,
    offsets: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    bytes: std.ArrayList(u8) = .empty,
    base_bytes: std.ArrayList(u8) = .empty,
    fn drop(p: *Probe) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        if (p.base_pipe) |b| b.deinit();
        p.rig.deinit();
        p.bytes.deinit(p.a);
        p.base_bytes.deinit(p.a);
        p.a.free(p.name);
    }
    const vt = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        v.intact = intact;
        v.check = null;
        break :blk v;
    };
    const nonlinear_vt = blk: {
        var v = vt;
        v.check = nonfinite;
        break :blk v;
    };
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = cases.spec(p.v, p.name, &p.served), .ctx = p, .vt = if (p.v.nonlinear()) &nonlinear_vt else &vt, .control = if (p.v.mode == .control) .shapes else null };
    }
    fn upload(p: *Probe, q: contract.affine.Affine) !Weights {
        const words = try p.a.alloc(u32, q.shape.n * q.shape.k * q.bits / 32);
        defer p.a.free(words);
        q.words(words);
        return .{ .w = try p.rig.input(std.mem.sliceAsBytes(words)), .s = try p.rig.input(std.mem.sliceAsBytes(q.scales)), .b = try p.rig.input(std.mem.sliceAsBytes(q.biases)) };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (!Fixture.takes(p.v, shape)) return false;
        const f = try Fixture.init(p.a, p.v, shape, fill, seed);
        p.f = f;
        const x = try p.a.alloc(u16, f.inputElements());
        defer p.a.free(x);
        f.packX(x);
        p.x = try p.rig.input(std.mem.sliceAsBytes(x));
        p.w = try p.upload(f.f);
        p.up = if (f.up) |q| try p.upload(q) else p.w;
        p.order = try p.rig.input(std.mem.sliceAsBytes(f.order));
        p.offsets = try p.rig.input(std.mem.asBytes(&f.offsets));
        p.y = try p.rig.output(f.outputElements() * 2);
        p.base = try p.rig.scratch(f.outputElements() * 2);
        return true;
    }
    fn weight(p: *Probe, up: bool) affine.Affine {
        const f = p.f.?;
        const q = if (up) f.up.? else f.f;
        const w = if (up) p.up else p.w;
        return .{ .w = ref(w.w), .s = ref(w.s), .b = ref(w.b), .n = @intCast(f.shape.n * f.heads), .k = @intCast(f.shape.k), .bits = q.bits, .group = @intCast(q.group) };
    }
    pub fn encode(p: *Probe, e: anytype, pipe: mtl.Pipeline, y: rig_.Ref, baseline: bool) void {
        const f = p.f.?;
        if (p.v.kind == .down or p.v.kind == .glu) {
            gather.encode(e, pipe, p.weight(false), if (p.v.kind == .glu) p.weight(true) else null, .{ .x = ref(p.x), .offsets = ref(p.offsets), .order = ref(p.order), .out = ref(y) }, @intCast(f.shape.rows), 4, 2, @intCast(f.ldo), 10);
        } else if (baseline) {
            e.setPipeline(pipe);
            inline for (.{ p.x, p.w.w, p.w.s, p.w.b }, 0..) |v, i| e.setBuffer(v.buf, v.off, i);
            e.setValue(wide.RowDims{ .k = @intCast(f.shape.k), .n = @intCast(f.shape.n), .group = @intCast(f.f.group), .rows = @intCast(f.shape.rows), .heads = @intCast(f.heads), .ldx = @intCast(f.ldx), .hsx = @intCast(f.shape.k), .ldo = @intCast(f.ldo), .hso = @intCast(f.hso), .hsw = @intCast(f.shape.n) }, 4);
            e.setBuffer(y.buf, y.off, 5);
            e.dispatchGroups(mtl.Size.of((f.shape.n + 7) / 8, f.heads, 1), mtl.Size.of(64, 1, 1));
        } else {
            var pipes: wide.Pipes = undefined;
            const at = std.mem.indexOfScalar(u8, &cases.bits, p.v.bits).?;
            pipes.rows[at] = pipe;
            pipes.rows_wide[at] = pipe;
            const io_: wide.Io = .{ .in = ref(p.x), .out = ref(y), .ldx = @intCast(f.ldx), .ldo = @intCast(f.ldo), .rows = @intCast(f.shape.rows) };
            if (f.heads == 1) wide.dense(&pipes, e, p.weight(false), io_) else wide.perHead(&pipes, e, p.weight(false), @intCast(f.heads), io_, @intCast(f.shape.k), @intCast(f.hso));
        }
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, p.pipe, p.y, false);
        try p.rig.finish(pass);
    }
    fn pack(p: *Probe, from: rig_.Ref, list: *std.ArrayList(u8)) ![]const u8 {
        list.clearRetainingCapacity();
        const f = p.f.?;
        const bytes = p.rig.bytes(from, f.outputElements() * 2);
        for (0..f.count()) |i| try list.appendSlice(p.a, bytes[2 * f.outputIndex(i) ..][0..2]);
        return list.items;
    }
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (!try intact(p)) return error.WroteOutsideOutputs;
        return p.pack(p.y, &p.bytes);
    }
    pub fn encodePredecessor(p: *Probe, e: anytype) !void {
        p.encode(e, p.base_pipe orelse return error.MissingPredecessor, p.base, true);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        const f = p.f.?;
        @memset(p.rig.bytes(p.base, f.outputElements() * 2), 0xff);
        const pass = p.rig.begin();
        try p.encodePredecessor(pass.e);
        try p.rig.finish(pass);
        return p.pack(p.base, &p.base_bytes);
    }
    fn intact(ctx: *anyopaque) !bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (!try p.rig.intact()) return false;
        const f = p.f.?;
        const bytes = p.rig.bytes(p.y, f.outputElements() * 2);
        for (0..f.shape.rows) |row| {
            var at: usize = 0;
            for (0..f.heads) |h| {
                for (bytes[2 * (row * f.ldo + at) .. 2 * (row * f.ldo + h * f.hso)]) |b| if (b != 0xff) return false;
                at = h * f.hso + f.shape.n;
            }
            for (bytes[2 * (row * f.ldo + at) .. 2 * ((row + 1) * f.ldo)]) |b| if (b != 0xff) return false;
        }
        return true;
    }
    fn nonfinite(ctx: *anyopaque, _: g.Gate) !?[]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const old = try p.replaced();
        const got = try output(p);
        for (0..p.count()) |i| if (c.classOf(c.read(.bf16, got, i)) != c.classOf(c.read(.bf16, old, i))) return "clipped SwiGLU classes differ from the original kernel";
        return null;
    }
    pub fn count(p: *Probe) usize {
        return p.f.?.count();
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.f.?.reference(want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.f.?.simulate(acc, order, out);
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    libs: [6]?mtl.Library = @splat(null),
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try cases.visit(s, only);
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        for (s.libs) |l| if (l) |lib| lib.deinit();
    }
    fn library(s: *Suite, v: cases.Entry, old: bool) !mtl.Library {
        const at: usize = if (old) 1 else if (v.mode == .control) (if (v.kind == .narrow or v.kind == .wide) 2 else 3) else if (v.kind == .glu and v.mode == .oracle) 4 else if (v.mode == .up_oracle) 5 else 0;
        if (s.libs[at]) |l| return l;
        const text = if (old) try s.a.dupe(u8, predecessor_source) else try cases.source(s.a, wide.gather_source, v);
        defer s.a.free(text);
        const lib = try mtl.Library.fromSource(s.device, text, mtl.CompileOptions.mlx());
        s.libs[at] = lib;
        return lib;
    }
    pub fn add(s: *Suite, v: cases.Entry, text: []const u8) !void {
        try s.probes.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const lib = try s.library(v, false);
        const pipe = try mtl.Pipeline.init(s.device, lib, v.symbol(), false);
        errdefer pipe.deinit();
        const needs_base = v.mode == .replacement or v.mode == .control;
        var base: ?mtl.Pipeline = null;
        errdefer if (base) |b| b.deinit();
        if (needs_base) {
            var original = v;
            if (v.kind == .wide) original.kind = .narrow;
            base = try mtl.Pipeline.init(s.device, try s.library(v, true), original.symbol(), false);
        }
        const name = try s.a.dupe(u8, text);
        errdefer s.a.free(name);
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .v = v, .name = name, .served = cases.served(v.kind), .pipe = pipe, .base_pipe = base };
        s.probes.appendAssumeCapacity(p);
        s.kernels.appendAssumeCapacity(p.kernel());
    }
};
