//! Nemotron's routed-expert pipelines under gates 1-4: w3 against float64, every other one against w3's bytes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const cases = @import("nemotron_experts_cases.zig");
const Fixture = @import("nemotron_experts_fixture.zig").Fixture;
const c = contract.spec;
const g = contract.gates;
const engine = tf.nemotron.kernels;

comptime {
    @setEvalBranchQuota(4000);
    std.debug.assert(engine.geometries.len == cases.geometries.len);
    for (engine.geometries, cases.geometries) |e, x| std.debug.assert(e[0] == x[0] and e[1] == x[1]);
    for (cases.entries) |v| {
        var found = false;
        for (engine.rows_names ++ engine.geo_names) |n| found = found or std.mem.eql(u8, n, v.symbol);
        std.debug.assert(found);
    }
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: cases.Entry,
    name: []const u8,
    served: [7]c.Shape,
    pipe: mtl.Pipeline,
    base_pipe: mtl.Pipeline,
    f: ?Fixture = null,
    bufs: [9]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,

    fn drop(p: *Probe) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        p.base_pipe.deinit();
        p.rig.deinit();
        p.a.free(p.name);
    }

    const vt_up = g.hooks(Probe);
    const vt_down = blk: {
        var v = vt_up;
        v.bounds = null;
        v.intervals = null;
        break :blk v;
    };
    const vt_up_replacement = blk: {
        var v = vt_up;
        v.check = check;
        break :blk v;
    };

    pub fn kernel(p: *Probe) g.Kernel {
        const vt = if (p.v.proj == .down) &vt_down else if (p.v.base()) &vt_up else &vt_up_replacement;
        return .{ .spec = cases.spec(p.v, p.name, &p.served), .ctx = p, .vt = vt, .control = if (p.v.fault != .none) .shapes else null };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (!Fixture.takes(p.v.proj, shape)) return false;
        p.f = try Fixture.init(p.a, p.v.proj, shape, fill, seed);
        const f = p.f.?;
        const words = try p.a.alloc(u32, f.q.codes.len / 8);
        defer p.a.free(words);
        f.words(words);
        const r = f.route;
        const parts = [_][]const u8{ std.mem.sliceAsBytes(f.q.x), std.mem.sliceAsBytes(r.uids), std.mem.sliceAsBytes(r.start), std.mem.sliceAsBytes(r.count), std.mem.sliceAsBytes(r.members), std.mem.asBytes(&r.ucount), std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.q.scales), std.mem.sliceAsBytes(f.q.biases) };
        for (&p.bufs, parts) |*b, bytes| b.* = try p.rig.input(bytes);
        p.out = try p.rig.output(f.count() * 2);
        p.base = try p.rig.scratch(f.count() * 2);
        return true;
    }

    /// One launch as layers.experts4 encodes it: (32 SG, N / (RPS SG), groups) threads in groups of 32 SG.
    pub fn encode(p: *Probe, e: anytype, pipe: mtl.Pipeline, out: rig_.Ref, rps: usize, sg: usize) void {
        const f = p.f.?;
        e.setPipeline(pipe);
        for (p.bufs, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
        e.setBuffer(out.buf, out.off, 9);
        e.dispatchThreads(mtl.Size.of(32 * sg, f.shape.n / (rps * sg), f.route.groups), mtl.Size.of(32 * sg, 1, 1));
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, p.pipe, p.out, p.v.rps, p.v.sg);
        try p.rig.finish(pass);
    }

    /// w3's bytes on the prepared inputs and tables: the window path every other pipeline must equal.
    pub fn replaced(p: *Probe) ![]const u8 {
        const n = p.f.?.count() * 2;
        @memset(p.rig.bytes(p.base, n), 0xff);
        const pass = p.rig.begin();
        p.encode(pass.e, p.base_pipe, p.base, 4, 4);
        try p.rig.finish(pass);
        return p.rig.bytes(p.base, n);
    }

    fn check(ctx: *anyopaque, _: g.Gate) !?[]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        const got = p.rig.bytes(p.out, p.f.?.count() * 2);
        if (!std.mem.eql(u8, got, try p.replaced())) return "a nonfinite input gives bytes other than tf_xup_w3's";
        return null;
    }

    pub fn count(p: *Probe) usize {
        return p.f.?.count();
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.f.?.reference(want, mag);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        try p.f.?.bounds(want, allow);
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        p.f.?.intervals(low, high);
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
    libs: [5]?mtl.Library = @splat(null),

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

    /// A source as kernels.load compiles it: the member-row file, each geometry file, or a control's one change.
    fn library(s: *Suite, v: cases.Entry) !mtl.Library {
        const at: usize = if (v.geo) (if (v.proj == .up) 1 else 2) else switch (v.fault) {
            .none => 0,
            .coupled => 3,
            .fp32_sums => 4,
        };
        if (s.libs[at]) |l| return l;
        const original = if (!v.geo) sources.nemotron_experts else if (v.proj == .up) engine.geo_up else engine.geo_down;
        const text = try cases.source(s.a, original, v.fault);
        defer s.a.free(text);
        s.libs[at] = try mtl.Library.fromSource(s.device, text, mtl.CompileOptions.mlx());
        return s.libs[at].?;
    }

    pub fn add(s: *Suite, v: cases.Entry, text: []const u8) !void {
        try s.probes.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const pipe = try mtl.Pipeline.init(s.device, try s.library(v), v.symbol, false);
        errdefer pipe.deinit();
        const base_entry: cases.Entry = .{ .proj = v.proj, .symbol = v.baseSymbol(), .rps = 4, .sg = 4, .mb = 3 };
        const base = try mtl.Pipeline.init(s.device, try s.library(base_entry), v.baseSymbol(), false);
        errdefer base.deinit();
        const name = try s.a.dupe(u8, text);
        errdefer s.a.free(name);
        const p = try s.a.create(Probe);
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .v = v, .name = name, .served = cases.served(v), .pipe = pipe, .base_pipe = base };
        s.probes.appendAssumeCapacity(p);
        s.kernels.appendAssumeCapacity(p.kernel());
    }
};
