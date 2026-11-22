//! Raw prompt taps compare direct planar-to-ring stores with the old128-row gather and CPU ring copies.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
fn depth(_: c.Shape) usize {
    return 0;
}
const shapes = [_]c.Shape{ .{ .rows = 1, .n = 2048, .k = 32 }, .{ .rows = 16, .n = 2048, .k = 32 }, .{ .rows = 17, .n = 2048, .k = 32 }, .{ .rows = 128, .n = 2048, .k = 32 }, .{ .rows = 129, .n = 2048, .k = 32 }, .{ .rows = 512, .n = 2048, .k = 32 }, .{ .rows = 2048, .n = 2048, .k = 32 }, .{ .rows = 129, .n = 16, .k = 32 }, .{ .rows = 17, .n = 2048, .k = 5120 } };
const Probe = struct {
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    old: mtl.Pipeline,
    planes: u32,
    control: bool,
    shape: c.Shape = undefined,
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    kept: rig_.Ref = undefined,
    temp: rig_.Ref = undefined,
    base: []u8 = &.{},
    start: u64 = (1 << 33) + 2045,
    fn init(a: std.mem.Allocator, d: mtl.Device, queue: mtl.Queue, planes: u32, control: bool) !Probe {
        const source = if (control) try std.mem.replaceOwned(u8, a, q.tap_prompt_source, "(p.start + row) % p.window", "(p.start + row + 1) % p.window") else try a.dupe(u8, q.tap_prompt_source);
        defer a.free(source);
        const pipe = try rig_.pipeline(d, source, "q27_tap_ring");
        errdefer pipe.deinit();
        return .{ .rig = .{ .a = a, .device = d, .queue = queue }, .pipe = pipe, .old = try rig_.pipeline(d, q.tap_prompt_source, "q27_tap_gather"), .planes = planes, .control = control };
    }
    fn deinit(p: *Probe) void {
        p.rig.a.free(p.base);
        p.pipe.deinit();
        p.old.deinit();
        p.rig.deinit();
    }
    const vt = g.hooks(Probe);
    fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null, .spec = .{ .name = if (p.control) "control: Qwen27 tap ring shifted slot" else if (p.planes == 8) "metal Qwen27 tap ring eight planes" else "metal Qwen27 tap ring five planes", .out = .raw, .accum = .none, .depth = depth, .nan = .{ .stated = "every BF16 word including NaN payloads and signed zero equals old gather plus CPU ring copies; untouched slots unchanged" }, .replaces = "q27_tap_gather in128-row chunks then CPU Ring.absorb", .served = &shapes } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        p.rig.reset();
        p.rig.a.free(p.base);
        p.base = &.{};
        if (shape.rows == 0 or shape.rows > 2048 or shape.n == 0 or shape.k == 0) return false;
        p.shape = shape;
        const stride = shape.k * p.planes * 2;
        const raw = try p.rig.a.alloc(u16, shape.rows * shape.k * p.planes);
        defer p.rig.a.free(raw);
        var prng = std.Random.DefaultPrng.init(seed);
        for (raw) |*v| v.* = prng.random().int(u16);
        for (raw[0..4], [_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x8000 }) |*v, planted| v.* = planted;
        p.input = try p.rig.input(std.mem.sliceAsBytes(raw));
        p.out = try p.rig.output(shape.n * stride);
        p.kept = try p.rig.scratch(128 * 4);
        p.temp = try p.rig.scratch(128 * stride);
        p.base = try p.rig.a.alloc(u8, shape.n * stride);
        return true;
    }
    pub fn run(p: *Probe) !void {
        @memset(p.rig.bytes(p.out, p.count()), 0xff);
        const s = p.shape;
        const copied = @min(s.rows, s.n);
        const args = q.tap_prompt.Args{ .width = @intCast(s.k), .capacity = @intCast(s.rows), .planes = p.planes, .window = @intCast(s.n), .first = @intCast(s.rows - copied), .rows = @intCast(copied), .start = p.start };
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.input.buf, p.input.off, 0);
        pass.e.setValue(args, 1);
        pass.e.setBuffer(p.out.buf, p.out.off, 2);
        pass.e.dispatchThreads(mtl.Size.of(s.k * p.planes, copied, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        @memset(p.base, 0xff);
        const s = p.shape;
        const stride = s.k * p.planes * 2;
        var first: usize = 0;
        while (first < s.rows) {
            const rows = @min(128, s.rows - first);
            const ids = p.kept.buf.slice(u32, p.kept.buf.length() / 4)[p.kept.off / 4 ..][0..rows];
            for (ids, 0..) |*v, i| v.* = @intCast(first + i);
            const pass = p.rig.begin();
            pass.e.setPipeline(p.old);
            pass.e.setBuffer(p.input.buf, p.input.off, 0);
            pass.e.setBuffer(p.kept.buf, p.kept.off, 1);
            pass.e.setValue([4]u32{ @intCast(rows), @intCast(s.k), @intCast(s.rows), p.planes }, 2);
            pass.e.setBuffer(p.temp.buf, p.temp.off, 3);
            pass.e.dispatchThreads(mtl.Size.of(s.k * p.planes, rows, 1), mtl.Size.of(256, 1, 1));
            try p.rig.finish(pass);
            const bytes = p.rig.bytes(p.temp, rows * stride);
            for (0..rows) |i| {
                const slot: usize = @intCast((p.start + first + i) % s.n);
                @memcpy(p.base[slot * stride ..][0..stride], bytes[i * stride ..][0..stride]);
            }
            first += rows;
        }
        return p.base;
    }
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        return if (std.mem.eql(u8, got, want)) null else "ring bytes differ from old128-row gather and CPU absorb";
    }
    pub fn count(p: *Probe) usize {
        return p.shape.n * p.shape.k * p.planes * 2;
    }
    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [3]Probe,
    pub fn create(a: std.mem.Allocator, d: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.a = a;
        var built: usize = 0;
        errdefer for (s.probes[0..built]) |*p| p.deinit();
        for (&s.probes, 0..) |*p, i| {
            p.* = try Probe.init(a, d, queue, if (i == 1) 8 else 5, i == 2);
            built += 1;
        }
        for (&s.probes) |*p| {
            const k = p.kernel();
            if (only) |text| if (std.mem.indexOf(u8, k.spec.name, text) == null) continue;
            try kernels.append(a, k);
        }
        return s;
    }
    pub fn destroy(s: *Suite) void {
        const a = s.a;
        for (&s.probes) |*p| p.deinit();
        a.destroy(s);
    }
};
