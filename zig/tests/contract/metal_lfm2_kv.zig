//! DSpark context-only KV: the canonical QK kernel's exact cache bytes, without discarded queries.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const controls = @import("metal_lfm2_controls.zig");

const hidden = 2048;
const heads = 8;
const hd = 64;
const width = heads * hd;
pub const name = "metal lfm2 dspark context kv";
const Args = extern struct { len: u32, cap: u32, eps: f32 = 1e-5, log2_theta: f32 };
// k is the first cache position; prefixes and three trailing positions must remain untouched.
const shapes = [_]c.Shape{ .{ .rows = 1, .n = width, .k = 0 }, .{ .rows = 9, .n = width, .k = 7 }, .{ .rows = 33, .n = width, .k = 777 }, .{ .rows = 2048, .n = width, .k = 3 } };

fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec() c.Spec {
    return .{ .name = name, .accum = .none, .depth = none, .out = .bf16, .served = &shapes, .replaces = "lfm2_qk's key and value outputs", .nan = .{ .stated = "the canonical QK cache bytes, including NaNs and copied infinities; other heads and positions unchanged" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the unchanged fp32 head helper, exact bf16 cache bytes against canonical QK" } };
}

fn value(i: usize, seed: u64, scale: f32) u16 {
    const h = (@as(u64, i) *% 0x9e3779b97f4a7c15 +% seed *% 0xbf58476d1ce4e5b9) >> 40;
    return c.toBf16((@as(f32, @floatFromInt(h % 2048)) / 1024 - 1) * scale);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    canonical: mtl.Pipeline,
    mutation: controls.Mutation,
    rows: usize = 0,
    start: usize = 0,
    cache_n: usize = 0,
    args: [4]rig_.Ref = undefined,
    q: rig_.Ref = undefined,
    queries: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    prior: []u16 = &.{},
    golden: []u8 = &.{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, mutation: controls.Mutation) !Probe {
        const text = try controls.source(a, mutation);
        defer if (mutation != .none) a.free(text);
        const pipe = try rig_.pipeline(device, text, "lfm2_dspark_kv");
        errdefer pipe.deinit();
        const canonical = try rig_.pipeline(device, sources.lfm2, "lfm2_qk");
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .canonical = canonical, .mutation = mutation };
    }

    fn drop(p: *Probe) void {
        p.a.free(p.prior);
        p.a.free(p.golden);
        p.prior = &.{};
        p.golden = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.canonical.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec();
        if (p.mutation != .none) s.name = p.mutation.name();
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.mutation.gate() };
    }

    pub fn prepare(p: *Probe, s: c.Shape, f: g.Fill, seed: u64) !bool {
        p.drop();
        if (s.n != width or s.rows == 0 or s.rows > 2048 or s.k > 131072) return false;
        p.rows = s.rows;
        p.start = s.k;
        const cap = s.k + s.rows + 3;
        p.cache_n = cap * width;
        const input = try p.a.alloc(u16, 2 * s.rows * width + hd);
        defer p.a.free(input);
        for (input, 0..) |*v, i| v.* = value(i, seed, if (f == .stress) 16 else 2);
        if (f == .nonfinite) {
            input[7] = 0x7fc0;
            input[hd + 11] = 0x7f80;
            input[s.rows * width + 5] = 0xff80;
            input[s.rows * width + hd + 3] = 0x7fc0;
        }
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(input[0 .. s.rows * width]));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(input[s.rows * width ..][0 .. s.rows * width]));
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(input[2 * s.rows * width ..]));
        const params = Args{ .len = @intCast(s.k), .cap = @intCast(cap), .log2_theta = std.math.log2(@as(f32, 1e7)) };
        p.args[3] = try p.rig.input(std.mem.asBytes(&params));
        p.q = try p.rig.scratch(s.rows * hidden * 2);
        @memset(p.rig.bytes(p.q, s.rows * hidden * 2), 0);
        p.queries = try p.rig.scratch(s.rows * hidden * 2);
        p.prior = try p.a.alloc(u16, p.count());
        for (p.prior, 0..) |*v, i| v.* = value(i, seed +% 3, 1);
        const baseline = try p.rig.scratch(p.count() * 2);
        try p.encode(baseline, true);
        p.golden = try p.a.dupe(u8, p.rig.bytes(baseline, p.count() * 2));
        p.y = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return 2 * p.cache_n;
    }

    fn encode(p: *Probe, out: rig_.Ref, baseline: bool) !void {
        @memcpy(p.rig.bytes(out, p.count() * 2), std.mem.sliceAsBytes(p.prior));
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(if (baseline) p.canonical else p.pipe);
        if (baseline) {
            e.setBuffer(p.q.buf, p.q.off, 0);
            e.setBuffer(p.args[0].buf, p.args[0].off, 1);
            e.setBuffer(p.args[1].buf, p.args[1].off, 2);
            e.setBuffer(p.args[2].buf, p.args[2].off, 3);
            e.setBuffer(p.args[2].buf, p.args[2].off, 4);
            e.setBuffer(p.args[3].buf, p.args[3].off, 5);
            e.setBuffer(p.queries.buf, p.queries.off, 6);
        } else for (p.args, 0..) |arg, i| e.setBuffer(arg.buf, arg.off, i);
        const first: usize = if (baseline) 7 else 4;
        e.setBuffer(out.buf, out.off, first);
        e.setBuffer(out.buf, out.off + p.cache_n * 2, first + 1);
        e.dispatchGroups(mtl.Size.of((if (baseline) @as(usize, 32) else 0) + heads, p.rows, 1), mtl.Size.of(32, 1, 1));
        return p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        return p.encode(p.y, false);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = c.fromBf16(std.mem.readInt(u16, p.golden[2 * i ..][0..2], .little));
            m.* = @abs(w.*);
        }
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        return p.golden;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const actual = p.rig.bytes(p.y, p.count() * 2);
        if (!std.mem.eql(u8, p.golden, actual)) return "context KV differs from canonical QK";
        const prior = std.mem.sliceAsBytes(p.prior);
        const cap = p.cache_n / width;
        for (0..2 * heads) |h| {
            const base = h * cap * hd * 2;
            const first = base + p.start * hd * 2;
            const end = first + p.rows * hd * 2;
            const limit = base + cap * hd * 2;
            if (!std.mem.eql(u8, prior[base..first], actual[base..first]) or !std.mem.eql(u8, prior[end..limit], actual[end..limit])) return "context KV changes an untouched cache position";
        }
        const values = p.rig.bytes(p.args[1], p.rows * width * 2);
        for (0..heads) |h| for (0..p.rows) |row| {
            const from = (row * heads + h) * hd * 2;
            const to = (p.cache_n + (h * cap + p.start + row) * hd) * 2;
            if (!std.mem.eql(u8, values[from..][0 .. hd * 2], actual[to..][0 .. hd * 2])) return "context V cache is not an exact copy of its input";
        };
        return null;
    }
};
