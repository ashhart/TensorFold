//! One independent binary64 attention formula checks the native split-softmax compositions on both backends.
const std = @import("std");
const core = @import("core");
const ct = core.contract;
const c = ct.spec;
pub const Control = @import("qsa_control_source.zig").Control;
pub const heads = 24;
pub const kv = 4;
pub const width = 256;
pub const scale: f32 = 0.0625;
pub const Fixture = struct {
    rows: usize,
    keys: usize,
    capacity: usize,
    sparse: bool,
    stress: bool,
    query: []u16,
    key: []u16,
    value: []u16,
    ids: []i32,
    nk: []i32,
    spr: []i32,
    pub fn make(a: std.mem.Allocator, shape: c.Shape, fill: ct.gates.Fill, seed: u64) !Fixture {
        const capacity = shape.k * (if (shape.n == 2) @as(usize, 2) else 1);
        const f: Fixture = .{ .rows = shape.rows, .keys = shape.k, .capacity = capacity, .sparse = shape.n == 2, .stress = fill == .stress, .query = try a.alloc(u16, shape.rows * heads * width), .key = try a.alloc(u16, capacity * kv * width), .value = try a.alloc(u16, capacity * kv * width), .ids = try a.alloc(i32, shape.rows * shape.k), .nk = try a.alloc(i32, shape.rows), .spr = try a.alloc(i32, shape.rows) };
        @memset(f.nk, @intCast(f.keys));
        @memset(f.spr, @intFromBool(f.sparse));
        var random = std.Random.DefaultPrng.init(seed);
        for (f.query) |*v| v.* = c.toBf16(if (f.stress) 0 else random.random().float(f32) * 2 - 1);
        for (f.key) |*v| v.* = c.toBf16(if (f.stress) 0 else random.random().float(f32) * 2 - 1);
        for (0..f.capacity) |j| for (0..kv) |h| for (0..width) |d| {
            const half = f.keys / 2;
            const ordinal = if (f.sparse) j / 2 else j;
            const at = if (ordinal < half) ordinal else ordinal - half;
            const positive: f32 = if (at % 2 == 0) 127 else 16256;
            const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
            const stress_value = if (ordinal < half) positive else if (ordinal < 2 * half) residual - positive else 0;
            const v = if (f.sparse and j % 2 == 0) 32 else if (f.stress) stress_value else random.random().float(f32) * 2 - 1;
            f.value[(j * kv + h) * width + d] = if (fill == .nonfinite and d < 3) ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[d] else c.toBf16(v);
        };
        for (0..f.rows) |r| for (0..f.keys) |j| {
            f.ids[r * f.keys + j] = @intCast(2 * (f.keys - 1 - j) + 1);
        };
        return f;
    }
    pub fn evaluate(f: Fixture, a: std.mem.Allocator, want: []f64, tolerance: []f64) !void {
        const q = try a.alloc(f64, width);
        defer a.free(q);
        const k = try a.alloc(f64, f.keys * width);
        defer a.free(k);
        const v = try a.alloc(f64, k.len);
        defer a.free(v);
        const ids = try a.alloc(usize, f.keys);
        defer a.free(ids);
        for (0..f.rows) |r| for (0..heads) |h| {
            const kh = h / (heads / kv);
            for (q, 0..) |*x, d| x.* = c.fromBf16(f.query[(r * heads + h) * width + d]);
            for (0..f.keys) |j| {
                ids[j] = j;
                const source: usize = if (f.sparse) @intCast(f.ids[r * f.keys + j]) else j;
                for (0..width) |d| {
                    k[j * width + d] = c.fromBf16(f.key[(source * kv + kh) * width + d]);
                    v[j * width + d] = c.fromBf16(f.value[(source * kv + kh) * width + d]);
                }
            }
            const at = (r * heads + h) * width;
            core.attention_ref.head(q, k, v, ids, width, scale, want[at..][0..width], tolerance[at..][0..width]);
        };
    }
    pub fn simulate(f: Fixture, a: std.mem.Allocator, acc: c.Accum, order: ct.rows.Order, out: []f64) !void {
        if (!f.stress) return error.QsaStressRequiresEqualScores;
        const terms = try a.alloc(f64, f.keys);
        defer a.free(terms);
        const ones = try a.alloc(f64, f.keys);
        defer a.free(ones);
        @memset(ones, 1);
        const l = ct.rows.sumIn(acc, order, ones);
        for (0..f.rows) |r| for (0..heads) |h| for (0..width) |d| {
            for (terms, 0..) |*v, j| {
                const key: usize = if (f.sparse) @intCast(f.ids[r * f.keys + j]) else j;
                v.* = c.fromBf16(f.value[(key * kv + h / (heads / kv)) * width + d]);
            }
            out[(r * heads + h) * width + d] = ct.rows.sumIn(acc, order, terms) / l;
        };
    }
};
pub fn Case(comptime contract: type, comptime B: type) type {
    const g = contract.gates;
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        rig: B.Rig,
        backend: B,
        control: Control = .production,
        label: [160]u8 = undefined,
        arena: ?std.heap.ArenaAllocator = null,
        fixture: Fixture = undefined,
        repeat: u32 = 0,
        pub fn deinit(s: *Self) void {
            s.rig.deinit();
            if (s.arena) |*a| a.deinit();
        }
        fn depth(shape: c.Shape) usize {
            return width + shape.k;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const name = std.fmt.bufPrint(&s.label, "{s} Qwen4 QSA attention parts+merge {t}", .{ B.name, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => .sums,
            }, .spec = .{
                .name = name,
                .accum = .f32,
                .depth = depth,
                .ops = 8,
                .out = .bf16,
                .nan = .{ .stated = "Finite Q/K and exceptional V columns retain binary64 classes; finite outputs obey attention_ref's BF16-P bound." },
                .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 32 } },
                .served = B.served,
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n < 1 or shape.n > 2 or shape.k == 0 or shape.k > 2048) return false;
            s.rig.reset();
            if (s.arena) |*a| a.deinit();
            s.arena = .init(s.a);
            s.repeat = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), shape, fill, seed);
            try s.backend.bind(&s.rig, s.fixture, s.control);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.rows * heads * width;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture, s.repeat);
            s.repeat += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            try s.fixture.evaluate(s.a, want, mag);
        }
        pub fn bounds(s: *Self, want: []const f64, tolerance: []const f64, allow: []f64) !void {
            for (want, tolerance, allow) |w, t, *limit| limit.* = if (s.fixture.stress) c.boundExact(s.kernel().spec, w) else t;
        }
        pub fn simulate(s: *Self, acc: c.Accum, order: contract.rows.Order, out: []f64) !void {
            try s.fixture.simulate(s.a, acc, order, out);
        }
        pub fn check(s: *Self, at: g.Gate) !?[]const u8 {
            if (at != .nonfinite and at != .shapes) return null;
            const want = try s.a.alloc(f64, s.count());
            defer s.a.free(want);
            const tolerance = try s.a.alloc(f64, want.len);
            defer s.a.free(tolerance);
            try s.reference(want, tolerance);
            const out = try s.rig.outputBytes();
            var exceptional: usize = 0;
            for (want, tolerance, 0..) |w, t, i| {
                const actual = c.read(.bf16, out, i);
                if (c.classOf(actual) != c.classOf(w)) return "attention output class differs from independent binary64";
                if (!std.math.isFinite(w)) {
                    exceptional += 1;
                    continue;
                }
                if (!(@abs(actual - w) <= t)) return "finite attention output leaves attention_ref's bound";
            }
            if (!try s.rig.intact()) return "attention wrote a guard";
            return if (at == .nonfinite and (exceptional == 0 or exceptional == want.len)) "attention fixture needs finite and all three exceptional classes" else null;
        }
    };
}
