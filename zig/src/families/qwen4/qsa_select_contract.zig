//! Independent numeric total-order sorting checks QSA membership, stable block emission and untouched padding.
const std = @import("std");
const ct = @import("core").contract;
const c = ct.spec;
pub const Control = @import("qsa_control_source.zig").Control;
pub const top = 128;
pub const ratio = 8;
pub const idw = (top + 1) * ratio;
pub const special = [_]u32{ 0xffc00007, 0xff800001, 0xff800000, 0x80000000, 0x00000000, 0x7f800000, 0x7f800001, 0x7fc00001, 0x7fc00007 };
const Entry = struct { score: f32, index: usize };
fn before(_: void, a: Entry, b: Entry) bool {
    const an = std.math.isNan(a.score);
    const bn = std.math.isNan(b.score);
    const ab: u32 = @bitCast(a.score);
    const bb: u32 = @bitCast(b.score);
    const negative_a = std.math.signbit(a.score);
    const negative_b = std.math.signbit(b.score);
    if (an or bn) {
        if (an and bn) {
            if (negative_a != negative_b) return !negative_a;
            const ap = ab & 0x7fffff;
            const bp = bb & 0x7fffff;
            if (ap != bp) return if (negative_a) ap < bp else ap > bp;
        } else return if (an) !negative_a else negative_b;
    } else {
        if (a.score != b.score) return a.score > b.score;
        if (negative_a != negative_b) return !negative_a;
    }
    return a.index < b.index;
}
pub const Fixture = struct {
    rows: usize,
    stride: usize,
    scores: []f32,
    complete: []i32,
    ends: []i32,
    want: []i32,
    pub fn make(a: std.mem.Allocator, shape: c.Shape, fill: ct.gates.Fill, seed: u64) !Fixture {
        const end = shape.n * ratio + shape.k - 1;
        const stride = (end + shape.rows - 1) / ratio + 7;
        const f: Fixture = .{ .rows = shape.rows, .stride = stride, .scores = try a.alloc(f32, shape.rows * stride), .complete = try a.alloc(i32, shape.rows), .ends = try a.alloc(i32, shape.rows), .want = try a.alloc(i32, shape.rows * idw) };
        @memset(f.want, -1);
        var random = std.Random.DefaultPrng.init(seed);
        const entries = try a.alloc(Entry, stride);
        const chosen = try a.alloc(bool, stride);
        for (0..shape.rows) |r| {
            const live = end + r;
            const blocks = live / ratio;
            f.ends[r] = @intCast(live);
            f.complete[r] = @intCast(blocks);
            for (f.scores[r * stride ..][0..stride], 0..) |*v, b| {
                v.* = if (b >= blocks) @bitCast(@as(u32, 0x7fffffff)) else if (fill == .stress) 0 else if (fill == .nonfinite) switch (r % 4) {
                    0 => @as(f32, 0),
                    1 => @bitCast(@as(u32, 0x7fc00000) + @as(u32, @intCast(b % 1024 + 1))),
                    2 => @bitCast(@as(u32, 0xffc00000) + @as(u32, @intCast(b % 1024 + 1))),
                    else => if (b % 2 == 0) @as(f32, 0) else -0.0,
                } else @floatFromInt(@as(i32, @intCast(random.random().uintLessThan(u32, 33))) - 16);
            }
            if (fill == .nonfinite) for (special, 0..) |bits, b| {
                if (b < blocks) f.scores[r * stride + b] = @bitCast(bits);
            };
            if (blocks <= top) continue;
            for (entries[0..blocks], f.scores[r * stride ..][0..blocks], 0..) |*entry, score, b| entry.* = .{ .score = score, .index = b };
            std.mem.sort(Entry, entries[0..blocks], {}, before);
            @memset(chosen, false);
            for (entries[0..top]) |entry| chosen[entry.index] = true;
            var at = r * idw;
            for (chosen[0..blocks], 0..) |yes, b| if (yes) {
                for (0..ratio) |j| {
                    f.want[at] = @intCast(b * ratio + j);
                    at += 1;
                }
            };
            for (blocks * ratio..live) |token| {
                f.want[at] = @intCast(token);
                at += 1;
            }
        }
        return f;
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
        fn depth(_: c.Shape) usize {
            return 0;
        }
        pub fn kernel(s: *Self) g.Kernel {
            const label = std.fmt.bufPrint(&s.label, "{s} Qwen4 QSA select {t}", .{ B.name, s.control }) catch unreachable;
            return .{ .ctx = s, .vt = &vt, .control = switch (s.control) {
                .production => null,
                .nan => .nonfinite,
                .repeat => .repeat,
                .tail => .shapes,
                .sums => unreachable,
            }, .spec = .{
                .name = label,
                .accum = .none,
                .out = .raw,
                .depth = depth,
                .nan = .{ .stated = "IEEE total order includes signed NaN payloads and zeros; ties choose earlier blocks; dense ids stay untouched." },
                .tiles = .{ .{ .size = 16 }, .{ .size = 128, .min = 0 }, .{ .size = 8 } },
                .served = B.served,
            } };
        }
        const vt = g.hooks(Self);
        pub fn prepare(s: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.rows > 128 or shape.n > 8192 or shape.k == 0 or shape.k > 8 or shape.n * 8 + shape.k == 1) return false;
            s.rig.reset();
            if (s.arena) |*a| a.deinit();
            s.arena = .init(s.a);
            s.repeat = 0;
            s.fixture = try Fixture.make(s.arena.?.allocator(), shape, fill, seed);
            try s.backend.bind(&s.rig, s.fixture, s.control);
            return true;
        }
        pub fn count(s: *Self) usize {
            return s.fixture.want.len * 4;
        }
        pub fn run(s: *Self) !void {
            try s.backend.launch(&s.rig, s.fixture, s.repeat);
            s.repeat += 1;
        }
        pub fn reference(s: *Self, want: []f64, mag: []f64) !void {
            for (std.mem.sliceAsBytes(s.fixture.want), want, mag) |v, *w, *m| {
                w.* = @floatFromInt(v);
                m.* = 0;
            }
        }
        pub fn check(s: *Self, at: g.Gate) !?[]const u8 {
            if (at == .nonfinite) for (special) |bits| {
                var found = false;
                for (s.fixture.scores) |v| if (@as(u32, @bitCast(v)) == bits) {
                    found = true;
                    break;
                };
                if (!found) return "selection fixture lacks a required exceptional IEEE value";
            };
            if (!try s.rig.intact()) return "selection wrote a guard";
            return if (std.mem.eql(u8, std.mem.sliceAsBytes(s.fixture.want), try s.rig.outputBytes())) null else "selected membership, stable block order, tails or untouched padding differs";
        }
    };
}
test "Qwen4 QSA selection: independent numeric ordering covers signed exceptional payloads" {
    var entries: [special.len]Entry = undefined;
    for (&entries, special, 0..) |*e, bits, i| e.* = .{ .score = @bitCast(bits), .index = i };
    std.mem.sort(Entry, &entries, {}, before);
    for (entries, 0..) |e, i| try std.testing.expectEqual(special.len - 1 - i, e.index);
}
