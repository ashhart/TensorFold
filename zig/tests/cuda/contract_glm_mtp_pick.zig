//! The device MTP rank scan replaces the original host scan byte for byte, including its nonfinite rule.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, control: ?g.Gate = null };
pub const entries = [_]Entry{
    .{ .name = "cuda glm MTP rank pick", .symbol = "tf_glm_mtp_pick" },
    .{ .name = "control: glm MTP rank tie to higher id", .symbol = "tf_ctl_glm_mtp_pick_tie", .control = .shapes },
    .{ .name = "control: glm MTP first-rank NaN dropped", .symbol = "tf_ctl_glm_mtp_pick_nan", .control = .nonfinite },
};
const served = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 14, .n = 1, .k = 2 } };
const rule = "scan ranks in order; greater float wins, equal values use lower unsigned id; NaN comparisons are false";
fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(entry: Entry) c.Spec {
    return .{ .name = entry.name, .accum = .none, .depth = depth, .out = .raw, .replaces = "host cuda_mtp_chance.pick", .nan = .{ .stated = rule }, .served = &served };
}

const Pick = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    entry: Entry,
    words: std.ArrayList([2][4]u32) = .empty,
    expected: std.ArrayList(u32) = .empty,
    in: u64 = 0,
    out: u64 = 0,
    const vt = g.hooks(Pick);

    fn deinit(p: *Pick) void {
        p.words.deinit(p.rig.gpu.gpa);
        p.expected.deinit(p.rig.gpu.gpa);
        p.rig.deinit();
    }

    fn kernel(p: *Pick) g.Kernel {
        return .{ .spec = spec(p.entry), .ctx = p, .vt = &vt, .control = p.entry.control };
    }

    pub fn prepare(p: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 2 or shape.n != 1 or shape.rows == 0 or shape.rows > 14) return false;
        p.rig.reset();
        try p.words.resize(p.rig.gpu.gpa, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const values = [_]u32{ 0x3f800000, 0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffcabcde, 0xbf800000 };
        for (p.words.items, 0..) |*row, r| {
            row.* = .{ .{ values[r % values.len], 17, rng.int(u32), rng.int(u32) }, .{ values[(r + 1) % values.len], 3, rng.int(u32), rng.int(u32) } };
        }
        p.words.items[0][0][0] = if (fill == .nonfinite) 0x7fc12345 else 0x3f800000;
        p.words.items[0][1][0] = 0x3f800000;
        p.in = try p.rig.input(std.mem.sliceAsBytes(p.words.items));
        p.out = try p.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(p: *Pick) !void {
        for (0..p.words.items.len) |r| {
            var args: cuda.Args = .{};
            args.add(p.in + r * 32);
            args.add(@as(u32, 2));
            args.add(p.out + r * 4);
            try cuda.launch.launch(p.f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, p.rig.s, &args);
        }
        try p.rig.s.synchronize();
    }

    pub fn count(p: *Pick) usize {
        return p.words.items.len * 4;
    }
    pub fn reference(_: *Pick, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn replaced(p: *Pick) ![]const u8 {
        try p.expected.resize(p.rig.gpu.gpa, p.words.items.len);
        for (p.words.items, p.expected.items) |*row, *out| out.* = glm.mtp_chance.pick(row, false).token;
        return std.mem.sliceAsBytes(p.expected.items);
    }
    pub fn check(p: *Pick, _: g.Gate) !?[]const u8 {
        const got = try p.rig.bytes(p.out, p.count());
        return if (std.mem.eql(u8, got, try p.replaced())) null else "MTP rank choice differs from the host scan";
    }
};

pub const Set = struct {
    picks: [entries.len]Pick = undefined,
    n: usize = 0,

    pub fn add(set: *Set, gpu: check.Gpu, stream: cuda.Stream, main: cuda.Module, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        for (entries, 0..) |entry, i| {
            const mod = if (entry.control == null) main else controls;
            set.picks[i] = .{ .rig = .{ .gpu = gpu, .s = stream }, .f = try mod.function(entry.symbol), .entry = entry };
            set.n += 1;
            try kernels.append(gpu.gpa, set.picks[i].kernel());
        }
    }

    pub fn deinit(set: *Set) void {
        for (set.picks[0..set.n]) |*p| p.deinit();
    }
};
