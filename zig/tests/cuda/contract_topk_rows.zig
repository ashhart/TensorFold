//! topk_rows.cu under the contract on CUDA: each row's picks equal core/sparse_index.zig's selection on its scores.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const si = core.sparse_index;
const topk = cuda.topk_rows;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "sparse_index.select's positions on every row (NaN above +inf, -0 as +0), and the order keys descending " ++
    "(-0 as +0, any NaN above +inf), ties to the lower index, the picks ascending and -1 after them";

/// Inputs: random levels, all tied, or subnormal (lost under flush-to-zero); `short`, the control: one score fewer.
pub const Plant = enum { none, tied, subnormal, short };

/// Shape: rows, scores a row (n) and picks (k). Kolibri 1.1's 2,048 of 16,384; GLM's and DeepSeek's widths.
pub fn spec(plant: Plant) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 1, .n = 16384, .k = 2048 }, .{ .rows = 4, .n = 16384, .k = 2048 }, .{ .rows = 16, .n = 16384, .k = 2048 }, .{ .rows = 24, .n = 4096, .k = 2048 }, .{ .rows = 6, .n = 65536, .k = 512 }, .{ .rows = 2, .n = 131072, .k = 2048 }, .{ .rows = 40, .n = 2048, .k = 1024 }, .{ .rows = 60, .n = 8192, .k = 2048 } };
    return .{
        .name = switch (plant) {
            .none => "cuda topk rows tf_topk_rows tf_topk_rows_wide",
            .tied => "cuda topk rows tf_topk_rows tf_topk_rows_wide (every score tied)",
            .subnormal => "cuda topk rows tf_topk_rows tf_topk_rows_wide (subnormal scores)",
            .short => "control: topk rows told each row one score fewer than it has",
        },
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = topk.wide_rows }, .{ .size = 8192 }, .{ .size = 2048 } },
        .served = served,
    };
}

/// The GPU's order key: s + 0 turns -0 into +0 and every NaN into one positive NaN.
fn orderKey(s: f32) u32 {
    const b: i32 = if (std.math.isNan(s)) std.math.maxInt(i32) else @bitCast(s + 0.0);
    return @bitCast(b ^ ((b >> 31) | std.math.minInt(i32)));
}

fn before(keys: []const u32, x: u32, y: u32) bool {
    return keys[x] > keys[y] or (keys[x] == keys[y] and x < y);
}

pub const Rows = struct {
    rig: rig_.Rig,
    t: *const topk.TopkRows,
    plant_: Plant = .none,
    shape: c.Shape = undefined,
    host: []f32 = &.{},
    counts: []i32 = &.{},
    told: []i32 = &.{}, // what the kernel reads: counts, cut by the control, a negative and an oversized one planted
    dev: [2]u64 = @splat(0), // scores, counts (0: every row has n)
    out: u64 = 0,
    picked: u64 = 0,
    other: [2]u64 = @splat(0), // the other form's out and picked
    with_select: usize = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Rows) void {
        const a = self.rig.gpu.gpa;
        a.free(self.host);
        a.free(self.counts);
        a.free(self.told);
        self.rig.deinit();
    }

    pub fn kernel(self: *Rows) g.Kernel {
        return .{ .spec = spec(self.plant_), .ctx = self, .vt = &vt, .control = if (self.plant_ == .short) .shapes else null };
    }

    const vt = g.hooks(Rows);

    fn sh(self: *const Rows) topk.Shape {
        return .{ .rows = self.shape.rows, .ld = self.shape.n, .k = self.shape.k };
    }

    /// Eighth steps in [-16, 16) with -0 among the zeros (thousands of ties a row); each row's last score its best.
    pub fn prepare(self: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        a.free(self.host);
        a.free(self.counts);
        a.free(self.told);
        self.host = &.{};
        self.counts = &.{};
        self.told = &.{};
        self.host = try a.alloc(f32, shape.rows * shape.n);
        self.counts = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // row r its first n - r n / rows scores; the last row fewer than k when there is room; one row: no counts
        for (self.counts, 0..) |*n, r| {
            var v = shape.n - r * shape.n / shape.rows;
            if (r + 1 == shape.rows and r > 0 and shape.k > 1) v = @min(v, shape.k - 1);
            n.* = @intCast(v);
        }
        for (0..shape.rows) |r| {
            for (self.host[r * shape.n ..][0..shape.n]) |*v| {
                const level: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 256))) - 128);
                v.* = switch (self.plant_) {
                    .tied => 0.5,
                    // 256 subnormal levels either sign (bits below 2^23): all 0 under flush-to-zero
                    .subnormal => @bitCast(@as(u32, @intCast(rng.uintLessThan(u32, 256))) * 4099 | (if (rng.boolean()) @as(u32, 0x80000000) else 0)),
                    else => if (level == 0 and rng.boolean()) -0.0 else level * 0.125,
                };
            }
            // each row's last score its best (tied rows stay tied; a subnormal row's best is subnormal too)
            const best: f32 = switch (self.plant_) {
                .tied => 0.5,
                .subnormal => @bitCast(@as(u32, 0x7fffff)),
                else => 100,
            };
            self.host[r * shape.n + @as(usize, @intCast(self.counts[r])) - 1] = best;
        }
        if (fill == .nonfinite) {
            // rows 0 and 1: NaN of either sign and with a payload, mixed in a row, rank as one value above +inf
            const nans = [2][2]u32{ .{ 0x7fc00000, 0x7f800001 }, .{ 0xffc00000, 0xff812345 } };
            const put = [_]f32{ std.math.nan(f32), std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, r| if (r < shape.rows) {
                const n: usize = @intCast(self.counts[r]);
                const step = @max(1, n / 8);
                for (0..8) |e| if (e * step + r < n) {
                    self.host[r * shape.n + e * step + r] = if (r < 2) @bitCast(nans[r][e % 2]) else v;
                };
            };
        }
        self.told = try a.dupe(i32, self.counts);
        const told = self.told;
        if (self.plant_ == .short) {
            for (told) |*n| n.* = @max(1, n.* - 1);
        } else if (shape.rows >= 5) {
            told[4] = -3; // the kernel clamps to [0, n]: no picks (rows 0-3 keep their NaN and infinities counted)
            told[3] = @intCast(shape.n + 7); // all n scores, none past the row
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.host));
        self.dev[1] = if (shape.rows == 1 and self.plant_ != .short) 0 else try self.rig.input(std.mem.sliceAsBytes(told));
        self.out = try self.rig.output(shape.rows * shape.k * 4);
        self.picked = try self.rig.output(shape.rows * 4);
        self.other = .{ try self.rig.scratch(shape.rows * shape.k * 4), try self.rig.scratch(shape.rows * 4) };
        return true;
    }

    pub fn run(self: *Rows) !void {
        try self.t.run(self.rig.s, self.dev[0], self.dev[1], self.out, self.picked, self.sh());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rows) usize {
        return self.shape.rows * (self.shape.k + 1) * 4;
    }

    pub fn reference(_: *Rows, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(self: *Rows, comptime fmt: []const u8, args: anytype) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, fmt, args);
    }

    /// The other form's bytes, then every row against sparse_index.select (the kernel's NaN order) and the key rule.
    pub fn check(self: *Rows, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const x = self.shape;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const wide = !topk.wide(self.sh());
        try self.t.launchWith(if (wide) self.t.wide else self.t.narrow, if (wide) 1024 else 128, self.rig.s, self.dev[0], self.dev[1], self.other[0], self.other[1], self.sh());
        const outs = x.rows * x.k * 4;
        if (std.mem.indexOfDiff(u8, got[0..outs], try self.rig.bytes(self.other[0], outs))) |at| return self.fail("byte {d} of the picks differs between the 128- and 1,024-thread forms", .{at});
        if (std.mem.indexOfDiff(u8, got[outs..], try self.rig.bytes(self.other[1], x.rows * 4))) |at| return self.fail("byte {d} of the counts differs between the two forms", .{at});
        const keys = try a.alloc(u32, x.n);
        defer a.free(keys);
        const idx = try a.alloc(u32, x.n);
        defer a.free(idx);
        const wide64 = try a.alloc(f64, x.n);
        defer a.free(wide64);
        self.with_select = 0;
        for (0..x.rows) |r| {
            // the true count for the control (it must miss), else the told count clamped to [0, n] as the kernel does
            const n: usize = if (self.plant_ == .short) @intCast(self.counts[r]) else @intCast(std.math.clamp(self.told[r], 0, @as(i32, @intCast(x.n))));
            const scores = self.host[r * x.n ..][0..n];
            const take = @min(n, x.k);
            const have = std.mem.bytesAsSlice(i32, got[r * x.k * 4 ..][0 .. x.k * 4]);
            const picked = std.mem.readInt(i32, got[outs + r * 4 ..][0..4], .little);
            if (picked != take) return self.fail("row {d}: {d} picks where {d} are due", .{ r, picked, take });
            for (keys[0..n], scores) |*k, s| k.* = orderKey(s);
            for (idx[0..n], 0..) |*v, p| v.* = @intCast(p);
            std.mem.sort(u32, idx[0..n], keys[0..n], before);
            const want = idx[0..take];
            std.mem.sort(u32, want, {}, std.sort.asc(u32));
            for (want, 0..) |w, i| if (have[i] != w) return self.fail("row {d} pick {d}: index {d}, the order-key rule gives {d}", .{ r, i, have[i], w });
            for (have[take..], take..) |v, i| if (v != -1) return self.fail("row {d} slot {d} past its picks holds {d}, not -1", .{ r, i, v });
            if (n > 0) {
                for (wide64[0..n], scores) |*y, s| y.* = s;
                const one = [_]f64{1};
                var sel = try si.select(a, .{ .topk = x.k, .phi = .identity }, .{ .q = &one, .heads = 1 }, .{ .k = wide64[0..n], .dim = 1 }, n - 1, null);
                defer sel.deinit(a);
                if (sel.keys.len != take) return self.fail("row {d}: sparse_index selects {d}, the kernel {d}", .{ r, sel.keys.len, take });
                for (sel.keys, 0..) |w, i| if (have[i] != w) return self.fail("row {d} pick {d}: index {d}, sparse_index.select gives {d}", .{ r, i, have[i], w });
                self.with_select += 1;
            }
        }
        if (self.with_select == 0) return self.fail("no row had scores: nothing compared with sparse_index", .{});
        return null;
    }
};

/// The kernel and its planted control, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    t: topk.TopkRows,
    tests: [4]Rows,

    pub fn init(gpu: tc.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.gpa = gpu.gpa;
        set.t = try topk.TopkRows.load(gpu.d);
        errdefer set.t.unload();
        for (&set.tests, [_]Plant{ .none, .tied, .subnormal, .short }) |*t, p| t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .t = &set.t, .plant_ = p };
        for (&set.tests) |*t| try list.append(a, t.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.tests) |*t| t.deinit();
        set.t.unload();
        set.gpa.destroy(set);
    }
};
