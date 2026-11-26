//! ERNIE 5.5's key lists (ernie5_5/index.cu) under the contract: each row's picks then its window, as window.merge.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_index_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const win = ernie.window;

/// A planted fault: every row takes all k picks, or the window's oldest position is dropped.
pub const Fault = enum { none, all_picks, short_window };

pub const Entry = struct { name: []const u8, symbol: [:0]const u8, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda ernie5_5 index tf_ernie55_window_merge (top-k picks, then the window)", .symbol = "tf_ernie55_window_merge" },
    .{ .name = "control: ernie5_5 window merge taking all k picks", .symbol = "tf_ctl_ernie55_window_merge_all_picks", .fault = .all_picks },
    .{ .name = "control: ernie5_5 window merge dropping the window's oldest position", .symbol = "tf_ctl_ernie55_window_merge_short_window", .fault = .short_window },
};

/// rows, row 0's seq (n) and k: across the window with fewer picks than k (the controls' faults show), then the rest.
const served = &[_]c.Shape{ .{ .rows = 16, .n = 120, .k = win.topk }, .{ .rows = 1, .n = 1, .k = win.topk }, .{ .rows = 16, .n = win.topk + win.window - 8, .k = win.topk }, .{ .rows = 1, .n = 100_000, .k = win.topk }, .{ .rows = 512, .n = 1, .k = win.topk }, .{ .rows = 256, .n = 4000, .k = win.topk }, .{ .rows = 4, .n = 1 << 20, .k = win.topk } };

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "window.merge: picks to picked in [0, k] and the first -1, then the window to seq - 1, -1 after; no NaN";

pub fn spec(e: Entry) c.Spec {
    const tiles = [3]c.Tile{ .{}, .{ .size = win.window }, .{} };
    return .{ .name = e.name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = rule }, .tiles = tiles, .served = if (e.fault == .none) served else served[0..1] };
}

pub fn control(e: Entry) ?g.Gate {
    return if (e.fault == .none) null else .shapes;
}

pub const Lists = struct {
    rig: rig_.Rig,
    entry: Entry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    picks: []i32 = &.{},
    told: []i32 = &.{}, // what the kernel reads as picked: the due counts, a negative and two oversized ones planted
    due: []i32 = &.{}, // each row's true count of picks
    dev: [3]u64 = @splat(0), // picks, picked, pos
    tok: u64 = 0,
    cnt: u64 = 0,
    msg: [200]u8 = undefined,

    const vt = g.hooks(Lists);

    pub fn kernel(self: *Lists) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control(self.entry) };
    }

    fn free(self: *Lists) void {
        const a = self.rig.gpu.gpa;
        a.free(self.picks);
        a.free(self.told);
        a.free(self.due);
        self.picks = &.{};
        self.told = &.{};
        self.due = &.{};
    }

    pub fn deinit(self: *Lists) void {
        self.free();
        self.rig.deinit();
    }

    fn sh(self: *const Lists) win.Shape {
        return .{ .rows = self.shape.rows, .k = self.shape.k, .w = self.shape.k + win.window };
    }

    /// Picks as topk_rows leaves them (n below seq - window, then -1); row 4 told -3, rows 3 and 5 told k + 7.
    pub fn prepare(self: *Lists, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        self.picks = try a.alloc(i32, shape.rows * shape.k);
        self.told = try a.alloc(i32, shape.rows);
        self.due = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..shape.rows) |r| {
            const older: usize = (shape.n + r) -| win.window;
            const n = @min(older, shape.k);
            for (self.picks[r * shape.k ..][0..shape.k], 0..) |*p, i| {
                if (i >= n) {
                    p.* = -1;
                    continue;
                }
                const lo = i * older / n;
                p.* = @intCast(lo + rng.uintLessThan(usize, (i + 1) * older / n - lo));
            }
            self.due[r] = @intCast(n);
        }
        @memcpy(self.told, self.due);
        if (shape.rows >= 5) {
            self.told[4] = -3;
            if (self.due[3] == shape.k) self.told[3] = @intCast(shape.k + 7);
        }
        if (shape.rows >= 6 and self.due[5] < shape.k) self.told[5] = @intCast(shape.k + 7); // past its valid picks
        const pos0 = [1]i32{@intCast(shape.n - 1)};
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.picks)), try self.rig.input(std.mem.sliceAsBytes(self.told)), try self.rig.input(std.mem.sliceAsBytes(&pos0)) };
        self.tok = try self.rig.output(shape.rows * (shape.k + win.window) * 4);
        self.cnt = try self.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(self: *Lists) !void {
        try ernie.launch(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.tok, self.cnt, self.sh());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Lists) usize {
        return self.shape.rows * (self.shape.k + win.window + 1) * 4;
    }

    pub fn reference(_: *Lists, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn fail(self: *Lists, comptime fmt: []const u8, args: anytype) !?[]const u8 {
        return try std.fmt.bufPrint(&self.msg, fmt, args);
    }

    /// window.merge's bytes; every row: its valid picks then the window, each below seq, ascending, last seq - 1.
    pub fn check(self: *Lists, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const s = self.sh();
        const got = try self.rig.outputBytes();
        const tok = try a.alloc(i32, s.rows * s.w);
        defer a.free(tok);
        const cnt = try a.alloc(i32, s.rows);
        defer a.free(cnt);
        win.merge(s, self.picks, self.told, @intCast(self.shape.n - 1), tok, cnt);
        const outs = s.rows * s.w * 4;
        if (std.mem.indexOfDiff(u8, got[0..outs], std.mem.sliceAsBytes(tok))) |at| return self.fail("list entry {d} is {d}, window.merge gives {d}", .{ at / 4, std.mem.bytesAsSlice(i32, got[0..outs])[at / 4], tok[at / 4] });
        if (std.mem.indexOfDiff(u8, got[outs..], std.mem.sliceAsBytes(cnt))) |at| return self.fail("row {d}'s count differs from window.merge's", .{at / 4});
        for (0..s.rows) |r| {
            const seq: i64 = @intCast(self.shape.n + r);
            const row = tok[r * s.w ..][0..s.w];
            const m: usize = @intCast(cnt[r]);
            // its count held to [0, k], then to its valid picks (the first `due`; -1 after them)
            const told: usize = @intCast(std.math.clamp(self.told[r], 0, @as(i32, @intCast(s.k))));
            const n = @min(told, @as(usize, @intCast(self.due[r])));
            const due = n + @as(usize, @intCast(@min(seq, win.window)));
            if (m != due) return self.fail("row {d}: {d} entries where its picks and window are {d}", .{ r, m, due });
            for (row[0..m], 0..) |v, i| if (v < 0 or v >= seq) return self.fail("row {d} entry {d}: {d}, not a row below seq {d}", .{ r, i, v, seq });
            for (row[1..m], 1..) |v, i| if (v <= row[i - 1]) return self.fail("row {d} entry {d}: {d} after {d}, not ascending", .{ r, i, v, row[i - 1] });
            if (n > 0 and row[n - 1] >= seq - win.window) return self.fail("row {d}: a pick inside the window", .{r});
            if (row[m - 1] != seq - 1) return self.fail("row {d}: the last entry is {d}, not seq - 1 = {d}", .{ r, row[m - 1], seq - 1 });
            for (row[m..]) |v| if (v != -1) return self.fail("row {d}: {d} past its count, not -1", .{ r, v });
        }
        return null;
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    index: ernie.Index,
    items: [entries.len]Lists = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var index = try ernie.Index.load(gpu.d);
        errdefer index.unload();
        set.* = .{ .gpa = gpu.gpa, .index = index };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.index.mod.function(e.symbol) };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.index.unload();
        set.gpa.destroy(set);
    }
};

/// The window merge's gates 1-4 and its controls alone.
pub fn run(gpu: check_.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x65726e6a, &t);
    std.debug.print("SUMMARY ernie5_5 index: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check_.expect(t.problems() == 0, "ernie5_5-index contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the served shapes fit the kernel, and the first leaves rows fewer picks than k on both sides of the window" {
    for (served) |x| try win.check(.{ .rows = x.rows, .k = x.k, .w = x.k + win.window });
    const first = served[0];
    try std.testing.expect(first.n <= win.window and first.n + first.rows > win.window + 1 and first.n + first.rows < win.window + first.k);
}
