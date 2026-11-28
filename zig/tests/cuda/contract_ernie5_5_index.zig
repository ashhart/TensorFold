//! ERNIE 5.5's DSA indexer (ernie5_5/index.cu) under the contract: its keys as key.key, its lists as window.merge.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ernie = @import("ernie5_5_index_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const bfr = core.bf16_round;
const win = ernie.window;
const kk = ernie.key;

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

const kd = kk.head_dim;
/// k rows as the vendor's fused wk and weights_proj give them: 128 of k, then the 64 head weights the kernel skips.
const ks = kd + 64;
/// The released config's rope_theta: the DSA layers' plain RoPE (YaRN is the HCA layers' alone).
const theta = 8.0e6;

/// A planted fault: GPT-J's pairs or no bias (control entries), or the next position (the kernel, told pos0 + 1).
pub const KeyFault = enum { none, gptj_pairs, no_bias, next_position };

pub const KeyEntry = struct { name: []const u8, symbol: [:0]const u8, fault: KeyFault = .none };

pub const key_entries = [_]KeyEntry{
    .{ .name = "cuda ernie5_5 index tf_ernie55_index_key (k's LayerNorm, then NeoX RoPE on its first 64 dims)", .symbol = "tf_ernie55_index_key" },
    .{ .name = "control: ernie5_5 index key turning GPT-J's pairs (2i, 2i + 1)", .symbol = "tf_ctl_ernie55_index_key_gptj_pairs", .fault = .gptj_pairs },
    .{ .name = "control: ernie5_5 index key without k_norm's bias", .symbol = "tf_ctl_ernie55_index_key_no_bias", .fault = .no_bias },
    .{ .name = "control: ernie5_5 index key turned at the next position's angle", .symbol = "tf_ernie55_index_key", .fault = .next_position },
};

/// rows, row 0's seq (n) and the width: a chunk, a decode row at 0, a long chunk, past sincosf's fast range, at 1M.
const key_served = &[_]c.Shape{ .{ .rows = 16, .n = 121, .k = kd }, .{ .rows = 1, .n = 1, .k = kd }, .{ .rows = 512, .n = 4000, .k = kd }, .{ .rows = 1, .n = 200_000, .k = kd }, .{ .rows = 4, .n = (1 << 20) - 3, .k = kd } };

/// A lane's 4 terms in order (3), the warp's butterfly (5), and in the squares their product and xc's rounding twice.
fn keyDepth(_: c.Shape) usize {
    return 11;
}

pub fn keySpec(e: KeyEntry) c.Spec {
    return .{ .name = e.name, .accum = .f32, .depth = keyDepth, .out = .bf16, .served = if (e.fault == .none) key_served else key_served[0..1] };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const Keys = struct {
    rig: rig_.Rig,
    entry: KeyEntry,
    f: cuda.Function,
    shape: c.Shape = undefined,
    x: []u16 = &.{}, // k rows ks apart, NaN past 128: read, they would show
    w: [kd]f64 = undefined, // k_norm's weight and bias, bf16 values
    b: [kd]f64 = undefined,
    inv: [kk.rope / 2]f32 = undefined,
    extra: []f64 = &.{}, // per output: the allowance for its own roundings beyond the spec's bound
    dev: [5]u64 = @splat(0), // x, w, b, pos, inv
    out: u64 = 0,

    const vt = g.hooks(Keys);

    pub fn kernel(self: *Keys) g.Kernel {
        return .{ .spec = keySpec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    fn free(self: *Keys) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.extra);
        self.x = &.{};
        self.extra = &.{};
    }

    pub fn deinit(self: *Keys) void {
        self.free();
        self.rig.deinit();
    }

    /// Rows offset 0, 64 or 512 over noise; stress: 254 to 260 in 2s, sums exact in fp32, the steps lost in fp16, bf16.
    pub fn prepare(self: *Keys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != kd) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var w: [kd]u16 = undefined;
        var b: [kd]u16 = undefined;
        for (&w, &b, &self.w, &self.b) |*v, *h, *fw, *fb| {
            v.* = bfr.bits(rng.float(f32) * 3 - 0.5);
            h.* = bfr.bits(rng.float(f32) * 6 - 3);
            fw.* = bf(v.*);
            fb.* = bf(h.*);
        }
        self.x = try a.alloc(u16, shape.rows * ks);
        for (0..shape.rows) |r| {
            const row = self.x[r * ks ..][0..ks];
            @memset(row[kd..], bfr.bits(std.math.nan(f32)));
            const offset: f32 = switch (r % 3) {
                0 => 0,
                1 => 64,
                else => 512,
            };
            for (row[0..kd], 0..) |*v, i| v.* = if (fill == .stress)
                bfr.bits(254 + 2 * @as(f32, @floatFromInt(rng.uintLessThan(u32, 4))))
            else
                bfr.bits(offset + (rng.float(f32) * 2 - 1) * (if (i % 64 == 9) @as(f32, 20) else 1));
        }
        if (fill == .nonfinite) {
            const planted = [_]struct { usize, usize, f32 }{ .{ 0, 42, std.math.nan(f32) }, .{ 1, 64, std.math.inf(f32) }, .{ 2, kd - 1, -std.math.inf(f32) } };
            for (planted) |p| if (p[0] < shape.rows) {
                self.x[p[0] * ks + p[1]] = bfr.bits(p[2]);
            };
        }
        core.rope_freq.inverseFrequencies(theta, kk.rope, &self.inv);
        const pos0 = [1]i32{@intCast(shape.n - 1 + @intFromBool(self.entry.fault == .next_position))};
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.x)), try self.rig.input(std.mem.sliceAsBytes(&w)), try self.rig.input(std.mem.sliceAsBytes(&b)), try self.rig.input(std.mem.sliceAsBytes(&pos0)), try self.rig.input(std.mem.sliceAsBytes(&self.inv)) };
        self.out = try self.rig.output(shape.rows * kd * 2);
        return true;
    }

    pub fn run(self: *Keys) !void {
        try ernie.launchKey(self.f, self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.dev[3], self.dev[4], .{ .rows = self.shape.rows, .ks = ks });
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Keys) usize {
        return self.shape.rows * kd;
    }

    /// Row r's k values and position.
    fn keyRow(self: *const Keys, r: usize, x: *[kd]f64) i64 {
        for (x, self.x[r * ks ..][0..kd]) |*v, h| v.* = bf(h);
        return @intCast(self.shape.n - 1 + r);
    }

    /// key.key in float64; mag: the sums' error at an output (contract_layer_norm's, turned); extra: its own roundings.
    pub fn reference(self: *Keys, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        const u = 0x1p-24;
        for (0..self.shape.rows) |r| {
            var x: [kd]f64 = undefined;
            const pos = self.keyRow(r, &x);
            kk.key(&x, &self.w, &self.b, &self.inv, pos, kk.exact, want[r * kd ..][0..kd]);
            const n = kk.norm(&x, kk.exact);
            var abs: f64 = 0;
            for (x) |v| abs += @abs(v);
            const m = mag[r * kd ..][0..kd];
            const e = self.extra[r * kd ..][0..kd];
            var my: [kd]f64 = undefined;
            var ey: [kd]f64 = undefined;
            var ay: [kd]f64 = undefined;
            for (&my, &ey, &ay, x, self.w, self.b) |*sm, *se, *sy, v, s, h| {
                const t = (v - n.mean) * n.rstd * s;
                sm.* = (2 * @abs(v - n.mean) + abs / kd) * n.rstd * @abs(s);
                // xc, eps's add, the root and the division (3.5 units of t), then the fma's into y
                se.* = 4 * u * @abs(t) + u * @abs(t + h);
                sy.* = @abs(t + h);
            }
            @memcpy(m, &my);
            @memcpy(e, &ey);
            for (self.inv, 0..) |f, i| {
                const ang = kk.angle(f, pos);
                const co = @abs(@cos(ang));
                const si = @abs(@sin(ang));
                for ([2][2]usize{ .{ i, i + kk.rope / 2 }, .{ i + kk.rope / 2, i } }) |p| {
                    m[p[0]] = co * my[p[0]] + si * my[p[1]];
                    // y's own error turned, then sincosf's 2 ulp, the two products and the sum: 8 units of the terms
                    e[p[0]] = co * ey[p[0]] + si * ey[p[1]] + 8 * u * (co * ay[p[0]] + si * ay[p[1]]);
                }
            }
        }
    }

    pub fn bounds(self: *Keys, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        for (allow, self.extra) |*l, x| l.* += x * (1 + 0x1p-7);
    }

    /// key.key with both sums in `acc`, in `order`.
    pub fn simulate(self: *Keys, acc: c.Accum, order: g.Order, out: []f64) !void {
        const sim = struct {
            acc: c.Accum,
            order: g.Order,
            pub fn of(s: @This(), t: []const f64) f64 {
                return rows_.sumIn(s.acc, s.order, t);
            }
        }{ .acc = acc, .order = order };
        for (0..self.shape.rows) |r| {
            var x: [kd]f64 = undefined;
            const pos = self.keyRow(r, &x);
            kk.key(&x, &self.w, &self.b, &self.inv, pos, sim, out[r * kd ..][0..kd]);
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    index: ernie.Index,
    keys: [key_entries.len]Keys = undefined,
    items: [entries.len]Lists = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var index = try ernie.Index.load(gpu.d);
        errdefer index.unload();
        set.* = .{ .gpa = gpu.gpa, .index = index };
        for (&set.keys, key_entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.index.mod.function(e.symbol) };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = try set.index.mod.function(e.symbol) };
        for (&set.keys) |*x| try list.append(a, x.kernel());
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.keys) |*x| x.deinit();
        for (&set.items) |*x| x.deinit();
        set.index.unload();
        set.gpa.destroy(set);
    }
};

/// The index image's gates 1-4 and its controls alone: the keys, then the key lists.
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
    for (key_entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}

test "the key shapes fit the kernel, and the first, which gates 1 and 4 and the controls take, turns 8 rows past 0" {
    for (key_served) |x| try kk.check(.{ .rows = x.rows, .ks = ks });
    try std.testing.expect(key_served[0].rows >= 8 and key_served[0].n > 1);
    try std.testing.expect(key_served[key_served.len - 1].n - 1 + key_served[key_served.len - 1].rows == 1 << 20);
}

test "the served shapes fit the kernel, and the first leaves rows fewer picks than k on both sides of the window" {
    for (served) |x| try win.check(.{ .rows = x.rows, .k = x.k, .w = x.k + win.window });
    const first = served[0];
    try std.testing.expect(first.n <= win.window and first.n + first.rows > win.window + 1 and first.n + first.rows < win.window + first.k);
}
