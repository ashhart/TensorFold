//! One row's routed fc1 and shared up in one launch under the contract on HIP: the two relu^2 stream tiles' bytes.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const relu2_k = @import("contract_relu2.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

const bits = 4;
const group = 64;
const experts = 8; // the experts the plan's items pick from (a layer has 128; the kernel reads the picked ones)
const top_k = 6;
/// launches/affine.zig's stream tile constants: columns a lane, row counts, waves a block, waves wanted, code words.
const stream_cbs = [_]c_int{ 2, 4, 8 };
const stream_rows = [_]u8{ 1, 2, 4 };
const stream_waves = 4;
const streams_wanted = 3000;
const stream_code_words = 32;
const bit_index = 2; // 4 in launches/affine.zig's bit_widths

fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / group + 8;
}

const names = [2][]const u8{
    "hip affine_dot2_stream_relu2_pair (4-bit, fp16)",
    "hip affine_dot2_stream_relu2_pair (4-bit, bf16)",
};

pub fn spec(k: kit.Kind) c.Spec {
    var s = relu2_k.spec(k);
    s.name = names[@intFromBool(k == .bf16)];
    s.nan = .{ .stated = "the separate launches' bytes and every float64 output class" };
    s.depth = depth;
    s.replaces = "the routed and the plain relu^2 stream tiles";
    // Nano's one-row fc1 (1,856 wide, six experts) and shared up (twice as wide), and a narrow layer
    s.served = &.{ .{ .rows = 1, .n = 1856, .k = 2688 }, .{ .rows = 1, .n = 256, .k = 512 } };
    return s;
}

/// x [1][k] of the kind; the plan's six items over `experts` stacked [n][k] matrices, and a plain [2n][k] one.
pub const Pair = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    fa: ?Affine = null,
    fb: ?Affine = null,
    picks: [top_k]usize = undefined,
    arg: [2]relu2_k.Arg = undefined,
    base: [2]u64 = @splat(0),

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Pair {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Pair) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Pair) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Pair);

    fn upload(self: *Pair, f: Affine, out16: u64) !relu2_k.Arg {
        const a = self.arena.allocator();
        const w = try a.alloc(u32, f.shape.n * f.shape.k * bits / 32);
        f.words(w);
        return .{
            .x = 0,
            .words = try self.rig.input(std.mem.sliceAsBytes(w)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = 0,
            .m = 1,
            .n = 0,
            .k = @intCast(f.shape.k),
            .bits = bits,
            .group = group,
            .fp16 = @intFromBool(self.k == .f16),
            .out16 = out16,
        };
    }

    pub fn prepare(self: *Pair, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // one row of x, whatever rows a gate asks for: the pair runs a one-row layer
        if (shape.n == 0 or shape.k % group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var fa = try Affine.init(a, .{ .rows = 1, .n = experts * shape.n, .k = shape.k }, bits, group);
        var fb = try Affine.init(a, .{ .rows = 1, .n = 2 * shape.n, .k = shape.k }, bits, group);
        // non-finite: in a few group tables, not in x, so the other columns stay finite
        try fa.fill(if (fill == .nonfinite) .random else fill, seed);
        try fb.fill(if (fill == .nonfinite) .random else fill, seed ^ 0x5eed);
        if (fill == .stress) for (fa.x) |*b| {
            b.* ^= 0x8000; // relu would hide the stress rows' negative products
        };
        if (self.k == .f16) for (fa.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0;
        };
        @memcpy(fb.x, fa.x);
        self.fa = fa;
        self.fb = fb;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x9);
        const rng = prng.random();
        for (&self.picks, 0..) |*p, i| p.* = (rng.uintLessThan(usize, experts) + i) % experts;
        if (fill == .nonfinite) {
            const groups = shape.k / group;
            fa.scales[(self.picks[0] * shape.n + 3) * groups] = c.toBf16(std.math.nan(f32));
            fa.biases[(self.picks[1] * shape.n + 5) * groups + 1] = c.toBf16(std.math.inf(f32));
            fb.scales[7 * groups + 2] = c.toBf16(-std.math.inf(f32));
        }
        const xb = try a.alloc(u8, fa.x.len * 2);
        for (fa.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const x = try self.rig.input(xb);
        // the plan as tf_nemo_select writes one row's: item k is (pick k, slot k, one row), member k is k
        var items: [3 * top_k]i32 = undefined;
        var members: [top_k]i32 = undefined;
        for (0..top_k) |i| {
            items[3 * i ..][0..3].* = .{ @intCast(self.picks[i]), @intCast(i), 1 };
            members[i] = @intCast(i);
        }
        const out = [2]u64{ try self.rig.output(top_k * shape.n * 2), try self.rig.output(2 * shape.n * 2) };
        self.base = .{ try self.rig.scratch(top_k * shape.n * 2), try self.rig.scratch(2 * shape.n * 2) };
        self.arg = .{ try self.upload(fa, out[0]), try self.upload(fb, out[1]) };
        for (&self.arg, [_]usize{ shape.n, 2 * shape.n }) |*arg, n| {
            arg.x = x;
            arg.n = @intCast(n);
        }
        self.arg[0].route = .{
            .items = try self.rig.input(std.mem.sliceAsBytes(&items)),
            .members = try self.rig.input(std.mem.sliceAsBytes(&members)),
            .x_div = top_k,
        };
        return true;
    }

    fn cdiv(n: c_int, by: c_int) u32 {
        return @intCast(@divTrunc(n + by - 1, by));
    }

    /// The pair's launch as launches/affine_pair.zig makes it: the routed product's own tile, the plain one's past it.
    pub fn run(self: *Pair) !void {
        const a = self.arg[0];
        const b = self.arg[1];
        var lpc_log2: u5 = 0;
        while ((@as(c_int, 1) << lpc_log2) < (a.k >> 5) and lpc_log2 < 5) lpc_log2 += 1;
        const ri: usize = 0; // one row
        var pick: usize = 0;
        var cbi: usize = stream_cbs.len;
        while (cbi > 0) {
            cbi -= 1;
            const cb = stream_cbs[cbi];
            if (cb * stream_rows[ri] > 16 or cb * a.bits > stream_code_words) continue;
            pick = cbi;
            const per: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * cb;
            const waves = @as(u64, cdiv(a.n, per)) * stream_waves * top_k;
            if (waves * cdiv(a.m, stream_rows[ri]) >= streams_wanted) break;
        }
        const per: c_int = stream_waves * (@as(c_int, 32) >> lpc_log2) * stream_cbs[pick];
        const split = cdiv(a.n, per);
        const f = self.l.affine.relu2_pair[bit_index][ri][pick].?;
        var args: hip.launch.Args = .{};
        try args.add(a);
        try args.add(b);
        try args.add(@as(c_int, @intCast(split)));
        try args.add(@as(c_int, lpc_log2));
        try args.add(@as(c_int, 1)); // groups of 64
        const grid: hip.launch.Dim3 = .{ .x = split + cdiv(b.n, per), .y = 1, .z = top_k };
        try hip.launch.launch(f, .{ .grid = grid, .block = .{ .x = 32 * stream_waves } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    /// The routed relu^2 tile over the plan's items, then the plain one, each its own launch.
    pub fn replaced(self: *Pair) ![]const u8 {
        const st = self.rig.s.handle;
        var routed = self.arg[0];
        routed.out16 = self.base[0];
        var plain = self.arg[1];
        plain.out16 = self.base[1];
        try self.l.affine.relu2Go(self.l.r, routed, top_k, st);
        try self.l.affine.relu2Go(self.l.r, plain, 1, st);
        const a = self.arena.allocator();
        const n: usize = @intCast(self.arg[0].n);
        const first = try a.dupe(u8, try self.rig.bytes(self.base[0], top_k * n * 2));
        return std.mem.concat(a, u8, &.{ first, try self.rig.bytes(self.base[1], 2 * n * 2) });
    }

    /// Every original table plant must match the float64 class and the separate launches' bytes.
    pub fn check(self: *Pair, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        if (!std.mem.eql(u8, out, try self.replaced())) return "non-finite pair differs from separate tiles' bytes";
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(kit.elem(self.k), out, i)) != c.classOf(w))
                return "non-finite pair class differs from float64";
        }
        return null;
    }

    pub fn count(self: *Pair) usize {
        return (top_k + 2) * @as(usize, @intCast(self.arg[0].n));
    }

    /// Each item's expert rows, then the plain rows, as relu(y)^2 (NaN kept as torch keeps it).
    fn outputs(self: *Pair, want: []f64, mag: []f64, sim: ?struct { c.Accum, g.Order }) !void {
        const a = self.rig.a;
        const n: usize = @intCast(self.arg[0].n);
        for ([_]Affine{ self.fa.?, self.fb.? }, 0..) |f, which| {
            const all = try a.alloc(f64, 2 * f.outputs());
            defer a.free(all);
            const y = all[0..f.outputs()];
            const m = all[f.outputs()..];
            if (sim) |how| {
                try f.simulate(how[0], how[1], y);
                @memset(m, 0); // simulations produce values only; no undefined magnitude may be copied or scaled
            } else try f.reference(y, m);
            if (which == 1) {
                @memcpy(want[top_k * n ..][0 .. 2 * n], y);
                @memcpy(mag[top_k * n ..][0 .. 2 * n], m);
                continue;
            }
            for (self.picks, 0..) |p, i| {
                @memcpy(want[i * n ..][0..n], y[p * n ..][0..n]);
                @memcpy(mag[i * n ..][0..n], m[p * n ..][0..n]);
            }
        }
        for (want, mag) |*w, *m| {
            const r = if (std.math.isNan(w.*)) w.* else @max(w.*, 0);
            m.* *= 2 * @abs(r);
            w.* = r * r;
        }
    }

    pub fn reference(self: *Pair, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *Pair, acc: c.Accum, order: g.Order, out: []f64) !void {
        const m = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(m);
        return self.outputs(out, m, .{ acc, order });
    }
};

test "HIP pair contract: simulation initializes magnitude scratch and preserves exact positive sums" {
    const a = std.testing.allocator;
    const n = 2;
    const fa = try Affine.init(a, .{ .rows = 1, .n = experts * n, .k = group }, bits, group);
    defer fa.deinit();
    const fb = try Affine.init(a, .{ .rows = 1, .n = 2 * n, .k = group }, bits, group);
    defer fb.deinit();
    for ([_]Affine{ fa, fb }) |f| {
        @memset(f.x, c.toBf16(1));
        @memset(f.codes, 1);
        @memset(f.scales, c.toBf16(1));
        @memset(f.biases, 0);
    }
    // Only the host inputs used by outputs; no runtime, stream or driver is constructed or accessed.
    var pair: Pair = undefined;
    pair.rig.a = a;
    pair.fa = fa;
    pair.fb = fb;
    pair.arg[0].n = n;
    for (&pair.picks, 0..) |*p, i| p.* = i;
    var out: [(top_k + 2) * n]f64 = undefined;
    var mag: [out.len]f64 = undefined;
    for ([_]c.Accum{ .f32, .f16, .bf16 }) |acc| for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        try pair.outputs(&out, &mag, .{ acc, order });
        for (out, mag) |value, magnitude| {
            try std.testing.expectEqual(@as(f64, group * group), value);
            try std.testing.expectEqual(@as(f64, 0), magnitude);
        }
    };
}
