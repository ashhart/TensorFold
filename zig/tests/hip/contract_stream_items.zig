//! Routed item-rows plain stream tile (tiles/stream.hpp): skips dead-row folds on RDNA3.5, HIP-byte-equal.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const Kernels = @FieldType(hip.Launcher, "affine");
const Arg = @typeInfo(@TypeOf(Kernels.streamGo)).@"fn".param_types[2].?;

const bits = 4;
const experts = 12; // the experts a plan picks from; a served layer has more, and the kernel reads only the picked ones
const top_k = 3;
const most_rows = 8;

const Case = struct { group: usize, name: []const u8, served: []const c.Shape };

/// Served single-round (K <= 1024) plain routed down shapes: Qwen3.6-35B-A3B and Flash Next.
pub const cases = [_]Case{
    .{ .group = 64, .name = "hip affine_dot2_stream item-rows (4-bit, group 64, fp32 out)", .served = &.{
        .{ .rows = 3, .n = 2048, .k = 512 }, .{ .rows = 4, .n = 2048, .k = 512 },
    } },
    .{ .group = 32, .name = "hip affine_dot2_stream item-rows (4-bit, group 32, fp32 out)", .served = &.{
        .{ .rows = 3, .n = 2560, .k = 640 }, .{ .rows = 4, .n = 2560, .k = 640 },
    } },
};

/// The control kernels in the controls' code object, at 2 and 4 columns a lane: each item one row short.
pub const control_symbols = [2][:0]const u8{ "tf_ctl_items_short_plain_2", "tf_ctl_items_short_plain_4" };

fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / 32 + 8;
}

pub fn spec(case: Case) c.Spec {
    return .{
        .name = case.name,
        .accum = .f32,
        .depth = depth,
        .ops = 0,
        .mids = 0,
        .out = .f32,
        .replaces = "affine_dot2_stream with item_rows off on the same plan",
        .nan = .{ .stated = "the tile's bytes and every float64 output class" },
        .served = case.served,
        .tiles = .{ .{ .size = 4, .min = 3 }, .{ .size = 16, .step = 2, .min = 2 }, .{ .size = 256, .step = case.group, .min = case.group } },
    };
}

pub const StreamItems = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    index: usize,
    planted: ?[2]hip.Function = null,
    arena: std.heap.ArenaAllocator,
    f: ?Affine = null,
    shape: c.Shape = undefined,
    picks: [most_rows][top_k]usize = undefined,
    items: c_int = 0,
    arg: Arg = undefined,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) StreamItems {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .index = i, .arena = .init(a) };
    }

    pub fn deinit(self: *StreamItems) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *StreamItems) g.Kernel {
        var s = spec(cases[self.index]);
        if (self.planted != null) s.name = "control: the item-rows tile with every item one row short";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.planted != null) .shapes else null };
    }

    const vt = g.hooks(StreamItems);

    fn case(self: *const StreamItems) Case {
        return cases[self.index];
    }

    fn pairs(self: *const StreamItems) usize {
        return self.shape.rows * top_k;
    }

    pub fn prepare(self: *StreamItems, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const cfg = self.case();
        const max_k = hip.launches.stream_pick.single_round_k;
        if (shape.rows < 3 or shape.rows > most_rows or shape.n == 0 or shape.k > max_k or shape.k % cfg.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        var f = try Affine.init(a, .{ .rows = self.pairs(), .n = experts * shape.n, .k = shape.k }, bits, @intCast(cfg.group));
        try f.fill(fill, seed);
        if (self.k == .f16) for (f.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0; // fp16 normals only, so each bf16 value is exact
        };
        // the first 4 tokens pick expert 0 and the rest take the others in turn: the largest item holds up to 4 rows
        var next: usize = 0;
        for (self.picks[0..shape.rows], 0..) |*row, t| for (row, 0..) |*p, s| {
            if (s == 0 and t < 4) {
                p.* = 0;
                continue;
            }
            p.* = 1 + next % (experts - 1);
            next += 1;
        };
        if (fill == .nonfinite) {
            const groups = shape.k / cfg.group;
            f.scales[(self.picks[0][1] * shape.n + 3) * groups] = c.toBf16(std.math.nan(f32));
            f.biases[(self.picks[1][2] * shape.n + 5) * groups + 1] = c.toBf16(std.math.inf(f32));
        }
        self.f = f;
        // items in expert order, each its pairs in order: (expert, first member, count)
        var items: [3 * experts]i32 = undefined;
        var members: [most_rows * top_k]i32 = undefined;
        var held: usize = 0;
        var at: usize = 0;
        for (0..experts) |e| {
            const first = at;
            for (0..self.pairs()) |p| if (self.picks[p / top_k][p % top_k] == e) {
                members[at] = @intCast(p);
                at += 1;
            };
            if (at == first) continue;
            items[3 * held ..][0..3].* = .{ @intCast(e), @intCast(first), @intCast(at - first) };
            held += 1;
        }
        self.items = @intCast(held);
        const xb = try a.alloc(u8, f.x.len * 2);
        for (f.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const w = try a.alloc(u32, f.shape.n * shape.k * bits / 32);
        f.words(w);
        const out = try self.rig.output(self.count() * 4);
        self.base = try self.rig.scratch(self.count() * 4);
        self.arg = .{
            .x = try self.rig.input(xb),
            .words = try self.rig.input(std.mem.sliceAsBytes(w)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = out,
            .m = @intCast(@min(shape.rows, 4)), // expert 0's item, the largest
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = bits,
            .group = @intCast(cfg.group),
            .fp16 = @intFromBool(self.k == .f16),
            .route = .{
                .items = try self.rig.input(std.mem.sliceAsBytes(items[0 .. 3 * held])),
                .members = try self.rig.input(std.mem.sliceAsBytes(members[0..self.pairs()])),
                .x_div = 1,
            },
        };
        return true;
    }

    fn launch(self: *StreamItems, kern: *const Kernels, arg: Arg) !void {
        try kern.routed(self.l.r, arg, self.items, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The launcher as served, or with the planted control in place of the variant at both column counts.
    pub fn run(self: *StreamItems) !void {
        var kern = self.l.affine;
        if (self.planted) |p| kern.variants.override = p;
        try self.launch(&kern, self.arg);
    }

    /// The same plan on the tile the variant replaces: the launcher with no variants.
    pub fn replaced(self: *StreamItems) ![]const u8 {
        var kern = self.l.affine;
        kern.variants = .{};
        var arg = self.arg;
        arg.out = self.base;
        try self.launch(&kern, arg);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    /// Every non-finite plant must give the replaced tile's bytes and each float64 output class.
    pub fn check(self: *StreamItems, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        if (!std.mem.eql(u8, out, try self.replaced())) return "non-finite output differs from the tile's bytes";
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(.f32, out, i)) != c.classOf(w)) return "non-finite output class differs from float64";
        }
        return null;
    }

    pub fn count(self: *StreamItems) usize {
        return self.pairs() * self.shape.n;
    }

    /// Expert `e`'s matrix against x row `r` alone, as a plain product of the contract's affine family.
    fn view(self: *const StreamItems, r: usize, e: usize) Affine {
        var v = self.f.?;
        const n = self.shape.n;
        const k = self.shape.k;
        const groups = k / self.case().group;
        v.shape = .{ .rows = 1, .n = n, .k = k };
        v.x = v.x[r * k ..][0..k];
        v.codes = v.codes[e * n * k ..][0 .. n * k];
        v.scales = v.scales[e * n * groups ..][0 .. n * groups];
        v.biases = v.biases[e * n * groups ..][0 .. n * groups];
        return v;
    }

    /// Each pair's fp32 sums (or their simulation in an accumulator).
    fn outputs(self: *StreamItems, want: []f64, mag: []f64, sim: ?struct { c.Accum, g.Order }) !void {
        const n = self.shape.n;
        const y = try self.rig.a.alloc(f64, 2 * n);
        defer self.rig.a.free(y);
        for (0..self.pairs()) |p| {
            const v = self.view(p, self.picks[p / top_k][p % top_k]);
            if (sim) |how| {
                try v.simulate(how[0], how[1], y[0..n]);
                @memset(y[n..], 0);
            } else try v.referenceSerial(y[0..n], y[n..]);
            @memcpy(want[p * n ..][0..n], y[0..n]);
            @memcpy(mag[p * n ..][0..n], y[n..]);
        }
    }

    pub fn reference(self: *StreamItems, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *StreamItems, acc: c.Accum, order: g.Order, out: []f64) !void {
        const m = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(m);
        return self.outputs(out, m, .{ acc, order });
    }
};
