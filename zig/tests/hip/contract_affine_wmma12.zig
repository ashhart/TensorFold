//! affine_wmma12_gemm under the contract on gfx12: BF16 x times MLX affine codes on the WMMA tile, fp32 out.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Kernels = @FieldType(hip.Launcher, "affine");
const Arg = @typeInfo(@TypeOf(Kernels.streamGo)).@"fn".param_types[2].?;
const Kind = @typeInfo(@TypeOf(Kernels.blockGo)).@"fn".param_types[5].?;
const rows_k = hip.launches.matrix_rows;

/// A width and group on a block: the 128-row tile, or tier `rows` of the few-row blocks (64, then 32 rows).
const Case = struct { bits: u8, group: usize, rows: ?usize = null, name: []const u8 };

/// Flash Next's oQ2 widths (2-bit experts in groups of 32, 3-8 bits in 64 and 128) on the 128/64/32/16-row blocks.
pub const cases = blk: {
    const widths = [_][2]usize{ .{ 2, 32 }, .{ 3, 64 }, .{ 4, 64 }, .{ 5, 128 }, .{ 6, 64 }, .{ 8, 128 } };
    const few = [_][3]usize{
        .{ 2, 32, 0 },              .{ 3, 64, 0 },               .{ 4, 64, 0 },              .{ 8, 128, 0 },
        .{ 2, 32, 1 },              .{ 3, 64, 1 },               .{ 4, 64, 1 },              .{ 5, 128, 1 },
        .{ 6, 64, 1 },              .{ 8, 128, 1 },              .{ 2, 32, rows_k.kp_tier }, .{ 3, 64, rows_k.kp_tier },
        .{ 4, 64, rows_k.kp_tier }, .{ 5, 128, rows_k.kp_tier }, .{ 6, 64, rows_k.kp_tier }, .{ 8, 128, rows_k.kp_tier },
    };
    var out: [widths.len + few.len]Case = undefined;
    for (widths, 0..) |w, i| out[i] = .{
        .bits = @intCast(w[0]),
        .group = w[1],
        .name = std.fmt.comptimePrint("hip affine_wmma12_gemm ({d}-bit, group {d}, fp32 out)", .{ w[0], w[1] }),
    };
    for (few, widths.len..) |w, i| out[i] = .{
        .bits = @intCast(w[0]),
        .group = w[1],
        .rows = w[2],
        .name = std.fmt.comptimePrint("hip affine_wmma12_rows<{d} x {d}> ({d}-bit, group {d}, fp32 out)", .{ rows_k.row_tiers[w[2]], rows_k.col_tiers[w[2]], w[0], w[1] }),
    };
    break :blk out;
};

/// The case a planted control runs on: its kernel decodes 4 bits on the 128-row tile, or on the 16-row block.
fn controlCase(comptime rows: ?usize) usize {
    for (cases, 0..) |case, i| if (case.bits == 4 and case.rows == rows) return i;
    @compileError("no 4-bit case for a control");
}

/// A group's products (at most 128, any order), its scale and its bias, then the groups' sum in k order.
fn depth(s: c.Shape) usize {
    return @min(s.k, 128) + 2 * (s.k / 32) + 8;
}

/// A prompt chunk of 1,024 rows on Flash Next's expert and attention products, 4,096 rows, then the block's edges.
const served = [_]c.Shape{
    .{ .rows = 1024, .n = 640, .k = 2560 }, .{ .rows = 1024, .n = 2560, .k = 640 }, .{ .rows = 4096, .n = 1280, .k = 2560 },
    .{ .rows = 300, .n = 2560, .k = 6144 }, .{ .rows = 22, .n = 1280, .k = 2560 },  .{ .rows = 17, .n = 65, .k = 128 },
    .{ .rows = 129, .n = 130, .k = 256 },   .{ .rows = 128, .n = 128, .k = 384 },   .{ .rows = 1, .n = 2560, .k = 2560 },
};

/// The planted controls: each breaks one thing the served tile relies on, and must go red at gate 3.
pub const Plant = enum {
    /// The tile without its K loop's barriers.
    unordered,
    /// Each row's bias folded with the next row's sum of x.
    stale_sums,
    /// The 16-row block folding its rounds without barriers.
    kp_unordered,
    /// The 16-row block folding a row's bias with the next row's sum of x: red only on served shapes of several rows.
    kp_stale_sums,

    pub fn symbol(p: Plant) [:0]const u8 {
        return switch (p) {
            .unordered => "tf_ctl_affine12_unordered",
            .stale_sums => "tf_ctl_affine12_stale_sums",
            .kp_unordered => "tf_ctl_affine12kp_unordered",
            .kp_stale_sums => "tf_ctl_affine12kp_stale_sums",
        };
    }

    fn name(p: Plant) []const u8 {
        return switch (p) {
            .unordered => "control: WMMA affine tile without its K-step barriers",
            .stale_sums => "control: WMMA affine tile folding biases with the next row's sums",
            .kp_unordered => "control: K-parallel WMMA affine block without its fold barriers",
            .kp_stale_sums => "control: K-parallel WMMA affine block folding biases with the next row's sums",
        };
    }

    /// The tier of the few-row blocks the control's kernel takes the place of, or null for the 128-row tile.
    pub fn tier(p: Plant) ?usize {
        return switch (p) {
            .unordered, .stale_sums => null,
            .kp_unordered, .kp_stale_sums => kp_tier,
        };
    }

    /// The case the control runs on.
    pub fn case(p: Plant) usize {
        return if (p.tier() == null) comptime controlCase(null) else comptime controlCase(kp_tier);
    }

    fn unordered_(p: Plant) bool {
        return p == .unordered or p == .kp_unordered;
    }
};

/// The few-row tier of the K-parallel block.
const kp_tier = rows_k.kp_tier;

/// Launches of the unordered control a shape: gate 3 counts it red if any differs from the reference.
const race_runs = 8;

pub const Affine12 = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    index: usize,
    /// A planted control (4-bit, from the controls' object) in place of the served tile.
    planted: ?hip.Function = null,
    plant: Plant = .unordered,
    f: ?contract.affine.Affine = null,
    arg: Arg = undefined,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, i: usize) Affine12 {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .index = i };
    }

    pub fn deinit(self: *Affine12) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Affine12) g.Kernel {
        const cfg = cases[self.index];
        const plant: ?Plant = if (self.planted != null) self.plant else null;
        return .{
            .spec = .{
                .name = if (plant) |p| p.name() else cfg.name,
                .accum = .f32,
                .depth = depth,
                .out = .f32,
                .served = &served,
                .tiles = .{ .{ .size = if (cfg.rows) |t| rows_k.row_tiers[t] else 128 }, .{ .size = if (cfg.rows) |t| rows_k.col_tiers[t] else 128 }, .{ .size = 6144, .step = cfg.group, .min = cfg.group } },
            },
            .ctx = self,
            .vt = &vt,
            .control = if (plant != null) .shapes else null,
        };
    }

    const vt = g.hooks(Affine12);

    pub fn prepare(self: *Affine12, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const cfg = cases[self.index];
        if (shape.rows == 0 or shape.n == 0 or shape.k % cfg.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const f = try contract.affine.Affine.init(a, shape, cfg.bits, cfg.group);
        try f.fill(fill, seed);
        self.f = f;
        const words = try a.alloc(u32, shape.n * shape.k * cfg.bits / 32);
        f.words(words);
        self.base = try self.rig.scratch(f.outputs() * 4);
        self.arg = .{
            .x = try self.rig.input(std.mem.sliceAsBytes(f.x)),
            .words = try self.rig.input(std.mem.sliceAsBytes(words)),
            .scale = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.scales)), .kind = 1 },
            .bias = .{ .p = try self.rig.input(std.mem.sliceAsBytes(f.biases)), .kind = 1 },
            .out = try self.rig.output(f.outputs() * 4),
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = cfg.bits,
            .group = @intCast(cfg.group),
            .fp16 = 0,
        };
        return true;
    }

    pub fn run(self: *Affine12) !void {
        const h = self.rig.s.handle;
        const f = self.planted orelse {
            if (cases[self.index].rows) |t| {
                try rows_k.go(&self.l.affine, .{ .r = self.l.r, .s = h, .arg = self.arg, .items = 1 }, t);
            } else try self.l.affine.blockGo(self.l.r, self.arg, h, 1, @as(Kind, .matrix));
            return self.rig.s.synchronize();
        };
        // an unordered control is a race: up to `race_runs` launches, to the first that differs from the served tile
        if (!self.plant.unordered_()) {
            try self.launchPlanted(f, self.arg);
            return self.rig.s.synchronize();
        }
        const want = try self.arena.allocator().dupe(u8, try self.servedBytes());
        for (0..race_runs) |_| {
            try self.launchPlanted(f, self.arg);
            try self.rig.s.synchronize();
            if (!std.mem.eql(u8, try self.rig.outputBytes(), want)) return;
        }
    }

    /// A control on the grid and block of the tile it takes the place of.
    fn launchPlanted(self: *Affine12, f: hip.Function, arg: Arg) !void {
        const h = self.rig.s.handle;
        if (self.plant.tier()) |t| {
            try rows_k.goWith(f, .{ .r = self.l.r, .s = h, .arg = arg, .items = 1 }, t);
        } else try self.l.affine.blockWith(self.l.r, f, arg, h);
    }

    /// The served tile's fp32 bytes on the same inputs, in the scratch buffer.
    fn servedBytes(self: *Affine12) ![]const u8 {
        var plain = self.arg;
        plain.out = self.base;
        const h = self.rig.s.handle;
        if (self.plant.tier()) |t| {
            try rows_k.go(&self.l.affine, .{ .r = self.l.r, .s = h, .arg = plain, .items = 1 }, t);
        } else try self.l.affine.blockGo(self.l.r, plain, h, 1, @as(Kind, .matrix));
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.f.?.outputs() * 4);
    }

    pub fn count(self: *Affine12) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Affine12, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Affine12, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
    }
};
