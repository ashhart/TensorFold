//! Grouped Inkling 2 Mini M3 contracts: bucketing and 0-ULP byte-for-byte parity against main's kernels.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const plan = @import("tensorfold").inkling2_metal_host.experts_plan;
const controls = @import("metal_inkling2_experts_controls.zig");
const frozen = @embedFile("baselines/inkling2_experts_main.metal.txt");

const MainArgs = extern struct { k: u32, n: u32, topk: u32, experts: u32 };
const max_terms = 2304;
const Sim = struct { acc: c.Accum, order: g.Order };
const Sample = struct { value: f64, magnitude: f64 };
const Dot = struct { value: f64, magnitude: f64 };

pub const GroupProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    control: bool,
    shape: c.Shape = undefined,
    ids: []i32 = &.{},
    in_ref: rig_.Ref = undefined,
    out_ref: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !GroupProbe {
        const m: controls.Mutation = if (control) .expert_group_drop_last else .none;
        const text = try controls.apply(a, m);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = try host.Pipelines.loadSource(device, text), .control = control };
    }

    pub fn deinit(p: *GroupProbe) void {
        p.a.free(p.ids);
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *GroupProbe) g.Kernel {
        const name = if (p.control) controls.Mutation.expert_group_drop_last.name() else "metal inkling2 experts tf_ink2_expert_group";
        return .{
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = noDepth,
                .out = .raw,
                .nan = .{ .stated = "buckets route pairs by ascending expert ID into <= 8-pair work items preserving pair order" },
                .tiles = .{ .{ .size = 8, .step = 1, .min = 2 }, .{}, .{} },
                .served = &.{
                    .{ .rows = 2, .n = 128, .k = 8 },
                    .{ .rows = 4, .n = 128, .k = 8 },
                    .{ .rows = 8, .n = 128, .k = 8 },
                    .{ .rows = 16, .n = 128, .k = 8 },
                    .{ .rows = 32, .n = 128, .k = 8 },
                    .{ .rows = 64, .n = 128, .k = 8 },
                    .{ .rows = 8, .n = 4, .k = 2 },
                },
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "integer bucketing kernel: no floating-point sums" },
            },
            .ctx = p,
            .vt = &vt,
            .control = if (p.control) .shapes else null,
        };
    }

    const vt = g.hooks(GroupProbe);
    fn noDepth(_: c.Shape) usize {
        return 1;
    }

    pub fn prepare(p: *GroupProbe, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.k == 0 or shape.k > shape.n) return false;
        if (shape.rows * shape.k > plan.max_grouped_pairs) return false;
        p.rig.reset();
        p.a.free(p.ids);
        p.shape = shape;
        const pairs = shape.rows * shape.k;
        p.ids = try p.a.alloc(i32, pairs);
        fillRouteIds(p.ids, shape.n, shape.k, seed);
        p.in_ref = try p.rig.input(std.mem.sliceAsBytes(p.ids));
        p.out_ref = try p.rig.output(plan.groupBytes(shape.rows, shape.k));
        return true;
    }

    pub fn run(p: *GroupProbe) !void {
        const pass = p.rig.begin();
        try host.groupBuckets(&p.pipes, pass.e, .{ .buf = p.in_ref.buf, .off = p.in_ref.off }, .{ .buf = p.out_ref.buf, .off = p.out_ref.off }, .{
            .rows = @intCast(p.shape.rows),
            .k = 32,
            .n = 32,
            .experts = @intCast(p.shape.n),
            .topk = @intCast(p.shape.k),
            .format = .nvfp4,
        });
        try p.rig.finish(pass);
    }

    pub fn count(p: *const GroupProbe) usize {
        return plan.groupBytes(p.shape.rows, p.shape.k);
    }

    pub fn reference(_: *GroupProbe, _: []f64, _: []f64) !void {
        return error.CheckedByOwnRule;
    }

    pub fn check(p: *GroupProbe, _: g.Gate) !?[]const u8 {
        const pairs = p.shape.rows * p.shape.k;
        const words = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out_ref, p.count()));
        const tile_count = words[0];
        var want_tiles: u32 = 0;
        var want_pair_off: u32 = 0;
        const items = words[1..][0..pairs];
        const pair_map = words[1 + pairs ..][0..pairs];
        for (0..p.shape.n) |e| {
            var cnt: u32 = 0;
            for (p.ids) |id| if (id == @as(i32, @intCast(e))) {
                cnt += 1;
            };
            if (cnt == 0) continue;
            const tiles = (cnt + plan.grouped_tile_m - 1) / plan.grouped_tile_m;
            for (0..tiles) |t| {
                if (want_tiles >= tile_count) return try std.fmt.bufPrint(&p.msg, "tile count {d} stopped before expert {d}", .{ tile_count, e });
                const rem = cnt - @as(u32, @intCast(t)) * plan.grouped_tile_m;
                const len = @min(rem, plan.grouped_tile_m);
                const off = want_pair_off + @as(u32, @intCast(t)) * plan.grouped_tile_m;
                const want_item = (@as(u32, @intCast(e)) & 0xffff) | ((off & 0xfff) << 16) | (((@as(u32, len) - 1) & 0xf) << 28);
                if (items[want_tiles] != want_item) return try std.fmt.bufPrint(&p.msg, "item {d} was 0x{x} where host expected 0x{x}", .{ want_tiles, items[want_tiles], want_item });
                want_tiles += 1;
            }
            for (p.ids, 0..) |id, idx| if (id == @as(i32, @intCast(e))) {
                if (pair_map[want_pair_off] != idx) return try std.fmt.bufPrint(&p.msg, "pair_map[{d}] was {d} where host expected {d}", .{ want_pair_off, pair_map[want_pair_off], idx });
                want_pair_off += 1;
            };
        }
        if (tile_count != want_tiles) return try std.fmt.bufPrint(&p.msg, "tile count was {d} where host expected {d}", .{ tile_count, want_tiles });
        return null;
    }
};

pub const Kind = enum { nv_gu, nv_down, mx_gu, mx_down };
pub const ControlKind = enum { plain, block_drop, tile_drop, arith };

pub const GemmProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    base: mtl.Pipeline,
    kind: Kind,
    ctl_kind: ControlKind,
    ctl: controls.Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u8 = &.{},
    scales: []u8 = &.{},
    amax: []f32 = &.{},
    denominator: []f32 = &.{},
    ids: []i32 = &.{},
    refs: [7]rig_.Ref = undefined,
    group_ref: rig_.Ref = undefined,
    base_out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, ctl_kind: ControlKind) !GemmProbe {
        const m: controls.Mutation = switch (ctl_kind) {
            .plain => .none,
            .block_drop => .grouped_block_drop,
            .tile_drop => .grouped_tile_drop,
            .arith => switch (kind) {
                .nv_gu => .grouped_nv_gu_up,
                .nv_down => .nv_down_phi,
                .mx_gu => .grouped_mx_gu_up,
                .mx_down => .grouped_mx_down_scale,
            },
        };
        const text = try controls.apply(a, m);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        const lib = try mtl.Library.fromSource(device, frozen, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const base = try mtl.Pipeline.init(device, lib, baseSymbol(kind), false);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .base = base, .kind = kind, .ctl_kind = ctl_kind, .ctl = m };
    }

    fn clear(p: *GemmProbe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.a.free(p.scales);
        p.a.free(p.amax);
        p.a.free(p.denominator);
        p.a.free(p.ids);
        p.x = &.{};
        p.w = &.{};
        p.scales = &.{};
        p.amax = &.{};
        p.denominator = &.{};
        p.ids = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *GemmProbe) void {
        p.clear();
        p.base.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn specName(kind: Kind, ctl_kind: ControlKind) []const u8 {
        return switch (ctl_kind) {
            .plain => switch (kind) {
                .nv_gu => "metal inkling2 experts tf_ink2_nvfp4_gu_grouped",
                .nv_down => "metal inkling2 experts tf_ink2_nvfp4_down_grouped",
                .mx_gu => "metal inkling2 experts tf_ink2_mxfp8_gu_grouped",
                .mx_down => "metal inkling2 experts tf_ink2_mxfp8_down_grouped",
            },
            .block_drop => switch (kind) {
                .nv_gu => "control: metal inkling2 experts tf_ink2_nvfp4_gu_grouped output block dropped",
                .nv_down => "control: metal inkling2 experts tf_ink2_nvfp4_down_grouped output block dropped",
                .mx_gu => "control: metal inkling2 experts tf_ink2_mxfp8_gu_grouped output block dropped",
                .mx_down => "control: metal inkling2 experts tf_ink2_mxfp8_down_grouped output block dropped",
            },
            .tile_drop => switch (kind) {
                .nv_gu => "control: metal inkling2 experts tf_ink2_nvfp4_gu_grouped tile pair dropped",
                .nv_down => "control: metal inkling2 experts tf_ink2_nvfp4_down_grouped tile pair dropped",
                .mx_gu => "control: metal inkling2 experts tf_ink2_mxfp8_gu_grouped tile pair dropped",
                .mx_down => "control: metal inkling2 experts tf_ink2_mxfp8_down_grouped tile pair dropped",
            },
            .arith => switch (kind) {
                .nv_gu => "control: metal inkling2 experts tf_ink2_nvfp4_gu_grouped up row",
                .nv_down => "control: metal inkling2 experts tf_ink2_nvfp4_down_grouped phi",
                .mx_gu => "control: metal inkling2 experts tf_ink2_mxfp8_gu_grouped up row",
                .mx_down => "control: metal inkling2 experts tf_ink2_mxfp8_down_grouped scale",
            },
        };
    }

    pub fn kernel(p: *GemmProbe) g.Kernel {
        const gu = p.kind == .nv_gu or p.kind == .mx_gu;
        return .{
            .spec = .{
                .name = specName(p.kind, p.ctl_kind),
                .accum = .f32,
                .depth = depthFn,
                .ops = if (gu) 12 else 3,
                .mids = if (gu) 3 else 0,
                .out = .bf16,
                .replaces = "main's kernel on its one-group-a-pair launch (5e390519d0)",
                .tiles = .{ .{ .size = 8, .step = 1, .min = 2 }, .{ .size = 64 }, .{} },
                .served = if (gu) &.{
                    .{ .rows = 2, .n = 65, .k = 64 },
                    .{ .rows = 2, .n = 65, .k = 2304 },
                    .{ .rows = 4, .n = 1152, .k = 2304 },
                    .{ .rows = 8, .n = 129, .k = 64 },
                    .{ .rows = 16, .n = 65, .k = 64 },
                    .{ .rows = 32, .n = 65, .k = 64 },
                    .{ .rows = 64, .n = 1152, .k = 2304 },
                } else &.{
                    .{ .rows = 2, .n = 65, .k = 64 },
                    .{ .rows = 2, .n = 65, .k = 1152 },
                    .{ .rows = 4, .n = 2304, .k = 1152 },
                    .{ .rows = 8, .n = 129, .k = 64 },
                    .{ .rows = 16, .n = 65, .k = 64 },
                    .{ .rows = 32, .n = 65, .k = 64 },
                    .{ .rows = 64, .n = 2304, .k = 1152 },
                },
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms and explicit BF16 nodes" },
            },
            .ctx = p,
            .vt = &vt,
            .control = p.ctl.gate(),
        };
    }

    const vt = blk: {
        var table = g.hooks(GemmProbe);
        table.check = null;
        break :blk table;
    };

    fn depthFn(s: c.Shape) usize {
        return s.k + 3;
    }
    fn isNv(k: Kind) bool {
        return k == .nv_gu or k == .nv_down;
    }
    fn isGu(k: Kind) bool {
        return k == .nv_gu or k == .mx_gu;
    }
    fn baseSymbol(k: Kind) [:0]const u8 {
        return switch (k) {
            .nv_gu => "tf_ink2_nvfp4_gu",
            .nv_down => "tf_ink2_nvfp4_down",
            .mx_gu => "tf_ink2_mxfp8_gu",
            .mx_down => "tf_ink2_mxfp8_down",
        };
    }
    fn expertCount(p: *const GemmProbe) u32 {
        return if (p.shape.n > 256) 16 else 128;
    }
    fn call(p: *const GemmProbe) host.Call {
        return .{ .rows = @intCast(p.shape.rows), .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .experts = p.expertCount(), .topk = 8, .format = if (isNv(p.kind)) .nvfp4 else .mxfp8 };
    }

    pub fn prepare(p: *GemmProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 2 or shape.rows > 64 or shape.n == 0 or shape.n > 2304 or shape.k == 0 or shape.k > max_terms or shape.k % 32 != 0) return false;
        p.clear();
        p.shape = shape;
        const nv = isNv(p.kind);
        const gu = isGu(p.kind);
        const experts: usize = p.expertCount();
        const topk: usize = 8;
        const routes = shape.rows * topk;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nx = if (gu) shape.rows * shape.k else routes * shape.k;
        p.x = try p.a.alloc(u16, nx);
        for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
        p.ids = try p.a.alloc(i32, routes);
        fillRouteIds(p.ids, experts, topk, seed);
        const rows_per_expert: usize = if (gu) 2 * shape.n else shape.n;
        const matrices = experts * rows_per_expert;
        p.w = try p.a.alloc(u8, matrices * (if (nv) shape.k / 2 else shape.k));
        p.scales = try p.a.alloc(u8, matrices * (shape.k / @as(usize, if (nv) 16 else 32)));
        rng.bytes(p.w);
        for (p.scales, 0..) |*v, i| v.* = if (nv) @intCast(0x30 + (i % 17)) else @intCast(125 + (i % 5));
        if (!nv) for (p.w) |*v| if (v.* & 0x7f == 0x7f) {
            v.* = (v.* & 0x80) | 0x7e;
        };
        if (nv) {
            p.amax = try p.a.alloc(f32, experts);
            for (p.amax, 0..) |*v, i| v.* = 2688.125 + @as(f32, @floatFromInt(i % 17)) * 0.375;
            p.denominator = try p.a.alloc(f32, 1);
            p.denominator[0] = 2695.75;
        }
        if (fill == .stress) {
            for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
            @memset(p.scales, if (nv) 0x38 else 127);
        } else if (fill == .nonfinite and p.x.len >= 3) {
            p.x[0] = c.toBf16(std.math.nan(f32));
            p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
            p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
        }
        const out = try p.rig.output(p.count() * 2);
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(p.w);
        p.refs[2] = try p.rig.input(p.scales);
        if (nv) {
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.amax));
            p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.denominator));
            p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[6] = out;
        } else {
            p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
            p.refs[4] = out;
        }
        p.group_ref = try p.rig.scratch(plan.groupBytes(shape.rows, topk));
        p.base_out = try p.rig.scratch(p.count() * 2);
        return true;
    }

    pub fn replaced(p: *GemmProbe) ![]const u8 {
        const len = p.count() * 2;
        @memset(p.rig.bytes(p.base_out, len), 0xff);
        const bound: usize = if (isNv(p.kind)) 7 else 5;
        const c_ = p.call();
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.base);
        for (p.refs[0 .. bound - 1], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(p.base_out.buf, p.base_out.off, bound - 1);
        e.setValue(MainArgs{ .k = c_.k, .n = c_.n, .topk = c_.topk, .experts = c_.experts }, bound);
        e.dispatchGroups(mtl.Size.of(c_.rows * c_.topk, 1, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_out, len);
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *GemmProbe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const c_ = p.call();
        const ids_ref = if (isNv(p.kind)) p.refs[5] else p.refs[3];
        try host.groupBuckets(&p.pipes, e, hr(ids_ref), hr(p.group_ref), c_);
        switch (p.kind) {
            .nv_gu => try host.nvfp4GuGrouped(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.group_ref), hr(p.refs[6]), c_),
            .nv_down => try host.nvfp4DownGrouped(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), hr(p.group_ref), hr(p.refs[6]), c_),
            .mx_gu => try host.mxfp8GuGrouped(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.group_ref), hr(p.refs[4]), c_),
            .mx_down => try host.mxfp8DownGrouped(&p.pipes, e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.group_ref), hr(p.refs[4]), c_),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *const GemmProbe) usize {
        return p.shape.rows * 8 * p.shape.n;
    }

    fn e2(code: u8) f64 {
        const v = ([_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[code & 7];
        return if (code & 8 != 0) -v else v;
    }
    fn e4(code: u8) f64 {
        const exponent = (code >> 3) & 15;
        const mantissa = code & 7;
        if (exponent == 15 and mantissa == 7) return std.math.nan(f64);
        const v = if (exponent == 0) @as(f64, @floatFromInt(mantissa)) / 512 else std.math.ldexp(@as(f64, 8 + @as(u16, mantissa)), @as(i32, exponent) - 10);
        return if (code & 0x80 != 0) -v else v;
    }
    fn weight(p: *const GemmProbe, expert: usize, row: usize, col: usize) f64 {
        const k = p.shape.k;
        const rows_per_expert: usize = if (isGu(p.kind)) 2 * p.shape.n else p.shape.n;
        const matrix_row = expert * rows_per_expert + row;
        if (!isNv(p.kind)) {
            return bf(std.math.ldexp(e4(p.w[matrix_row * k + col]), @as(i32, p.scales[matrix_row * (k / 32) + col / 32]) - 127));
        }
        const codes = p.w[matrix_row * (k / 2) + col / 2];
        const code = if (col % 2 == 0) codes & 15 else codes >> 4;
        const scale = p.scales[matrix_row * (k / 16) + col / 16];
        const factor = bf(p.amax[expert]) / bf(p.denominator[0]);
        return bf(@as(f32, @floatCast(e2(code) * e4(scale) * factor)));
    }
    fn dot(p: *const GemmProbe, expert: usize, row: usize, x: []const u16, sim: ?Sim) Dot {
        var terms: [max_terms]f64 = undefined;
        var magnitude: f64 = 0;
        for (0..p.shape.k) |col| {
            const term = p.weight(expert, row, col) * @as(f64, c.fromBf16(x[col]));
            terms[col] = term;
            magnitude += @abs(term);
        }
        const value = if (sim) |s| rows.sumIn(s.acc, s.order, terms[0..p.shape.k]) else blk: {
            var sum: f64 = 0;
            for (terms[0..p.shape.k]) |term| sum += term;
            break :blk sum;
        };
        return .{ .value = value, .magnitude = magnitude };
    }
    fn evaluate(p: *const GemmProbe, at: usize, sim: ?Sim) Sample {
        const pair = at / p.shape.n;
        const n = at % p.shape.n;
        const token = pair / 8;
        const expert: usize = @intCast(p.ids[pair]);
        const x = if (isGu(p.kind)) p.x[token * p.shape.k ..][0..p.shape.k] else p.x[pair * p.shape.k ..][0..p.shape.k];
        if (!isGu(p.kind)) {
            const d = p.dot(expert, n, x, sim);
            return .{ .value = bf(d.value), .magnitude = d.magnitude };
        }
        const gd = p.dot(expert, 2 * n, x, sim);
        const ud = p.dot(expert, 2 * n + 1, x, sim);
        const gate = bf(gd.value);
        const up = bf(ud.value);
        const left = 4 * std.math.tanh(gate / 4) * (1 / (1 + @exp(-gate)));
        return .{ .value = bf(left * (25 * std.math.tanh(up / 25))), .magnitude = gd.magnitude + ud.magnitude };
    }
    pub fn reference(p: *GemmProbe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*v, *m, i| {
            const sample = p.evaluate(i, null);
            v.* = sample.value;
            m.* = sample.magnitude;
        }
    }
    pub fn simulate(p: *GemmProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.evaluate(i, .{ .acc = acc, .order = order }).value;
    }
};

/// Mix distinct, repeated and multi-tile (> 8 pair) expert picks while keeping slots per token valid.
fn fillRouteIds(ids: []i32, experts: usize, topk: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed ^ 0x9e3779b97f4a7c15);
    const rng = prng.random();
    const num_rows = ids.len / topk;
    for (0..num_rows) |r| {
        for (0..topk) |slot| {
            const id: usize = if (slot < 2 and experts >= 2)
                slot
            else if (slot < topk / 2 and experts >= 4)
                rng.uintLessThan(usize, @min(experts, 4))
            else
                rng.uintLessThan(usize, experts);
            ids[r * topk + slot] = @intCast(id);
        }
    }
}

pub const Suite = struct {
    group_probes: [2]GroupProbe = undefined,
    group_made: usize = 0,
    gemm_probes: [std.enums.values(Kind).len * std.enums.values(ControlKind).len]GemmProbe = undefined,
    gemm_made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.group_probes[0..s.group_made]) |*p| p.deinit();
        for (s.gemm_probes[0..s.gemm_made]) |*p| p.deinit();
    }

    pub fn collect(s: *Suite, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |control| {
            const label = if (control) controls.Mutation.expert_group_drop_last.name() else "metal inkling2 experts tf_ink2_expert_group";
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.group_probes[s.group_made] = try GroupProbe.init(a, device, queue, control);
            s.group_made += 1;
            try kernels.append(a, s.group_probes[s.group_made - 1].kernel());
        }
        for (std.enums.values(Kind)) |kind| for (std.enums.values(ControlKind)) |ctl| {
            const label = GemmProbe.specName(kind, ctl);
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.gemm_probes[s.gemm_made] = try GemmProbe.init(a, device, queue, kind, ctl);
            s.gemm_made += 1;
            try kernels.append(a, s.gemm_probes[s.gemm_made - 1].kernel());
        };
    }
};
