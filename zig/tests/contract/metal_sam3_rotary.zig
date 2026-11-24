//! SAM3's rotary kernels (rotary.metal): the table, memory attention rows' rotation and the trunk's split rotation.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const tbl = @import("metal_sam3_rotary_table.zig");
const c = contract.spec;
const g = contract.gates;
const vt = g.hooks(Turn);

const RotarySplit = extern struct { rows: u32, width: u32, head_width: u32, positions: u32 };
const Kind = enum { rotate32, rotate256, split32, split16 };

const memory_positions = 72 * 72;
const split_width = 64;

const Edit = struct { function: []const u8, edit: [2][]const u8 };
const edits = [_]Edit{
    .{ .function = "sam3_rotary_table", .edit = tbl.table_edit },
    .{ .function = "sam3_rotate", .edit = .{ "y4[i] = sam3_rotary_quad(x4[i], cs);", "y4[i] = sam3_rotary_quad(x4[i], float4(cs.x, -cs.y, cs.z, -cs.w));" } },
    .{ .function = "sam3_rotate_split_f32", .edit = .{ "b = a + quads;", "b = a;" } },
    .{ .function = "sam3_rotate_split_f16", .edit = .{ "const float4 vv = x4[a + 2u * quads];", "const float4 vv = x4[a + quads];" } },
};

pub fn text(a: std.mem.Allocator, control: ?[]const u8) ![]u8 {
    const edit: ?[2][]const u8 = if (control) |f| for (edits) |e| {
        if (std.mem.eql(u8, e.function, f)) break e.edit;
    } else return error.UnknownLaunch else null;
    return tbl.fileText(a, "struct Sam3Rotary {", "// SAM3 mixed-precision trunk kernels", edit);
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    return text(a, if (control) name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..] else null);
}

fn two(_: c.Shape) usize {
    return 2;
}

/// rotate: rows, n = heads x D, k = trailing pointer-token rows left unrotated (memory keys; queries 0).
const rotate32_served = [_]c.Shape{ .{ .rows = 5184, .n = 256, .k = 0 }, .{ .rows = 7 * 5184 + 16 * 16, .n = 256, .k = 256 }, .{ .rows = 5184 + 16, .n = 256, .k = 16 } };
const rotate256_served = [_]c.Shape{ .{ .rows = 5184, .n = 256, .k = 0 }, .{ .rows = 7 * 5184 + 16 * 4, .n = 256, .k = 64 }, .{ .rows = 5184 + 4, .n = 256, .k = 4 } };
/// split: rows, n = vision width W, k = table positions (24 x 24 tiles or the 72 x 72 grid).
const split32_served = [_]c.Shape{ .{ .rows = 5184, .n = 1024, .k = 576 }, .{ .rows = 5184, .n = 1024, .k = 5184 } };
const split16_served = [_]c.Shape{.{ .rows = 5184, .n = 1024, .k = 5184 }};

const Turn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    control: bool,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    table: []f32 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    fn label(kind: Kind, control: bool) []const u8 {
        return switch (kind) {
            .rotate32 => if (control) "control: sam3 rotary rotate flips the partner's sine" else "metal sam3 rotary rotate SAM3.1 memory heads",
            .rotate256 => "metal sam3 rotary rotate SAM3 memory head",
            .split32 => if (control) "control: sam3 rotary split f32 takes k from the q third" else "metal sam3 rotary split f32",
            .split16 => if (control) "control: sam3 rotary split f16 copies v from the k third" else "metal sam3 rotary split f16",
        };
    }

    fn spec(p: *const Turn) c.Spec {
        return .{
            .name = label(p.kind, p.control),
            .accum = .f32,
            .depth = two,
            .out = if (p.kind == .split16) .f16 else .f32,
            .served = switch (p.kind) {
                .rotate32 => &rotate32_served,
                .rotate256 => &rotate256_served,
                .split32 => &split32_served,
                .split16 => &split16_served,
            },
            .tiles = .{ .{ .size = switch (p.kind) {
                .rotate32, .rotate256, .split16 => 5184,
                .split32 => 576,
            } }, .{}, .{} },
        };
    }

    fn split(p: *const Turn) bool {
        return p.kind == .split32 or p.kind == .split16;
    }

    fn d(p: *const Turn) usize {
        return switch (p.kind) {
            .rotate32 => 32,
            .rotate256 => 256,
            .split32, .split16 => split_width,
        };
    }

    fn positions(p: *const Turn) usize {
        return if (p.split()) p.shape.k else memory_positions;
    }

    fn rotated(p: *const Turn) usize {
        return if (p.split()) p.shape.rows else p.shape.rows - @min(p.shape.k, p.shape.rows / 2);
    }

    pub fn reset(p: *Turn) void {
        p.a.free(p.x);
        p.a.free(p.table);
        p.x = &.{};
        p.table = &.{};
        p.rig.reset();
    }

    pub fn prepare(p: *Turn, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.n % p.d() != 0 or (p.split() and (s.k == 0 or s.n % 256 != 0))) return false;
        p.reset();
        p.shape = s;
        const width = if (p.split()) 3 * s.n else s.n;
        p.x = try p.a.alloc(f32, s.rows * width);
        p.table = try p.a.alloc(f32, p.positions() * p.d());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        for (p.x) |*v| v.* = if (stress) 1 else rng.float(f32) * 2 - 1;
        // Stress: one lane of each pair sums 16385 - 16384 = 1, the other 32769; fp16 and bf16 terms lose the 1.
        for (p.table, 0..) |*v, i| v.* = if (stress) (if (i % 2 == 0) 16385 else 16384) else (0.25 + 0.75 * rng.float(f32)) * @as(f32, if (rng.boolean()) 1 else -1);
        if (fill == .nonfinite) {
            p.x[0] = std.math.nan(f32);
            p.x[width + 1] = std.math.inf(f32);
            p.x[2 * width + 2] = -std.math.inf(f32);
            p.x[3 * width + 4] = std.math.inf(f32);
            p.x[3 * width + 5] = -std.math.inf(f32);
            if (p.split()) {
                p.x[4 * width + s.n + 6] = std.math.nan(f32);
                p.x[4 * width + 2 * s.n + 7] = -std.math.inf(f32);
            }
            p.table[2] = std.math.inf(f32);
        }
        p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.ins[1] = try p.rig.input(std.mem.sliceAsBytes(p.table));
        const parts: usize = if (p.split()) 3 else 1;
        const elem: usize = if (p.kind == .split16) 2 else 4;
        for (p.outs[0..parts]) |*o| o.* = try p.rig.output(s.rows * s.n * elem);
        return true;
    }

    pub fn run(p: *Turn) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.ins[0].buf, p.ins[0].off, 0);
        pass.e.setBuffer(p.ins[1].buf, p.ins[1].off, 1);
        if (p.split()) {
            for (p.outs, 2..) |o, i| pass.e.setBuffer(o.buf, o.off, i);
            pass.e.setValue(RotarySplit{ .rows = @intCast(s.rows), .width = @intCast(s.n), .head_width = split_width, .positions = @intCast(s.k) }, 5);
        } else {
            pass.e.setBuffer(p.outs[0].buf, p.outs[0].off, 2);
            pass.e.setValue(tbl.Rotary{ .rows = @intCast(s.rows), .heads = @intCast(s.n / p.d()), .head_width = @intCast(p.d()), .rotated_rows = @intCast(p.rotated()), .positions = memory_positions, .side = 0, .scale = 1 }, 3);
        }
        pass.e.dispatchThreads(mtl.Size.of(s.rows * s.n / 4, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Turn) usize {
        return (if (p.split()) @as(usize, 3) else 1) * p.shape.rows * p.shape.n;
    }

    /// Output i as float64 terms: a rotated lane's x c and x' (sign s), or a copied value alone (copy true).
    fn terms(p: *const Turn, i: usize) struct { t: [2]f64, copy: bool } {
        const s = p.shape;
        const plane = s.rows * s.n;
        const third = i / plane;
        const j = i % plane;
        const row = j / s.n;
        const col = j % s.n;
        const at = if (p.split()) row * 3 * s.n + col + third * s.n else j;
        if (third == 2 or row >= p.rotated()) return .{ .t = .{ p.x[at], 0 }, .copy = true };
        const lane = col % p.d();
        const pair = ((row % p.positions()) * (p.d() / 2) + lane / 2) * 2;
        const sine: f64 = p.table[pair + 1];
        const turned = if (lane % 2 == 1) sine else -sine;
        return .{ .t = .{ @as(f64, p.x[at]) * p.table[pair], @as(f64, p.x[at ^ 1]) * turned }, .copy = false };
    }

    pub fn reference(p: *Turn, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const t = p.terms(i).t;
            w.* = t[0] + t[1];
            m.* = @abs(t[0]) + @abs(t[1]);
        }
    }

    /// Copied values exactly (rounded once to f16 by the f16 split); rotated lanes are bounded by the reference.
    pub fn intervals(p: *Turn, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const t = p.terms(i);
            if (t.copy) {
                l.* = if (p.kind == .split16) @as(f16, @floatCast(t.t[0])) else t.t[0];
                h.* = l.*;
            } else {
                l.* = -std.math.floatMax(f64);
                h.* = std.math.floatMax(f64);
            }
        }
    }

    pub fn simulate(p: *Turn, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| {
            const t = p.terms(i);
            v.* = contract.rows.sumIn(acc, order, if (t.copy) t.t[0..1] else &t.t);
        }
    }

    fn deinit(p: *Turn) void {
        p.reset();
        p.rig.deinit();
        p.pipe.deinit();
    }
};

const Entry = struct { kind: ?Kind, control: bool };
const entries = [_]Entry{
    .{ .kind = null, .control = false },      .{ .kind = null, .control = true },
    .{ .kind = .rotate32, .control = false }, .{ .kind = .rotate256, .control = false },
    .{ .kind = .rotate32, .control = true },  .{ .kind = .split32, .control = false },
    .{ .kind = .split32, .control = true },   .{ .kind = .split16, .control = false },
    .{ .kind = .split16, .control = true },
};

fn function(kind: ?Kind) []const u8 {
    return if (kind) |k| switch (k) {
        .rotate32, .rotate256 => "sam3_rotate",
        .split32 => "sam3_rotate_split_f32",
        .split16 => "sam3_rotate_split_f16",
    } else "sam3_rotary_table";
}

pub const Suite = struct {
    a: std.mem.Allocator,
    tables: [2]?*tbl.Table = @splat(null),
    turns: [entries.len]?*Turn = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (entries, 0..) |e, i| {
            const name = if (e.kind) |k| Turn.label(k, e.control) else tbl.Table.label(e.control);
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
            const source = try text(s.a, if (e.control) function(e.kind) else null);
            defer s.a.free(source);
            const pipe = try rig_.pipeline(device, source, function(e.kind));
            errdefer pipe.deinit();
            try kernels.ensureUnusedCapacity(s.a, 1);
            const control: ?g.Gate = if (e.control) .shapes else null;
            if (e.kind) |k| {
                const p = try s.a.create(Turn);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .kind = k, .control = e.control };
                s.turns[i] = p;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = control });
            } else {
                const p = try s.a.create(tbl.Table);
                p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .control = e.control };
                s.tables[i] = p;
                kernels.appendAssumeCapacity(.{ .spec = p.spec(), .ctx = p, .vt = &tbl.Table.vt, .control = control });
            }
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.turns) |maybe| if (maybe) |p| {
            p.deinit();
            s.a.destroy(p);
        };
        for (s.tables) |maybe| if (maybe) |p| {
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
