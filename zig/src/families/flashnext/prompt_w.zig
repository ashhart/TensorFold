//! Prompt products at any MLX width: qmmw_nax compiled per (bits, group), and which weights take which width.
const std = @import("std");
const mtl = @import("metal");
pub const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Prompt = replay.Prompt;
const Model = replay.Model;
const variants = @import("variants.zig");

/// One width's kernels: dense, split-K, sorted-expert gathers (32, 64, 128 rows) and fused gate/up (32, 64 rows).
pub const Pipes = struct { qmm: mtl.Pipeline, splitk: mtl.Pipeline, gather: [3]mtl.Pipeline, gu: [2]mtl.Pipeline };

/// A fused member: its rows and width, in member order.
pub const Part = struct { rows: usize, bits: u8, group: u16 };

/// A weight at a width other than 6/32: one width, or members of their own widths.
pub const Weight = struct { bits: u8 = 0, group: u16 = 0, parts: []const Part = &.{} };

/// A weight's device buffer and offset.
const Key = struct { id: usize, off: usize };

/// Every prompt weight by buffer and offset (6/32 too: a missing one is refused, never read at 6 bits) and kernels.
pub const Wide = struct {
    weights: std.AutoHashMapUnmanaged(Key, Weight) = .empty,
    pipes: std.AutoHashMapUnmanaged(u32, Pipes) = .empty,
    missed: bool = false, // a launch met a weight or width the load didn't register: the prompt pass fails

    /// The weight's width when not 6/32 (null: the recorded 6/32 launch); an unregistered weight marks `missed`.
    pub fn wide(w: *Wide, b: Buf) ?Weight {
        const got = w.weights.get(key(b)) orelse {
            w.missed = true;
            return null;
        };
        return if (got.parts.len == 0 and got.bits == 6 and got.group == 32) null else got;
    }

    /// The kernels at a width compiled at load; one never compiled marks `missed` (any width's kernels run meanwhile).
    pub fn of(w: *Wide, bits: u8, group: u16) Pipes {
        return w.pipes.get(@as(u32, bits) << 16 | group) orelse {
            w.missed = true;
            var any = w.pipes.valueIterator();
            return any.next().?.*;
        };
    }

    /// The pass's refusal when a launch met a weight or width the load didn't register.
    pub fn check(w: *const Wide) !void {
        if (w.missed) return error.PromptWeightWidthUnknown;
    }
};

fn key(b: Buf) Key {
    return .{ .id = @intFromPtr(b.b.id), .off = b.off };
}

/// qmmw_nax at `bits`/`group` with nax.h inlined for this GPU.
pub fn compile(r: *Run, bits: u8, group: u16) !Pipes {
    const head = try std.fmt.allocPrint(r.arena, "#define WB {d}\n#define WG {d}\n", .{ bits, group });
    const body = if (r.window.ssd_experts != null) try @import("fsr_prompt_source.zig").patch(r.arena, ks.flashnext_qmmw, true) else ks.flashnext_qmmw;
    const text = try frags.source(r.device, r.arena, try std.mem.concat(r.arena, u8, &.{ head, body }));
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx()));
    const p = struct {
        fn of(rr: *Run, l: mtl.Library, name: [:0]const u8) !mtl.Pipeline {
            return rr.ownResource(try mtl.Pipeline.init(rr.device, l, name, false));
        }
    };
    return .{
        .qmm = try p.of(r, lib, "tf_qmmw_t_nax"),
        .splitk = try p.of(r, lib, "tf_qmmw_splitk_nax"),
        .gather = .{ try p.of(r, lib, "tf_gather_qmmw_nax_32"), try p.of(r, lib, "tf_gather_qmmw_nax_64"), try p.of(r, lib, "tf_gather_qmmw_nax_128") },
        .gu = .{ try p.of(r, lib, "tf_gather_guw_nax_32"), try p.of(r, lib, "tf_gather_guw_nax_64") },
    };
}

fn need(w: *Wide, r: *Run, bits: u8, group: u16) !void {
    const k = @as(u32, bits) << 16 | group;
    if (!w.pipes.contains(k)) try w.pipes.put(r.arena, k, try compile(r, bits, group));
}

fn put(w: *Wide, r: *Run, b: Buf, weight: Weight) !void {
    try w.weights.put(r.arena, key(b), weight);
    if (weight.parts.len == 0 and weight.bits == 6 and weight.group == 32) return;
    if (weight.parts.len == 0) return need(w, r, weight.bits, weight.group);
    for (weight.parts) |p| try need(w, r, p.bits, p.group);
}

fn width(v: [2]usize) Weight {
    return .{ .bits = @intCast(v[0]), .group = @intCast(v[1]) };
}

/// A pack_mlx projection's `.parts` table, when it has one.
fn parts(r: *Run, stem: []const u8) !?[]const Part {
    const name = try std.fmt.allocPrint(r.arena, "{s}.parts", .{stem});
    if (!r.index.contains(name)) return null;
    const t = try variants.table(r, name);
    const n = t.len / 3;
    const out = try r.arena.alloc(Part, n);
    for (out, 0..) |*p, i| p.* = .{ .rows = t[i * 3], .bits = @intCast(t[i * 3 + 1]), .group = @intCast(t[i * 3 + 2]) };
    return out;
}

const six: Weight = .{ .bits = 6, .group = 32 };

fn hc(w: *Wide, r: *Run, h: replay.Hc) !void {
    try put(w, r, h.uw, if (h.w) |hw| width(hw[1]) else six);
    if (h.w) |hw| try need(w, r, @intCast(hw[0][0]), @intCast(hw[0][1])); // the down projection's split-K
}

/// The prompt's weights at their widths: null on a 6/32 checkpoint, which keeps every recorded launch.
pub fn attach(p: *Prompt, m: *const Model) !void {
    const r = p.r;
    const w = try r.arena.create(Wide);
    w.* = .{};
    for (&m.layers, 0..) |*L, i| {
        try hc(w, r, L.ahc);
        try hc(w, r, L.mhc);
        const x = L.xw orelse variants.Experts{ .gu = .{ 6, 32, 6, 32 }, .down = .{ 6, 32, 6, 32 } };
        for ([_]usize{ 0, 3, 6, 9, 12, 15 }, [_][2]usize{ .{ x.gu[0], x.gu[1] }, .{ x.gu[0], x.gu[1] }, .{ x.gu[2], x.gu[3] }, .{ x.gu[2], x.gu[3] }, .{ x.down[0], x.down[1] }, .{ x.down[2], x.down[3] } }) |j, v|
            if (r.fsr != null and r.fsr.?.pool != null and (j == 0 or j == 3 or j == 12)) {
                if (v[0] != 6 or v[1] != 32) try need(w, r, @intCast(v[0]), @intCast(v[1]));
            } else try put(w, r, L.ex[j], width(v)); // routed gate and up (one fused pass or two), shared, then the downs
        const kind = if (L.linear) "gdn" else "att";
        for ([_][]const u8{ if (L.linear) "in" else "proj", if (L.linear) "out" else "o" }, [_]Buf{ p.proj[i][0], p.out[i][0] }) |n, b| {
            try put(w, r, b, if (try parts(r, try std.fmt.allocPrint(r.arena, "L{d}.{s}.{s}", .{ i, kind, n }))) |ps| uniformOr(ps) else six);
        }
    }
    try hc(w, r, m.mix);
    try hc(w, r, m.mtp.ahc);
    try put(w, r, p.ple_kv[0], if (try parts(r, "ple.kv")) |ps| uniformOr(ps) else six);
    if (p.mtp_w) |mw| for ([_][]const u8{ "mtp.att.proj", "mtp.fce", "mtp.fch" }, 0..) |n, j| {
        try put(w, r, mw[j][0], if (try parts(r, n)) |ps| uniformOr(ps) else six);
    };
    p.wide = if (w.pipes.count() == 0) null else w; // all 6/32: every recorded launch as before
}

/// Members of one width are one product at that width; several widths stay members.
fn uniformOr(ps: []const Part) Weight {
    for (ps[1..]) |x| if (x.bits != ps[0].bits or x.group != ps[0].group) return .{ .parts = ps };
    return .{ .bits = ps[0].bits, .group = ps[0].group };
}

/// y[rows, n] (row stride ldy, 0: n) = x[rows, k] W^T at the weight's width(s): one launch per member when several.
pub fn qmm(p: *Prompt, wd: *Wide, w: Weight, x: Buf, ws: [3]Buf, k: usize, n: usize, rows: usize, y: Buf, ldy: usize) void {
    const ld = if (ldy == 0) n else ldy;
    if (w.parts.len == 0) return launch(p, wd.of(w.bits, w.group).qmm, ws, x, k, n, rows, y, ld);
    var at = [3]usize{ ws[0].off, ws[1].off, ws[2].off };
    var col: usize = 0;
    for (w.parts) |part| {
        const m: [3]Buf = .{ .{ .b = ws[0].b, .off = at[0] }, .{ .b = ws[1].b, .off = at[1] }, .{ .b = ws[2].b, .off = at[2] } };
        launch(p, wd.of(part.bits, part.group).qmm, m, x, k, part.rows, rows, y.at(col * 2), ld);
        at[0] += part.rows * k * part.bits / 8;
        at[1] += part.rows * (k / part.group) * 2;
        at[2] += part.rows * (k / part.group) * 2;
        col += part.rows;
    }
}

fn launch(p: *Prompt, pipe: mtl.Pipeline, w: [3]Buf, x: Buf, k: usize, n: usize, rows: usize, y: Buf, ld: usize) void {
    p.bind(pipe, &.{ w[0], w[1], w[2], x });
    const prm = [4]i32{ @intCast(k), @intCast(n), @intCast(rows), @intCast(ld) };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 4);
    p.r.enc.setBuffer(y.b, y.off, 5);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, (rows + 63) / 64, 1), mtl.Size.of(128, 1, 1));
    p.barrier();
}

/// The sorted-expert gather at the weight's width: tiles of 32, 64 or 128 rows as Prompt.gather picks them.
pub fn gather(p: *Prompt, pipes: Pipes, bm: usize, x: Buf, w: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
    p.bind(pipes.gather[if (bm == 128) 2 else if (bm == 64) 1 else 0], &.{ x, w[0], w[1], w[2], p.b.off });
    const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 5);
    p.r.enc.setBuffer(y.b, y.off, 6);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
    p.barrier();
}

/// The fused gate/up gather at the weights' width.
pub fn gatherGU(p: *Prompt, pipes: Pipes, bm: usize, x: Buf, wg: []const Buf, wu: []const Buf, k: usize, n: usize, pairs: usize, y: Buf) void {
    p.bind(pipes.gu[if (bm == 64) 1 else 0], &.{ x, wg[0], wg[1], wg[2], wu[0], wu[1], wu[2], p.b.off });
    const prm = [4]i32{ @intCast(pairs), @intCast(n), @intCast(k), 512 };
    p.r.enc.setBytes(std.mem.asBytes(&prm), 8);
    p.r.enc.setBuffer(y.b, y.off, 9);
    p.r.enc.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, pairs / bm + 512, 1), mtl.Size.of(128, 1, 1));
    p.barrier();
}

test {
    _ = @import("prompt_w_test.zig");
}
