//! The prompt tail's launches recorded on the host: a two-layer model's span through a launcher with no GPU behind it.
const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const state = @import("state.zig");
const pages = @import("pages.zig");
const fwd = @import("forward.zig");
const T = std.testing;

/// What a recorded launch did: its kernel, and the rows, first position, keys and weight it named where it has them.
const Launch = struct { name: []const u8, rows: i64 = -1, at: i64 = -1, keys: i64 = -1, weight: u64 = 0 };

/// The leading fields of a product's argument (launches/affine.zig's Arg, which the dense tile takes too).
const Product = extern struct { x: u64, words: u64, scale: [2]u64, bias: [2]u64, out: u64, m: c_int, n: c_int, k: c_int };

const Mock = struct {
    var rt: hip.Runtime = undefined;
    var log: [512]Launch = undefined;
    var count: usize = 0;

    fn int(p: [*]?*anyopaque, i: usize) i64 {
        return @as(*const c_int, @ptrCast(@alignCast(p[i].?))).*;
    }

    fn word(p: [*]?*anyopaque, i: usize) u64 {
        return @as(*const u64, @ptrCast(@alignCast(p[i].?))).*;
    }

    /// A kernel's handle is its name, so a launch says which one ran.
    fn function(out: *hip.abi.Function, _: hip.abi.Module, symbol: [*:0]const u8) callconv(.c) hip.abi.Result {
        out.* = @ptrCast(@constCast(symbol));
        return 0;
    }

    fn launch(f: hip.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: hip.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) hip.abi.Result {
        const kernel = std.mem.span(@as([*:0]const u8, @ptrCast(f.?)));
        const p = params.?;
        var l: Launch = .{ .name = kernel };
        if (std.mem.eql(u8, kernel, "rms_rows")) {
            l.weight = word(p, 2);
            l.rows = int(p, 4);
        } else if (isProduct(kernel)) {
            const a: *const Product = @ptrCast(@alignCast(p[0].?));
            l.weight = a.words;
            l.rows = a.m;
        } else if (std.mem.eql(u8, kernel, "fa_paged")) {
            l.rows = int(p, 4);
            l.keys = int(p, 5);
            l.at = int(p, 10);
        } else if (std.mem.eql(u8, kernel, "rope_prefill")) {
            l.rows = int(p, 6);
            l.at = int(p, 10);
        } else if (std.mem.eql(u8, kernel, "page_write")) {
            l.weight = word(p, 1);
            l.rows = int(p, 3);
            l.at = int(p, 8);
        }
        if (count == log.len) return 1;
        log[count] = l;
        count += 1;
        return 0;
    }
};

/// A product's kernel: the dense tile (named by its field) or an MLX tile (by its symbol, which takes an Affine).
fn isProduct(kernel: []const u8) bool {
    return std.mem.eql(u8, kernel, "fns") or std.mem.indexOf(u8, kernel, "6AffineE") != null;
}

/// Every kernel of `v` named by its field, so a launch says which one ran.
fn name(comptime V: type, v: *V, comptime field: [:0]const u8) void {
    if (V == hip.Function) {
        v.* = .{ .r = &Mock.rt, .handle = @ptrCast(@constCast(field.ptr)) };
        return;
    }
    switch (@typeInfo(V)) {
        .@"struct" => |s| if (s.layout != .@"packed") inline for (s.field_names, s.field_types, s.field_attrs) |n, F, a| {
            if (!a.@"comptime") name(F, &@field(v.*, n), n);
        },
        .array => |a| for (v) |*e| name(a.child, e, field),
        .optional => |o| if (o.child == hip.Function) {
            var f: hip.Function = undefined;
            name(hip.Function, &f, field);
            v.* = f;
        },
        else => {},
    }
}

const hidden = 256;
const heads = 2;
const hd = 64;

/// Layer `i`'s weight in slot `at`: an address the launches name it by.
fn addr(i: usize, at: u64) u64 {
    return 0x1_0000_0000 + i * 0x100_0000 + at * 0x1_0000;
}

const final_norm: u64 = 0x2_0000_0000;
const slot = struct {
    const q = 2;
    const k = 3;
    const v = 4;
    const o = 5;
    const q_norm = 6;
    const k_norm = 7;
    const gate = 8;
    const up = 9;
    const down = 10;
};

fn dense(w: u64, n: u32, k: u32) view.Projection {
    return .{ .n = n, .k = k, .handle = .{ .dense = .{ .weight = w, .dtype = .bf16 } } };
}

/// A 4-bit MLX product whose words are at `w`, as the MLX checkpoints hold them.
fn affine(w: u64, n: u32, k: u32) view.Projection {
    return .{ .n = n, .k = k, .handle = .{ .mlx = .{ .words = w, .scale = w + 0x8000, .bias = w + 0xC000, .tables = .bf16, .bits = 4, .group = 64 } } };
}

/// The format of a layer's q, o and MLP products; its k and v are dense, which the tail never weighs.
const Format = enum { dense, mlx };

fn product(format: Format, w: u64, n: u32, k: u32) view.Projection {
    return if (format == .mlx) affine(w, n, k) else dense(w, n, k);
}

fn full(i: usize, format: Format) view.Full {
    const mlp: view.Mlp = .{ .dense = .{ .gate = product(format, addr(i, slot.gate), hidden, hidden), .up = product(format, addr(i, slot.up), hidden, hidden), .down = product(format, addr(i, slot.down), hidden, hidden) } };
    return .{
        .input_norm = addr(i, 0),
        .post_norm = addr(i, 1),
        .q = product(format, addr(i, slot.q), heads * 2 * hd, hidden),
        .k = dense(addr(i, slot.k), hd, hidden),
        .v = dense(addr(i, slot.v), hd, hidden),
        .o = product(format, addr(i, slot.o), hidden, heads * hd),
        .q_norm = addr(i, slot.q_norm),
        .k_norm = addr(i, slot.k_norm),
        .mlp = mlp,
    };
}

/// The launches of one span of `len` rows from `pos0` over two full layers, read as `reads`; the last's in `last`.
fn record(len: usize, pos0: usize, reads: fwd.Reads, last: Format) !fwd.Final {
    Mock.rt.api.hipModuleLaunchKernel = Mock.launch;
    Mock.rt.api.hipModuleGetFunction = Mock.function;
    Mock.count = 0;
    var l: hip.Launcher = undefined;
    name(hip.Launcher, &l, "launcher");
    l.r = &Mock.rt;
    // the MLX tiles and their registry as gfx1151 loads them, each kernel named by its symbol
    const module: hip.Module = .{ .r = &Mock.rt, .handle = @ptrFromInt(0x20) };
    l.affine = try @FieldType(hip.Launcher, "affine").load(module, module, hip.Caps.of("gfx1151").?, .{});
    l.wide = true;
    l.edge = false;
    l.fuse = false;
    l.dense.bf16 = true;
    l.dense.wmma12 = false;
    l.dense.serve = .dot2;
    var spec = std.mem.zeroes(view.Spec);
    spec.hidden = hidden;
    spec.intermediate = hidden;
    spec.n_layers = 2;
    spec.heads = heads;
    spec.kv_heads = 1;
    spec.head_dim = hd;
    spec.rotary_dim = hd;
    spec.vocab = 512;
    spec.eps = 1e-6;
    spec.rope_theta = 1e6;
    const layers = [_]view.Layer{ .{ .full = full(0, .dense) }, .{ .full = full(1, last) } };
    const m: view.Model = .{ .spec = spec, .act = .bf16, .embed = dense(0x3_0000_0000, 512, hidden), .layers = &layers, .final_norm = final_norm, .head = dense(0x3_1000_0000, 512, hidden), .qk = .{ .q_weight = 0, .k_weight = 0, .eps = 1e-6 } };
    var keys = [_]hip.DeviceBuffer{ .{ .r = &Mock.rt, .ptr = @ptrFromInt(0x4_0000_0000), .len = 1 }, .{ .r = &Mock.rt, .ptr = @ptrFromInt(0x4_1000_0000), .len = 1 } };
    var values = [_]hip.DeviceBuffer{ .{ .r = &Mock.rt, .ptr = @ptrFromInt(0x5_0000_0000), .len = 1 }, .{ .r = &Mock.rt, .ptr = @ptrFromInt(0x5_1000_0000), .len = 1 } };
    var pool: pages.Pool = undefined;
    pool.keys = &keys;
    pool.values = &values;
    pool.count = 64;
    var cache_layers = [_]state.LayerCache{ .full, .full };
    var caches: state.Caches = .{ .layers = &cache_layers, .total = 4096, .desc = .{ .r = &Mock.rt, .ptr = @ptrFromInt(0x6_0000_0000), .len = 1 }, .pool = &pool };
    var arena = hip.Arena.counting();
    const o: hip.ops.Ops = .{ .l = &l, .bf16 = true, .stream = @ptrFromInt(0x10), .arena = &arena };
    return fwd.spanFor(o, &m, &caches, 0x7_0000_0000, len, pos0, null, reads);
}

/// The recorded launches of kernel `kernel` ("product": any product's) and weight `weight` where nonzero, in order.
fn launches(buf: []Launch, kernel: []const u8, weight: u64) []Launch {
    var n: usize = 0;
    for (Mock.log[0..Mock.count]) |l| {
        const named = if (std.mem.eql(u8, kernel, "product")) isProduct(l.name) else std.mem.eql(u8, l.name, kernel);
        if (!named or (weight != 0 and l.weight != weight)) continue;
        buf[n] = l;
        n += 1;
    }
    return buf[0..n];
}

/// The rows each product of layer `i` in `slots` ran, one launch each.
fn productRows(i: usize, slots: []const u64, rows: i64) !void {
    var buf: [8]Launch = undefined;
    for (slots) |s| {
        const got = launches(&buf, "product", addr(i, s));
        try T.expectEqual(@as(usize, 1), got.len);
        try T.expectEqual(rows, got[0].rows);
    }
}

const query_slots = [_]u64{ slot.q, slot.o, slot.gate, slot.up, slot.down };
const all_slots = [_]u64{ slot.k, slot.v } ++ query_slots;

/// One attention launch's query rows, first query position and keys.
fn expectAttention(rows: i64, at: i64, keys: i64, l: Launch) !void {
    try T.expectEqual(rows, l.rows);
    try T.expectEqual(at, l.at);
    try T.expectEqual(keys, l.keys);
}

test "qwen3_5 hip prompt tail: a chunk nothing reads runs its last layer's keys and values and stops" {
    const kept = try record(300, 64, .{ .logits = false, .readers = false }, .mlx);
    try T.expectEqual(@as(usize, 300), kept.from);
    var buf: [16]Launch = undefined;
    // layer 0 whole; layer 1 k and v over every row into its pages, then nothing
    try productRows(0, &all_slots, 300);
    try productRows(1, &.{ slot.k, slot.v }, 300);
    for (query_slots) |s| try T.expectEqual(@as(usize, 0), launches(&buf, "product", addr(1, s)).len);
    try T.expectEqual(@as(usize, 1), launches(&buf, "fa_paged", 0).len);
    try T.expectEqual(@as(usize, 1), launches(&buf, "attn_gate", 0).len);
    const k1 = launches(&buf, "page_write", 0x4_1000_0000);
    try T.expectEqual(@as(usize, 1), k1.len);
    try T.expectEqual(@as(i64, 300), k1[0].rows);
    try T.expectEqual(@as(i64, 64), k1[0].at);
    try T.expectEqual(@as(usize, 1), launches(&buf, "page_write", 0x5_1000_0000).len);
    // layer 0's q and k, then layer 1's k alone
    try T.expectEqual(@as(usize, 3), launches(&buf, "rope_prefill", 0).len);
    try T.expectEqual(@as(usize, 0), launches(&buf, "rms_rows", final_norm).len);
    try T.expectEqual(@as(usize, 0), launches(&buf, "rms_rows", addr(1, 1)).len);
}

test "qwen3_5 hip prompt tail: a chunk that keeps its last row runs the last layer's queries over its last 128 rows" {
    const kept = try record(300, 64, .{ .logits = true, .readers = false }, .mlx);
    try T.expectEqual(@as(usize, 300 - 128), kept.from);
    try T.expectEqual(kept.rows.ptr + 127 * hidden * 2, kept.row(299, hidden).ptr);
    var buf: [16]Launch = undefined;
    try productRows(0, &all_slots, 300);
    try productRows(1, &.{ slot.k, slot.v }, 300);
    try productRows(1, &query_slots, 128);
    const att = launches(&buf, "fa_paged", 0);
    try T.expectEqual(@as(usize, 2), att.len);
    try expectAttention(300, 64, 364, att[0]);
    try expectAttention(128, 64 + 172, 364, att[1]);
    const ropes = launches(&buf, "rope_prefill", 0);
    try T.expectEqual(@as(usize, 4), ropes.len);
    try T.expectEqual(@as(i64, 300), ropes[2].rows);
    try T.expectEqual(@as(i64, 64), ropes[2].at);
    try T.expectEqual(@as(i64, 128), ropes[3].rows);
    try T.expectEqual(@as(i64, 64 + 172), ropes[3].at);
    try T.expectEqual(@as(i64, 128 * heads), launches(&buf, "rms_rows", addr(1, slot.q_norm))[0].rows);
    try T.expectEqual(@as(i64, 128), launches(&buf, "rms_rows", addr(1, 1))[0].rows);
    const last = launches(&buf, "rms_rows", final_norm);
    try T.expectEqual(@as(usize, 1), last.len);
    try T.expectEqual(@as(i64, 128), last[0].rows);
}

test "qwen3_5 hip prompt tail: a readout's chunk runs every row, and a tail across two steps runs each step's part" {
    var buf: [16]Launch = undefined;
    const whole = try record(300, 0, .{ .logits = true, .readers = true }, .mlx);
    try T.expectEqual(@as(usize, 0), whole.from);
    try productRows(1, &all_slots, 300);
    try T.expectEqual(@as(i64, 300), launches(&buf, "rms_rows", final_norm)[0].rows);
    // 2,100 rows: the tail from 1,972 ends step one's 2,048 rows and is all of step two's 52
    const two = try record(2100, 0, .{ .logits = true, .readers = false }, .mlx);
    try T.expectEqual(@as(usize, 2100 - 128), two.from);
    const att = launches(&buf, "fa_paged", 0);
    try T.expectEqual(@as(usize, 4), att.len);
    try expectAttention(76, 1972, 2048, att[2]);
    try expectAttention(52, 2048, 2100, att[3]);
    try T.expectEqual(@as(i64, 128), launches(&buf, "rms_rows", final_norm)[0].rows);
}

test "qwen3_5 hip prompt tail: advance asks for keys only, prefill for the last row's tile, a readout for every row" {
    const passes = @import("../engine/engine.zig").Engine.passes;
    const open: @import("prompt_tail.zig").Facts = .{};
    try T.expectEqual(@as(usize, 4096), open.first(passes.advance, 4096));
    try T.expectEqual(@as(usize, 4096 - 128), open.first(passes.prefill, 4096));
    try T.expectEqual(@as(usize, 0), open.first(passes.readout, 4096));
}

test "qwen3_5 hip prompt tail: a last layer with dense products keeps its keys-only pass but runs every row when read" {
    var buf: [16]Launch = undefined;
    const read = try record(300, 64, .{ .logits = true, .readers = false }, .dense);
    try T.expectEqual(@as(usize, 0), read.from);
    try productRows(1, &all_slots, 300);
    try T.expectEqual(@as(i64, 300), launches(&buf, "rms_rows", final_norm)[0].rows);
    const unread = try record(300, 64, .{ .logits = false, .readers = false }, .dense);
    try T.expectEqual(@as(usize, 300), unread.from);
    try productRows(1, &.{ slot.k, slot.v }, 300);
    for (query_slots) |s| try T.expectEqual(@as(usize, 0), launches(&buf, "product", addr(1, s)).len);
}

test "qwen3_5 hip prompt tail: across a step edge, the read row's step runs main's rows and tiles" {
    // 2,175 rows: the tail is the first step's last row and all 127 of the second
    var whole: [5][2]Launch = undefined;
    _ = try record(2175, 0, .{ .logits = true, .readers = true }, .mlx);
    for (query_slots, &whole) |s, *w| {
        var buf: [4]Launch = undefined;
        const got = launches(&buf, "product", addr(1, s));
        try T.expectEqual(@as(usize, 2), got.len);
        w.* = got[0..2].*;
    }
    _ = try record(2175, 0, .{ .logits = true, .readers = false }, .mlx);
    for (query_slots, whole) |s, w| {
        var buf: [4]Launch = undefined;
        const got = launches(&buf, "product", addr(1, s));
        try T.expectEqual(@as(usize, 2), got.len);
        try T.expectEqual(@as(i64, 127), got[1].rows);
        try T.expectEqual(w[1].rows, got[1].rows);
        try T.expectEqualStrings(w[1].name, got[1].name);
        // the first step's piece is one row: a few-row tile where main ran 2,048 rows, for a row no reader reaches
        try T.expectEqual(@as(i64, 1), got[0].rows);
        try T.expectEqual(@as(i64, 2048), w[0].rows);
        try T.expect(!std.mem.eql(u8, w[0].name, got[0].name));
    }
}
