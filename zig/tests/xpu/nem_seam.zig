//! Prefill seam check: Mamba conv/SSM state, attention KV bytes and following decode logits, row by row vs in chunks.

const std = @import("std");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;

const gpa = std.heap.page_allocator;
const steps = 8;

const Kind = enum { conv, ssm, key, value };
const kinds = [_]Kind{ .conv, .ssm, .key, .value };

/// Every layer's buffer of one kind, as downloaded bytes.
const Snap = struct { bytes: [4][][]u8 };

fn snapshot(r: *rt.Runtime, m: *model.Model) !Snap {
    var out: [4]std.ArrayList([]u8) = @splat(.empty);
    try r.sync();
    for (m.layers) |ly| switch (ly.mixer) {
        .mamba => |mw| {
            for ([_]struct { k: Kind, b: @TypeOf(mw.cstate) }{ .{ .k = .conv, .b = mw.cstate }, .{ .k = .ssm, .b = mw.sstate } }) |e| {
                const bytes = try gpa.alloc(u8, e.b.len);
                try r.download(bytes, e.b);
                try out[@intFromEnum(e.k)].append(gpa, bytes);
            }
        },
        .attention => |aw| {
            for ([_]struct { k: Kind, b: @TypeOf(aw.kc) }{ .{ .k = .key, .b = aw.kc }, .{ .k = .value, .b = aw.vc } }) |e| {
                const bytes = try gpa.alloc(u8, e.b.len);
                try r.download(bytes, e.b);
                try out[@intFromEnum(e.k)].append(gpa, bytes);
            }
        },
        else => {},
    };
    try r.sync();
    var s: Snap = undefined;
    for (&s.bytes, &out) |*d, *o| d.* = try o.toOwnedSlice(gpa);
    return s;
}

fn at(kind: Kind, bytes: []const u8, i: usize) f64 {
    return switch (kind) {
        .ssm => @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little)))),
        else => @as(f64, @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16))),
    };
}

fn width(kind: Kind) usize {
    return if (kind == .ssm) 4 else 2;
}

/// Prints one comparison line per kind: elements, differing elements, worst absolute difference, reference max.
fn compare(label: []const u8, a: Snap, b: Snap, kv_dim: usize, n_pos: usize) bool {
    var same = true;
    for (kinds) |k| {
        var n: usize = 0;
        var diff: usize = 0;
        var worst: f64 = 0;
        var mag: f64 = 0;
        var worst_layer: usize = 0;
        var first_pos: usize = std.math.maxInt(usize);
        var last_pos: usize = 0;
        for (a.bytes[@intFromEnum(k)], b.bytes[@intFromEnum(k)], 0..) |x, y, li| {
            // KV bytes past the prompt hold stale data from earlier runs: only positions below the prompt count
            const cnt = if (k == .key or k == .value) @min(x.len / width(k), n_pos * kv_dim) else x.len / width(k);
            n += cnt;
            for (0..cnt) |i| {
                const p = at(k, x, i);
                const q = at(k, y, i);
                mag = @max(mag, @abs(p));
                if (std.mem.eql(u8, x[i * width(k) ..][0..width(k)], y[i * width(k) ..][0..width(k)])) continue;
                diff += 1;
                if (k == .key or k == .value) {
                    first_pos = @min(first_pos, i / kv_dim);
                    last_pos = @max(last_pos, i / kv_dim);
                }
                if (@abs(p - q) > worst) {
                    worst = @abs(p - q);
                    worst_layer = li;
                }
            }
        }
        if (diff != 0) same = false;
        std.debug.print("  {s} {s}: {d} elements, {d} differ, worst |diff| {e:.2} (layer {d} of this kind), max |ref| {e:.2}", .{ label, @tagName(k), n, diff, worst, worst_layer, mag });
        if (diff != 0 and (k == .key or k == .value)) std.debug.print(", differing positions {d}..{d}", .{ first_pos, last_pos });
        std.debug.print("\n", .{});
    }
    return same;
}

fn top1(v: []const f32) u32 {
    var b: u32 = 0;
    for (v, 0..) |x, i| if (x > v[b]) {
        b = @intCast(i);
    };
    return b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 640;
    var chunks: std.ArrayList(u32) = .empty;
    for (args[@min(args.len, 3)..]) |a| try chunks.append(gpa, try std.fmt.parseInt(u32, a, 10));
    if (chunks.items.len == 0) try chunks.appendSlice(gpa, &.{ 128, 512 });
    var big: u32 = 16;
    for (chunks.items) |c| big = @max(big, c);
    nw.default_rows = big;
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var m = try model.Model.load(gpa, &r, &l, cfg.value, @intCast(n + 64));
    m.bf16_logits = true;
    var w = try nw.Win.init(gpa, &m);
    const vocab = cfg.value.vocab_size;
    const kv_dim: usize = cfg.value.num_key_value_heads * cfg.value.head_dim;
    const ids = try gpa.alloc(u32, n + steps);
    var prng = std.Random.DefaultPrng.init(42);
    for (ids) |*x| x.* = 1000 + prng.random().uintLessThan(u32, 60000);
    // SEAM_IDS=/abs/file: comma separated token ids of real text replace the random ones
    if (std.c.getenv("SEAM_IDS")) |path| {
        var it = std.mem.tokenizeAny(u8, try ld.readFile(gpa, "/", std.mem.span(path)[1..]), ", \r\n");
        for (ids) |*x| x.* = try std.fmt.parseInt(u32, it.next() orelse return error.TooFewIds, 10);
    }

    // the reference: one row at a time
    try w.reset(&m);
    for (0..n) |i| {
        try nw.stopCheck(&r);
        try w.forward(&m, ids[i .. i + 1], true, 0);
        if (i % 64 == 63) try r.sync();
    }
    const ref = try snapshot(&r, &m);
    const ref_logits = try gpa.alloc(f32, steps * vocab);
    for (0..steps) |s| {
        try w.forward(&m, ids[n + s .. n + s + 1], true, 1);
        try w.fetchRowLogits(&m, ref_logits[s * vocab ..][0..vocab]);
    }

    var prev: ?Snap = null;
    var prev_logits: ?[]f32 = null;
    var bad = false;
    for (chunks.items) |c| {
        try w.reset(&m);
        var i: usize = 0;
        while (i < n) {
            try nw.stopCheck(&r);
            const rows = @min(c, n - i);
            try w.forward(&m, ids[i .. i + rows], true, 0);
            i += rows;
        }
        const snap = try snapshot(&r, &m);
        const logits = try gpa.alloc(f32, steps * vocab);
        for (0..steps) |s| {
            try w.forward(&m, ids[n + s .. n + s + 1], true, 1);
            try w.fetchRowLogits(&m, logits[s * vocab ..][0..vocab]);
        }
        // handoff: the same prompt in windows, then the plain single-row Model.forward for the decode steps
        try w.reset(&m);
        i = 0;
        while (i < n) {
            try nw.stopCheck(&r);
            const rows = @min(c, n - i);
            try w.forward(&m, ids[i .. i + rows], true, 0);
            i += rows;
        }
        const handoff = try gpa.alloc(f32, steps * vocab);
        for (0..steps) |s| {
            try m.forward(ids[n + s], true);
            try m.fetchLogits(handoff[s * vocab ..][0..vocab]);
        }
        const hand_same = std.mem.eql(u8, std.mem.sliceAsBytes(handoff), std.mem.sliceAsBytes(logits));
        var hand_worst: f32 = 0;
        for (handoff, logits) |x, y| hand_worst = @max(hand_worst, @abs(x - y));
        std.debug.print("chunks of {d}: windows then Model.forward decode against windows then window decode: logits {s} (max |diff| {e:.2})\n", .{ c, if (hand_same) "identical" else "differ", hand_worst });
        std.debug.print("prefill in chunks of {d} against row by row ({d} prompt tokens):\n", .{ c, n });
        _ = compare("vs rows", ref, snap, kv_dim, n);
        var worst: f32 = 0;
        var mag: f32 = 0;
        var same_top: usize = 0;
        for (0..steps) |s| {
            const a = ref_logits[s * vocab ..][0..vocab];
            const b = logits[s * vocab ..][0..vocab];
            for (a, b) |x, y| {
                worst = @max(worst, @abs(x - y));
                mag = @max(mag, @abs(x));
            }
            same_top += @intFromBool(top1(a) == top1(b));
        }
        std.debug.print("  decode after the prompt, {d} steps: logits max |diff| {e:.2} (max |logit| {e:.2}), top-1 equal in {d} of {d}\n", .{ steps, worst, mag, same_top, steps });
        if (prev) |p| {
            const state_same = compare("vs previous chunk size", p, snap, kv_dim, n);
            const logit_same = std.mem.eql(u8, std.mem.sliceAsBytes(prev_logits.?), std.mem.sliceAsBytes(logits));
            std.debug.print("  states identical to the previous chunk size: {s}; decode logits identical: {s}\n", .{ if (state_same) "yes" else "no", if (logit_same) "yes" else "no" });
            if (!state_same or !logit_same) bad = true;
        }
        prev = snap;
        prev_logits = logits;
    }
    std.debug.print("{s}\n", .{if (bad) "seam check: chunk sizes differ" else "seam check ok: chunk sizes agree bitwise"});
}
