//! tf-qwen27-learn's reads of a taught folder: the pocket's gate taps (Q27_GATE) and replies scored by serial decode.
const qwen27_learn_mod = @import("qwen27_learn.zig");
const std = @import("std");
const qwen = @import("qwen27");
const D = 5120;
const bf = qwen27_learn_mod.bf;

/// Q27_GATE=LIST (flashnext-gate's list and output): for each line "tag<TAB>rows<TAB>ids" the prompt runs through the.
pub fn gates(gpa: std.mem.Allocator, io: std.Io, e: *qwen.Engine, path: []const u8) !u8 {
    const L = e.w.g.layers;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var gate_rows = try gpa.alloc([]f32, L);
    defer gpa.free(gate_rows);
    // Q27_GATE_ACT: also each neuron's up row times x, its activation silu(g x) (w x).
    const acts = std.c.getenv("Q27_GATE_ACT") != null;
    var up_rows = try gpa.alloc([]f32, L);
    defer gpa.free(up_rows);
    var down_norm = try gpa.alloc([]f64, L);
    defer gpa.free(down_norm);
    var used = try gpa.alloc(usize, L);
    defer gpa.free(used);
    var layers: std.ArrayList(usize) = .empty;
    defer layers.deinit(gpa);
    var total: usize = 0;
    for (0..L) |l| {
        const cap = if (e.pockets.layers[l]) |r| r.cap else 0;
        gate_rows[l] = try gpa.alloc(f32, cap * D);
        up_rows[l] = try gpa.alloc(f32, cap * D);
        down_norm[l] = try gpa.alloc(f64, cap);
        used[l] = 0;
        if (cap == 0) continue;
        const dn = try gpa.alloc(f32, cap * D);
        defer gpa.free(dn);
        used[l] = e.pockets.read(e.s, l, gate_rows[l], up_rows[l], dn);
        for (0..cap) |i| {
            var sq: f64 = 0;
            for (dn[i * D ..][0..D]) |v| sq += @as(f64, v) * v;
            down_norm[l][i] = @sqrt(sq);
        }
        total += used[l];
        if (used[l] > 0) try layers.append(gpa, l);
    }
    defer for (gate_rows, up_rows, down_norm) |g, u, dn| {
        gpa.free(g);
        gpa.free(u);
        gpa.free(dn);
    };
    std.debug.print("pocket: {d} neurons in use at {d} layers\n", .{ total, layers.items.len });
    var tp: ?qwen.tap.Tap = null;
    defer if (tp) |*x| x.deinit(gpa);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var out = std.Io.File.stdout().writer(io, &.{});
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const tag = cols.next() orelse continue;
        var rows: std.ArrayList(u32) = .empty;
        defer rows.deinit(gpa);
        var rt = std.mem.tokenizeAny(u8, cols.next() orelse continue, ", ");
        while (rt.next()) |part| try rows.append(gpa, try std.fmt.parseInt(u32, part, 10));
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, cols.next() orelse continue, ", ");
        while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
        const n = ids.items.len;
        if (tp) |*x| if (x.seq.capacity < n + 1) {
            x.deinit(gpa);
            tp = null;
        };
        if (tp == null) tp = try qwen.tap.Tap.init(gpa, e, @max(4096, std.mem.alignForward(usize, n + 1, 1024)));
        const got = try gpa.alloc(u16, rows.items.len * layers.items.len * D);
        defer gpa.free(got);
        try tp.?.read(ids.items, 0, rows.items, layers.items, got);
        for (rows.items, 0..) |r, ri| for (layers.items, 0..) |l, j| {
            const x = got[(ri * layers.items.len + j) * D ..][0..D];
            for (0..used[l]) |i| {
                var ux: f64 = 0;
                for (gate_rows[l][i * D ..][0..D], x) |g, b| ux += @as(f64, g) * bf(b);
                if (!acts) {
                    try out.interface.print("{s} {d} {d} {d} {d:.4}\n", .{ tag, r, l, i, ux });
                    continue;
                }
                var wx: f64 = 0;
                for (up_rows[l][i * D ..][0..D], x) |w, b| wx += @as(f64, w) * bf(b);
                const act = ux / (1 + @exp(-ux)) * wx;
                try out.interface.print("{s} {d} {d} {d} {d:.4} {d:.5} {e:.4} {e:.4}\n", .{ tag, r, l, i, ux, wx, act, @abs(act) * down_norm[l][i] });
            }
        };
    }
    return 0;
}

/// Q27_SCORE=LIST: for each line "tag<TAB>prompt_len<TAB>ids" the prompt is prefilled and the rest teacher-forced.
pub fn scores(gpa: std.mem.Allocator, io: std.Io, e: *qwen.Engine, path: []const u8) !u8 {
    const V = qwen.config.vocab;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const row = try gpa.alloc(u16, V);
    defer gpa.free(row);
    var out = std.Io.File.stdout().writer(io, &.{});
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var cols = std.mem.splitScalar(u8, line, '\t');
        const tag = cols.next() orelse continue;
        const plen = try std.fmt.parseInt(usize, std.mem.trim(u8, cols.next() orelse continue, " "), 10);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, cols.next() orelse continue, ", ");
        while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
        if (plen == 0 or plen >= ids.items.len) continue;
        var seq = try e.sequence(@max(4096, std.mem.alignForward(usize, ids.items.len + 16, 1024)));
        defer seq.deinit();
        try e.prefill(&seq, ids.items[0..plen]);
        var f = e.forward();
        try f.head(&seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        var j: usize = 0;
        while (plen + j < ids.items.len) : (j += 1) {
            try e.ops().download(std.mem.sliceAsBytes(row), e.scratch.logits);
            try e.s.synchronize();
            const want = ids.items[plen + j];
            var most: f64 = -std.math.inf(f64);
            var top: usize = 0;
            for (row, 0..) |x, k| if (bf(x) > most) {
                most = bf(x);
                top = k;
            };
            var sum: f64 = 0;
            var above: usize = 0;
            const mine = bf(row[want]);
            for (row) |x| {
                sum += @exp(bf(x) - most);
                above += @intFromBool(bf(x) > mine);
            }
            const lse = most + @log(sum);
            try out.interface.print("{s} {d} {d} {d:.4} {d} {d} {d:.4}\n", .{ tag, j, want, mine - lse, above + 1, top, most - lse });
            if (plen + j + 1 >= ids.items.len) break;
            const part = [_]qwen.Part{.{ .seq = &seq, .tokens = &.{want} }};
            try f.round(&part);
            try f.picks(1, &pick);
            try f.commit(&part, &.{&.{0}});
        }
    }
    return 0;
}
