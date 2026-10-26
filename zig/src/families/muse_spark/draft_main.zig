//! tf-muse dflash / drafted: the DFlash2 drafter's gates (docs/recipes/muse-spark.md), one GPU, a pair or TP=2.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const muse = @import("muse_spark");
const Model = muse.forward.Model;
const Row = muse.forward.Row;
const Drafter = muse.dflash2.Drafter;

pub const Mode = enum { dflash, drafted };
pub const Knobs = struct { context: usize, n_greedy: usize, graphs: bool, max_layers: ?usize, rows_per: usize, dump_taps: bool };

const Text = struct { name: []const u8, ids: []i32, out: std.ArrayList(i32) = .empty, plain: std.ArrayList(i32) = .empty };

fn readAll(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 32));
}
fn writeAll(io: std.Io, path: []const u8, bytes: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

const Ctx = struct {
    gpa: std.mem.Allocator,
    s: cuda.Stream,
    model: *Model,
    dr: *Drafter,
    graphs: bool,
    am: [muse.forward.DECODE_ROWS]i32 = undefined,

    fn fwd(c: *Ctx, rows: []const Row) !void {
        if (c.graphs) try c.model.graphed(c.s, rows, null) else try c.model.forward(c.s, rows, null, null);
        try c.model.argmaxTo(c.s, rows.len, c.am[0..rows.len]);
    }
};

/// One rank's run of `mode`: load the target and the drafter, every text, rank 0 writes OUT_DIR.
pub fn run(mode: Mode, gpa: std.mem.Allocator, ar: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, args: []const [:0]const u8, rank: usize, world: usize, comm: ?tp.Collective, kn: Knobs) !void {
    var s = try cuda.Stream.init(driver, false);
    defer s.deinit();
    var w = try muse.load.load(gpa, io, driver, args[2], rank, world, .{ .max_layers = kn.max_layers, .log = rank == 0 });
    defer w.deinit();
    var k3 = try muse.exl3.Kernels.load(driver);
    defer k3.deinit();
    var texts: std.ArrayList(Text) = .empty;
    const names_text = try readAll(ar, io, try std.fs.path.join(ar, &.{ args[3], "names.txt" }));
    var it = std.mem.tokenizeAny(u8, names_text, " \r\n\t");
    while (it.next()) |name| {
        const raw = try readAll(ar, io, try std.fmt.allocPrint(ar, "{s}/{s}.ids", .{ args[3], name }));
        const ids = try ar.alloc(i32, raw.len / 4);
        @memcpy(std.mem.sliceAsBytes(ids), raw[0 .. ids.len * 4]);
        if (ids.len + kn.n_greedy + 16 > kn.context) return error.TextTooLong;
        try texts.append(ar, .{ .name = name, .ids = ids });
    }
    const slots = if (mode == .drafted) @min(texts.items.len, muse.dflash2.MAX_STREAMS) else 1;
    if (mode == .drafted and texts.items.len > slots) return error.TooManyTexts;
    var model = try Model.init(gpa, driver, &w, &k3, comm, kn.context, muse.forward.DECODE_ROWS, slots, s);
    defer model.deinit();
    const ddir = try muse.engine.drafterDir(ar, args[2], if (std.c.getenv("TF_MUSE_DRAFTER")) |v| std.mem.span(v) else null);
    var tb: [8]usize = undefined;
    const dr = try Drafter.init(gpa, io, &model, ddir, s, kn.graphs, try tapsOverride(&tb));
    defer dr.deinit();
    model.dr = dr;
    std.debug.print("[draft] rank {d}/{d}: target {d} layers ({d:.1} GiB), drafter {s} ({d:.2} GiB, taps {any}), {d} slots, {s}\n", .{ rank, world, w.layers.len, @as(f64, @floatFromInt(w.arena.total)) / (1 << 30), ddir, @as(f64, @floatFromInt(dr.bytes())) / (1 << 30), dr.taps, slots, if (kn.graphs) "graphed" else "eager" });
    var c: Ctx = .{ .gpa = gpa, .s = s, .model = &model, .dr = dr, .graphs = kn.graphs };
    switch (mode) {
        .dflash => for (texts.items) |*t| try fixedProtocol(&c, ar, io, args[4], t, w.full, kn, rank == 0),
        .drafted => try drafted(&c, ar, io, args[4], texts.items, w.full, kn, rank == 0),
    }
    std.debug.print("DONE {s} rank {d}\n", .{ @tagName(mode), rank });
}

/// The fixed protocol for one text (slot 0): the prompt a row at a time, then greedy steps that draft first.
fn fixedProtocol(c: *Ctx, ar: std.mem.Allocator, io: std.Io, out_dir: []const u8, t: *Text, cfg: muse.Config, kn: Knobs, write: bool) !void {
    const dr = c.dr;
    const P = dr.block - 1;
    const NT = dr.taps.len;
    const D = cfg.hidden;
    const T = t.ids.len;
    var taps: std.ArrayList(u16) = .empty;
    var drafts: std.ArrayList(i32) = .empty;
    var seq: std.ArrayList(i32) = .empty;
    try seq.appendSlice(ar, t.ids);
    const row_buf = try ar.alloc(u16, NT * D);
    var pending: i32 = 0;
    for (0..T + kn.n_greedy) |j| {
        if (j >= T) { // a greedy step: draft from the pending token at position j, then feed it
            var out: [16]i32 = undefined;
            try dr.draft(c.s, &.{pending}, &.{@intCast(j)}, &.{0}, &out);
            try drafts.appendSlice(ar, out[0..P]);
            try seq.append(ar, pending);
        }
        const tok: i32 = if (j < T) t.ids[j] else pending;
        try c.fwd(&.{.{ .slot = 0, .pos = @intCast(j), .token = tok }});
        if (kn.dump_taps) {
            try c.s.synchronize();
            try c.model.d.check(c.model.d.api.cuMemcpyDtoH_v2(@ptrCast(row_buf.ptr), dr.b.tapbuf, NT * D * 2), "taps");
            try taps.appendSlice(ar, row_buf);
        }
        pending = c.am[0];
        if (j >= T and cfg.isEos(@intCast(tok))) break;
    }
    const G = seq.items.len - T;
    var acc_sum: usize = 0;
    var hist: [17]usize = @splat(0);
    for (0..G) |i| {
        var a: usize = 0;
        while (a < P and T + i + 1 + a < seq.items.len and drafts.items[i * P + a] == seq.items[T + i + 1 + a]) a += 1;
        acc_sum += a;
        hist[a] += 1;
    }
    std.debug.print("[dflash] {s}: prompt {d}, greedy {d}, mean accepted {d:.3} of {d} (counts by accepted {any})\n", .{ t.name, T, G, @as(f64, @floatFromInt(acc_sum)) / @as(f64, @floatFromInt(@max(G, 1))), P, hist[0 .. P + 1] });
    if (!write) return;
    try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.seq.i32", .{ out_dir, t.name }), std.mem.sliceAsBytes(seq.items));
    try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.drafts.i32", .{ out_dir, t.name }), std.mem.sliceAsBytes(drafts.items));
    if (kn.dump_taps) try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.taps.f16", .{ out_dir, t.name }), std.mem.sliceAsBytes(taps.items));
}

/// Prompts of every slot in windows of up to rows_per rows a slot; each slot's first greedy token.
fn prefill(c: *Ctx, texts: []Text, rows_per: usize, first: []i32) !void {
    var fed: [muse.dflash2.MAX_STREAMS]usize = @splat(0);
    var rows: [muse.forward.DECODE_ROWS]Row = undefined;
    var last: [muse.dflash2.MAX_STREAMS]usize = undefined;
    const per = @min(rows_per, muse.forward.DECODE_ROWS / texts.len);
    while (true) {
        var n: usize = 0;
        var who: [muse.dflash2.MAX_STREAMS]bool = @splat(false);
        for (texts, 0..) |t, j| if (fed[j] < t.ids.len) {
            const k = @min(per, t.ids.len - fed[j]);
            for (0..k) |q| rows[n + q] = .{ .slot = @intCast(j), .pos = @intCast(fed[j] + q), .token = t.ids[fed[j] + q] };
            n += k;
            fed[j] += k;
            last[j] = n - 1;
            who[j] = fed[j] == t.ids.len;
        };
        if (n == 0) break;
        try c.fwd(rows[0..n]);
        for (0..texts.len) |j| if (who[j]) {
            first[j] = c.am[last[j]];
        };
    }
}

/// Plain greedy, then the speculative loop on the same slots; every reply must be byte-identical.
fn drafted(c: *Ctx, ar: std.mem.Allocator, io: std.Io, out_dir: []const u8, texts: []Text, cfg: muse.Config, kn: Knobs, write: bool) !void {
    const S = texts.len;
    const B = c.dr.block;
    const MS = muse.dflash2.MAX_STREAMS;
    var first: [MS]i32 = undefined;
    var rows: [muse.forward.DECODE_ROWS]Row = undefined;
    // plain: one row a slot a window
    try prefill(c, texts, kn.rows_per, &first);
    var pend: [MS]i32 = undefined;
    var pos: [MS]usize = undefined;
    var live: [MS]bool = @splat(true);
    for (texts, 0..) |*t, j| {
        pend[j] = first[j];
        pos[j] = t.ids.len;
        try t.plain.append(ar, first[j]);
        live[j] = !cfg.isEos(@intCast(first[j])) and kn.n_greedy > 1;
    }
    while (true) {
        var n: usize = 0;
        var at: [MS]usize = undefined;
        for (0..S) |j| if (live[j]) {
            rows[n] = .{ .slot = @intCast(j), .pos = @intCast(pos[j]), .token = pend[j] };
            at[j] = n;
            n += 1;
        };
        if (n == 0) break;
        try c.fwd(rows[0..n]);
        for (texts, 0..) |*t, j| if (live[j]) {
            pend[j] = c.am[at[j]];
            pos[j] += 1;
            try t.plain.append(ar, pend[j]);
            if (t.plain.items.len >= kn.n_greedy or cfg.isEos(@intCast(pend[j]))) live[j] = false;
        };
    }
    // speculative: draft, verify [pending, drafts] of every live slot in one window, accept while equal
    try prefill(c, texts, kn.rows_per, &first);
    var rounds: usize = 0;
    var drafted_rows: usize = 0;
    var accepted: usize = 0;
    var stream_rounds: usize = 0;
    var new_tokens: usize = 0;
    for (texts, 0..) |*t, j| {
        pend[j] = first[j];
        pos[j] = t.ids.len;
        try t.out.append(ar, first[j]);
        live[j] = !cfg.isEos(@intCast(first[j])) and kn.n_greedy > 1;
    }
    while (true) {
        var toks: [MS]i64 = undefined;
        var p0: [MS]u32 = undefined;
        var sl: [MS]u32 = undefined;
        var idx: [MS]usize = undefined;
        var m: usize = 0;
        for (0..S) |j| if (live[j]) {
            toks[m] = pend[j];
            p0[m] = @intCast(pos[j]);
            sl[m] = @intCast(j);
            idx[m] = j;
            m += 1;
        };
        if (m == 0) break;
        var dout: [MS * 16]i32 = undefined;
        try c.dr.draft(c.s, toks[0..m], p0[0..m], sl[0..m], &dout);
        var n: usize = 0;
        var at: [MS]usize = undefined;
        var width: [MS]usize = undefined;
        for (0..m) |q| {
            const j = idx[q];
            const left = kn.n_greedy - texts[j].out.items.len;
            width[q] = @min(B, left);
            at[q] = n;
            for (0..width[q]) |r| rows[n + r] = .{ .slot = @intCast(j), .pos = @intCast(pos[j] + r), .token = if (r == 0) pend[j] else dout[q * (B - 1) + r - 1] };
            n += width[q];
        }
        try c.fwd(rows[0..n]);
        rounds += 1;
        stream_rounds += m;
        for (0..m) |q| {
            const j = idx[q];
            const t = &texts[j];
            var a: usize = 0;
            while (a + 1 < width[q] and dout[q * (B - 1) + a] == c.am[at[q] + a]) a += 1;
            drafted_rows += width[q] - 1;
            accepted += a;
            for (0..a + 1) |k| {
                const tok = c.am[at[q] + k];
                try t.out.append(ar, tok);
                new_tokens += 1;
                if (t.out.items.len >= kn.n_greedy or cfg.isEos(@intCast(tok))) {
                    live[j] = false;
                    break;
                }
            }
            pend[j] = t.out.items[t.out.items.len - 1];
            pos[j] += a + 1;
        }
    }
    var pass = true;
    for (texts) |t| {
        const same = std.mem.eql(i32, t.plain.items, t.out.items);
        pass = pass and same;
        var first_diff: usize = 0;
        while (first_diff < @min(t.plain.items.len, t.out.items.len) and t.plain.items[first_diff] == t.out.items[first_diff]) first_diff += 1;
        std.debug.print("[drafted] {s}: plain {d} tokens, drafted {d}: {s}", .{ t.name, t.plain.items.len, t.out.items.len, if (same) "identical\n" else "DIFFER\n" });
        if (!same) std.debug.print(" at token {d}\n", .{first_diff});
        if (write) {
            try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.plain.i32", .{ out_dir, t.name }), std.mem.sliceAsBytes(t.plain.items));
            try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.drafted.i32", .{ out_dir, t.name }), std.mem.sliceAsBytes(t.out.items));
        }
    }
    std.debug.print("[drafted] {d} streams, {d} rounds, accepted {d} / drafted {d} ({d:.3}), {d:.2} tokens a stream-round: {s}\n", .{ S, rounds, accepted, drafted_rows, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(@max(drafted_rows, 1))), @as(f64, @floatFromInt(new_tokens)) / @as(f64, @floatFromInt(@max(stream_rounds, 1))), if (pass) "PASS" else "FAIL" });
}

/// TF_MUSE_DRAFT_TAPS=a,b,..: tap these target layers instead of the drafter's (plumbing checks on k-layer packs).
fn tapsOverride(buf: []usize) !?[]const usize {
    const v = std.mem.span(std.c.getenv("TF_MUSE_DRAFT_TAPS") orelse return null);
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, v, ',');
    while (it.next()) |t| : (n += 1) {
        if (n == buf.len) return error.TooManyTaps;
        buf[n] = try std.fmt.parseInt(usize, t, 10);
    }
    return buf[0..n];
}
