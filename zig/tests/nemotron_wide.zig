//! Wide-window probes for Nemotron on Metal: window cost by kernel class, and expert picks of drafted, sibling and stream rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const probe = @import("wide_probe.zig");
const streams = @import("wide_streams.zig");
const trees = @import("wide_trees.zig");
const heads_check = @import("wide_heads.zig");

const nm = tf.nemotron;
const kern = nm.kernels;
const fwd = nm.forward;
const Probe = probe.Probe;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

var io: std.Io = undefined; // the process's I/O, for model loads

const usage =
    \\usage: tf-nemotron-wide profile MODEL --prompts FILE --prompt NAME [--rows 1,2,4,8,16,32] [--reps 9] [--at 48]
    \\       tf-nemotron-wide overlap MODEL --prompts FILE --out FILE [--positions 48] [--depth 7] [--siblings 4]
    \\       tf-nemotron-wide curve MODEL --prompts FILE --prompt NAME [--rows 1,2,4,8,16] [--reps 9] [--at 48]
    \\       tf-nemotron-wide trees MODEL --prompts FILE --prompt NAME [--positions 16] [--at 8] [--reps 5]
    \\       tf-nemotron-wide treecheck MODEL --prompts FILE --prompt NAME [--positions 8] [--at 8]
    \\       tf-nemotron-wide treecost MODEL --prompts FILE --prompt NAME [--positions 8] [--at 8] [--reps 5]
    \\       tf-nemotron-wide headcheck MODEL --prompts FILE --prompt NAME [--positions 4] [--at 8] [--lanes 32]
    \\       tf-nemotron-wide heads MODEL --prompts FILE --out FILE [--positions 256] [--depth 7] [--siblings 4]
    \\       tf-nemotron-wide streams MODEL --prompts FILE [--at 48] [--reps 9]
    \\       tf-nemotron-wide serve MODEL --prompts FILE [--rows 1,2,4,8,16,32] [--tokens 128]
    \\
;

const Options = struct {
    cmd: []const u8 = "",
    model: []const u8 = "",
    prompts: []const u8 = "",
    prompt: []const u8 = "story",
    rows: []const usize = &.{ 1, 2, 4, 8, 16, 32 },
    reps: usize = 9,
    at: usize = 48,
    out: []const u8 = "",
    positions: usize = 48,
    depth: usize = 7,
    siblings: usize = 4,
    tokens: u32 = 128,
    lanes: usize = 32,
};

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tf-nemotron-wide: " ++ fmt ++ "\n" ++ usage, args);
    std.process.exit(2);
}

fn parse(arena: std.mem.Allocator, args: []const [:0]const u8) !Options {
    if (args.len < 3) fail("a command and a model", .{});
    var o = Options{ .cmd = args[1], .model = args[2] };
    var i: usize = 3;
    while (i + 1 < args.len) : (i += 2) {
        const a = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, a, "--prompts")) o.prompts = v else if (std.mem.eql(u8, a, "--prompt")) o.prompt = v else if (std.mem.eql(u8, a, "--rows")) o.rows = try probe.parseList(arena, v) else if (std.mem.eql(u8, a, "--reps")) o.reps = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--at")) o.at = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--out")) o.out = v else if (std.mem.eql(u8, a, "--positions")) o.positions = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--depth")) o.depth = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--siblings")) o.siblings = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--tokens")) o.tokens = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, a, "--lanes")) o.lanes = try std.fmt.parseInt(usize, v, 10) else fail("unknown option {s}", .{a});
    }
    if (o.prompts.len == 0) fail("--prompts is required", .{});
    return o;
}

/// Kernel classes by key prefix: each class's dispatches can be left out of a window, or summed when timed alone.
const classes = [_]struct { name: []const u8, prefixes: []const []const u8 }{
    .{ .name = "experts", .prefixes = &.{ "expert_up", "expert_down" } },
    .{ .name = "dense", .prefixes = &.{ "coop_in", "coop_out", "coop_qkv", "coop_down", "up_relu2" } },
    .{ .name = "lm_head", .prefixes = &.{ "coop_head", "tf_argmax", "sample" } },
    .{ .name = "attention", .prefixes = &.{ "attn_", "tf_kv_write", "tf_attn_q", "tf_attn_out", "xsum_4096" } },
    .{ .name = "mamba", .prefixes = &.{ "mamba_conv", "mamba_scan", "group_norm" } },
    .{ .name = "glue", .prefixes = &.{ "add_norm", "route", "xsum_2688", "tf_embed", "tf_rms", "tf_copy" } },
};

fn classOf(key: []const u8) ?usize {
    for (classes, 0..) |c, i| for (c.prefixes) |p| if (std.mem.startsWith(u8, key, p)) return i;
    return null;
}

/// Bytes of one routed expert's up and down projections (4-bit weights, bf16 scales and biases in groups of 64).
fn expertBytes(c: nm.config.Config) f64 {
    const w = c.expert_width * c.hidden / 2;
    const sb = (c.expert_width * c.hidden / 64) * 2 * 2;
    return @floatFromInt(2 * (w + sb));
}

/// Distinct experts each MoE layer reads for a logged window, summed over layers.
fn distinct(log: []const u32, layers: usize, picks: usize) usize {
    var total: usize = 0;
    for (0..layers) |l| {
        var seen: [256]bool = @splat(false);
        for (log[l * picks .. (l + 1) * picks]) |e| {
            if (!seen[e & 255]) total += 1;
            seen[e & 255] = true;
        }
    }
    return total;
}

/// A window's logits with the backend's current settings (host copy).
fn logitsOf(p: *Probe, b: *nm.backend.Metal, ids: []const u32, out: []u16) !void {
    _ = try p.window(ids, &.{}, null);
    @memcpy(out, b.scratch.logits.slice(u16, out.len));
}

fn setVariant(b: *nm.backend.Metal, v: usize, ng: usize) void {
    b.geometry = if (v < ng) v else 0;
    b.members = if (v < ng) 0 else 2 * (v - ng + 1);
}

/// Window GPU ms for each routed-expert geometry (round robin), each checked bit for bit against the generated one.
fn geometry(gpa: std.mem.Allocator, arena: std.mem.Allocator, o: Options) !void {
    const prompts = try probe.readPrompts(arena, o.prompts);
    const prompt = prompts.map.get(o.prompt) orelse fail("no prompt {s}", .{o.prompt});
    var most: usize = 1;
    for (o.rows) |r| most = @max(most, r);
    const m = try nm.Model.load(gpa, io, o.model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + o.at + most + 128, .drafts = true, .streams = 2, .batch_rows = @max(most, 16) });
    defer b.deinit();
    var truth: std.ArrayList(u32) = .empty;
    var p = try Probe.init(b);
    defer p.deinit();
    try p.prefill(prompt);
    var t = try p.first();
    for (0..o.at) |_| t = try p.step(t);
    {
        var a = try Probe.init(b);
        defer a.deinit();
        try a.prefill(prompt);
        var u = try a.first();
        for (0..o.at) |_| u = try a.step(u);
        for (0..most) |_| {
            try truth.append(arena, u);
            u = try a.step(u);
        }
    }
    // variants: each geometry, then member rows 2 and 4 a pass (generated geometry)
    const ng = kern.geometries.len + 1;
    const nv = ng + 2;
    for (o.rows) |rows| {
        const ids = truth.items[0..rows];
        const want = try arena.alloc(u16, rows * m.config.vocab);
        const got = try arena.alloc(u16, rows * m.config.vocab);
        try logitsOf(&p, b, ids, want);
        var ms: [kern.geometries.len + 3][64]f64 = undefined;
        for (0..o.reps) |rep| for (0..nv) |v| {
            setVariant(b, v, ng);
            ms[v][rep] = try p.window(ids, &.{}, null);
        };
        std.debug.print("R {d:>2}:", .{rows});
        for (0..nv) |v| {
            setVariant(b, v, ng);
            try logitsOf(&p, b, ids, got);
            const same = std.mem.eql(u16, want, got);
            if (v < ng) {
                const g: [2]usize = if (v == 0) .{ 4, 2 } else kern.geometries[v - 1];
                std.debug.print(" {d}x{d} {d:.3}{s}", .{ g[0], g[1], probe.median(ms[v][0..o.reps]), if (same) "" else " DIFFER" });
            } else std.debug.print(" rows{d} {d:.3}{s}", .{ b.members, probe.median(ms[v][0..o.reps]), if (same) "" else " DIFFER" });
        }
        setVariant(b, 0, ng);
        std.debug.print("\n", .{});
    }
}

fn profile(gpa: std.mem.Allocator, arena: std.mem.Allocator, o: Options) !void {
    const prompts = try probe.readPrompts(arena, o.prompts);
    const prompt = prompts.map.get(o.prompt) orelse fail("no prompt {s}", .{o.prompt});
    var most: usize = 1;
    for (o.rows) |r| most = @max(most, r);
    const m = try nm.Model.load(gpa, io, o.model, true);
    defer m.deinit();
    const c = m.config;
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + o.at + most + 128, .drafts = true, .streams = 2, .batch_rows = @max(most, 16) });
    defer b.deinit();
    const kept = 3; // a verify's replayed rows: the last window's kept rows

    // the greedy continuation, then a fresh stream at `at` tokens into it with `kept` rows to replay
    var truth: std.ArrayList(u32) = .empty;
    {
        var a = try Probe.init(b);
        defer a.deinit();
        try a.prefill(prompt);
        var t = try a.first();
        for (0..o.at + kept + most) |_| {
            try truth.append(arena, t);
            t = try a.step(t);
        }
    }
    var p = try Probe.init(b);
    defer p.deinit();
    const ctx = try std.mem.concat(arena, u32, &.{ prompt, truth.items[0..o.at] });
    try p.prefill(ctx);
    try p.keep(truth.items[o.at .. o.at + kept]);
    std.debug.print("profile {s}: windows at position {d} replaying {d} rows, {d} reps a variant, round robin\n", .{ o.prompt, ctx.len + kept, kept, o.reps });

    var drops: [classes.len][]mtl.objc.Id = undefined;
    for (classes, 0..) |cl, i| drops[i] = probe.pipelines(&m.kernels, cl.prefixes, try arena.alloc(mtl.objc.Id, kern.total));
    const log = try m.device.buffer(c.layers * most * c.top_k * 4, opts);
    defer log.deinit();
    const moes = c.count(.moe);

    for (o.rows) |rows| {
        const ids = truth.items[o.at + kept .. o.at + kept + rows];
        // variants: the window (replaying `kept` rows), then each class left out, then per-row stores with no replay
        const nv = 2 + classes.len;
        var ms: [2 + classes.len][64]f64 = undefined;
        for (0..2) |_| _ = try p.window(ids, &.{}, null);
        for (0..o.reps) |rep| for (0..nv) |v| {
            ms[v][rep] = if (v == nv - 1) try p.windowAs(ids, .all, &.{}, null) else try p.window(ids, if (v >= 1) drops[v - 1] else &.{}, null);
        };
        var med: [2 + classes.len]f64 = undefined;
        for (0..nv) |v| med[v] = probe.median(ms[v][0..o.reps]);

        var alone: [classes.len + 1]f64 = @splat(0);
        var dispatches: usize = 0;
        var prof = fwd.Profiler{ .queue = m.queue, .kernels = &m.kernels };
        const alone_reps = 3;
        for (0..alone_reps) |_| _ = try p.window(ids, &.{}, &prof);
        for (prof.ms, prof.calls, 0..) |v, n, i| {
            if (n == 0) continue;
            dispatches += n / alone_reps;
            alone[classOf(kern.keyOf(i)) orelse classes.len] += v / alone_reps;
        }
        var alone_sum: f64 = 0;
        for (alone) |v| alone_sum += v;

        b.scratch.route_log = log;
        b.scratch.route_at = 0;
        _ = try p.window(ids, &.{}, null);
        b.scratch.route_log = null;
        const used = distinct(log.slice(u32, moes * rows * c.top_k), moes, rows * c.top_k);
        const gb = @as(f64, @floatFromInt(used)) * expertBytes(c) / 1e9;

        std.debug.print("R {d:>2}: GPU ms {d:.3} (per-row stores {d:.3}) | left out:", .{ rows, med[0], med[nv - 1] });
        for (classes, 0..) |cl, i| std.debug.print(" {s} -{d:.3}", .{ cl.name, med[0] - med[1 + i] });
        std.debug.print(" | alone:", .{});
        for (classes, 0..) |cl, i| std.debug.print(" {s} {d:.3}", .{ cl.name, alone[i] });
        std.debug.print(" other {d:.3} sum {d:.3} ({d} dispatches) | experts {d} ({d:.3} GB, {d:.0} GB/s alone, {d:.0} in the window)\n", .{ alone[classes.len], alone_sum, dispatches, used, gb, gb / (alone[0] / 1e3), gb / ((med[0] - med[1]) / 1e3) });
    }
}

/// Window cost curves in one process, round robin: per-row stores (before), replayed rows, replayed rows + 2-row experts.
fn curve(gpa: std.mem.Allocator, arena: std.mem.Allocator, o: Options) !void {
    const prompts = try probe.readPrompts(arena, o.prompts);
    const prompt = prompts.map.get(o.prompt) orelse fail("no prompt {s}", .{o.prompt});
    var most: usize = 1;
    for (o.rows) |r| most = @max(most, r);
    const m = try nm.Model.load(gpa, io, o.model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + o.at + most + 128, .drafts = true, .streams = 2, .batch_rows = @max(most, 16) });
    defer b.deinit();
    const kept = 3;
    var truth: std.ArrayList(u32) = .empty;
    {
        var a = try Probe.init(b);
        defer a.deinit();
        try a.prefill(prompt);
        var t = try a.first();
        for (0..o.at + kept + most) |_| {
            try truth.append(arena, t);
            t = try a.step(t);
        }
    }
    var p = try Probe.init(b);
    defer p.deinit();
    try p.prefill(try std.mem.concat(arena, u32, &.{ prompt, truth.items[0..o.at] }));
    try p.keep(truth.items[o.at .. o.at + kept]);
    std.debug.print("{s}: GPU ms, median of {d}: rows, per-row stores (before), replay {d} rows, replay + 2-row experts\n", .{ o.prompt, o.reps, kept });
    for (o.rows) |rows| {
        const ids = truth.items[o.at + kept .. o.at + kept + rows];
        var ms: [3][64]f64 = undefined;
        for (0..2) |_| _ = try p.window(ids, &.{}, null);
        for (0..o.reps) |rep| for (0..3) |v| {
            b.members = if (v == 2) 2 else 0;
            ms[v][rep] = try p.windowAs(ids, if (v == 0) .all else .lag, &.{}, null);
        };
        b.members = 0;
        std.debug.print("{d:>2} {d:.3} {d:.3} {d:.3}\n", .{ rows, probe.median(ms[0][0..o.reps]), probe.median(ms[1][0..o.reps]), probe.median(ms[2][0..o.reps]) });
    }
}

/// `, "key": [a, b, ...]` onto a JSON line.
fn ints(out: *std.ArrayList(u8), arena: std.mem.Allocator, key: []const u8, v: []const u32) !void {
    try out.print(arena, ", \"{s}\": [", .{key});
    for (v, 0..) |x, i| try out.print(arena, "{s}{d}", .{ if (i > 0) ", " else "", x });
    try out.append(arena, ']');
}

fn overlap(gpa: std.mem.Allocator, arena: std.mem.Allocator, init: std.process.Init, o: Options) !void {
    const prompts = try probe.readPrompts(arena, o.prompts);
    const m = try nm.Model.load(gpa, io, o.model, true);
    defer m.deinit();
    const c = m.config;
    var longest: usize = 0;
    for (prompts.map.values()) |v| longest = @max(longest, v.len);
    const width = o.depth + 1;
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = longest + o.positions + 64, .drafts = true, .streams = 2, .batch_rows = @max(width, 16) });
    defer b.deinit();
    const moes = c.count(.moe);
    const log = try m.device.buffer(moes * width * c.top_k * 4, opts);
    defer log.deinit();
    var out: std.ArrayList(u8) = .empty;
    const t0 = mtl.clock.seconds();

    for (prompts.map.keys(), prompts.map.values()) |name, ids| {
        var p = try Probe.init(b);
        defer p.deinit();
        try p.prefill(ids);
        var t = try p.first();
        var row = p.c.rows - 1;
        for (0..o.positions) |pos| {
            // the head's top siblings at depth 1, then its chain to `depth` (the cache keeps one row either way)
            const kept = p.c.mtp_len;
            _ = try p.drafts(row, t, 1);
            var sibs: [8]u32 = undefined;
            p.topDrafts(o.siblings, &sibs);
            p.c.mtp_len = kept;
            var window: [64]u32 = undefined;
            window[0] = t;
            @memcpy(window[1..width], try p.drafts(row, t, o.depth));

            try out.print(arena, "{{\"prompt\": \"{s}\", \"pos\": {d}, \"t\": {d}", .{ name, pos, t });
            try ints(&out, arena, "drafts", window[1..width]);
            try ints(&out, arena, "sibs", sibs[0..o.siblings]);
            b.scratch.route_log = log;
            b.scratch.route_at = 0;
            _ = try p.window(window[0..width], &.{}, null);
            try ints(&out, arena, "cons", log.slice(u32, moes * width * c.top_k));
            for (1..o.siblings) |s| {
                b.scratch.route_at = 0;
                _ = try p.window(&.{ t, sibs[s] }, &.{}, null);
                try ints(&out, arena, try std.fmt.allocPrint(arena, "sib{d}", .{s}), log.slice(u32, moes * 2 * c.top_k));
            }
            b.scratch.route_log = null;
            const next = try p.step(t);
            try out.print(arena, ", \"next\": {d}}}\n", .{next});
            t = next;
            row = 0;
        }
        std.debug.print("{s}: {d} positions ({d:.1} s)\n", .{ name, o.positions, mtl.clock.seconds() - t0 });
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = o.out, .data = out.items });
}

/// Each position's greedy token and the head's top tokens and probabilities at each chained level (JSON lines).
fn heads(gpa: std.mem.Allocator, arena: std.mem.Allocator, init: std.process.Init, o: Options) !void {
    const prompts = try probe.readPrompts(arena, o.prompts);
    const m = try nm.Model.load(gpa, io, o.model, true);
    defer m.deinit();
    var longest: usize = 0;
    for (prompts.map.values()) |v| longest = @max(longest, v.len);
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = longest + o.positions + 64, .drafts = true, .streams = 2, .batch_rows = 16 });
    defer b.deinit();
    var out: std.ArrayList(u8) = .empty;
    const k = o.siblings;
    for (prompts.map.keys(), prompts.map.values()) |name, ids| {
        var p = try Probe.init(b);
        defer p.deinit();
        try p.prefill(ids);
        var t = try p.first();
        var row = p.c.rows - 1;
        for (0..o.positions) |pos| {
            var toks: [16][8]u32 = undefined;
            var probs: [16][8]f64 = undefined;
            try p.headLevels(row, t, o.depth, k, toks[0..o.depth], probs[0..o.depth]);
            try out.print(arena, "{{\"prompt\": \"{s}\", \"pos\": {d}, \"t\": {d}, \"lv\": [", .{ name, pos, t });
            for (0..o.depth) |j| {
                try out.print(arena, "{s}[", .{if (j > 0) ", " else ""});
                for (0..k) |r| try out.print(arena, "{s}[{d}, {d:.6}]", .{ if (r > 0) ", " else "", toks[j][r], probs[j][r] });
                try out.append(arena, ']');
            }
            try out.appendSlice(arena, "]}\n");
            t = try p.step(t);
            row = 0;
        }
        std.debug.print("{s}: {d} positions\n", .{ name, o.positions });
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = o.out, .data = out.items });
}

pub fn main(init: std.process.Init) !void {
    io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const o = try parse(arena, try init.minimal.args.toSlice(arena));
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (std.mem.eql(u8, o.cmd, "profile")) return profile(gpa, arena, o);
    if (std.mem.eql(u8, o.cmd, "geometry")) return geometry(gpa, arena, o);
    if (std.mem.eql(u8, o.cmd, "curve")) return curve(gpa, arena, o);
    if (std.mem.eql(u8, o.cmd, "trees")) return trees.run(gpa, arena, init.io, o.model, o.prompts, o.prompt, o.positions, o.at, o.reps);
    if (std.mem.eql(u8, o.cmd, "treecost")) return trees.cost(gpa, arena, init.io, o.model, o.prompts, o.prompt, o.positions, o.at, o.reps);
    if (std.mem.eql(u8, o.cmd, "headcheck")) return heads_check.check(gpa, arena, init.io, o.model, o.prompts, o.prompt, o.positions, o.at, o.lanes);
    if (std.mem.eql(u8, o.cmd, "treecheck")) return trees.check(gpa, arena, init.io, o.model, o.prompts, o.prompt, o.positions, o.at);
    if (std.mem.eql(u8, o.cmd, "heads")) {
        if (o.out.len == 0) fail("--out is required", .{});
        return heads(gpa, arena, init, o);
    }
    if (std.mem.eql(u8, o.cmd, "streams")) return streams.costs(gpa, arena, init.io, o.model, o.prompts, o.at, o.reps);
    if (std.mem.eql(u8, o.cmd, "serve")) return streams.serve(gpa, arena, init.io, o.model, o.prompts, o.rows, o.tokens);
    if (std.mem.eql(u8, o.cmd, "overlap")) {
        if (o.out.len == 0) fail("--out is required", .{});
        return overlap(gpa, arena, init, o);
    }
    fail("unknown command {s}", .{o.cmd});
}
