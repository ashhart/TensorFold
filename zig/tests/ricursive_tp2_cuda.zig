//! `tf-ricursive-tp2 <model dir> <mode> ...`: Ripple's two-rank forward rows for the exllamav3 parity check.
const std = @import("std");
const cuda = @import("cuda");
const rc = @import("ricursive_tp2");

const usage =
    \\usage: tf-ricursive-tp2 <model dir> logits <ids> <out.f32> <first> [--truncated]
    \\       tf-ricursive-tp2 <model dir> greedy <ids> <out.txt> <count> [--truncated]
    \\       tf-ricursive-tp2 <model dir> layers <ids> <out.f32> [--truncated]
    \\       tf-ricursive-tp2 <model dir> suite <run dir> <plain|layers> [--truncated]
    \\       tf-ricursive-tp2 <model dir> long <ids> <out prefix> <tail> <stride>
    \\       tf-ricursive-tp2 <model dir> bench <ids> <decode rows>
    \\       tf-ricursive-tp2 <model dir> follow
    \\suite: for each "name n" line of <run dir>/texts.txt, name.prompt.f32, name.decode.f32 (first n/2),
    \\name.greedy.txt (32) and with layers name.layers.f32, from one load.
    \\<ids>: a file of token ids (comma, space or newline separated). logits: the first <first> ids as prompt chunks of
    \\128 rows, then every later id as its own decode row; fp32 logits [ids, vocab] (first = ids: all prompt; 0: all
    \\decode rows). greedy: the ids as a prompt, then <count> argmax tokens one decode row at a time, written as text.
    \\layers: the prompt path's fp32 residual after each layer [layers, ids, hidden]. --truncated admits a layer prefix
    \\of the release (the parity pack), which a serve never does. long: all but the last <tail> ids as prompt chunks,
    \\those as decode rows; writes prefix.nll.f32 (-log p of the next id), prefix.top.i32 (argmax) and prefix.rows.f32
    \\(rows i % stride == stride - 1 and the tail). bench: the ids as a timed prefill (depth marks), then greedy decode.
    \\Two ranks: TF_RICURSIVE_RANK (0 or 1) and NCCL_COMM_ID; rank 1 follows rank 0 whatever its mode.
    \\TF_RICURSIVE_CONTEXT (4096), TF_RICURSIVE_FULL_PROMPT=0 (full layers' prompt chunks on tree rows),
    \\TF_RICURSIVE_ROCE=1 (RoCE small gathers), TF_RICURSIVE_GRAPHS=1 (decode rows as CUDA graphs).
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const truncated = std.mem.eql(u8, args[args.len - 1], "--truncated");
    const mode = args[2];
    const is_suite = std.mem.eql(u8, mode, "suite");
    var ids: std.ArrayList(u32) = .empty;
    const pl = try place();
    if (!is_suite and pl.rank == 0 and !std.mem.eql(u8, mode, "follow")) ids = try readIds(init.io, a, args[3]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const io = init.io;
    var t0 = now(io);
    const context = if (std.c.getenv("TF_RICURSIVE_CONTEXT")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 4096;
    const m = try rc.Model.load(init.gpa, init.io, &ctx, &driver, args[1], .{ .context = context, .slots = if (std.c.getenv("TF_RICURSIVE_SLOTS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 1, .truncated = truncated, .place = pl, .full_prompt = !off("TF_RICURSIVE_FULL_PROMPT"), .roce = env("TF_RICURSIVE_ROCE"), .graphs = env("TF_RICURSIVE_GRAPHS") });
    defer m.deinit();
    std.debug.print("rank {d}/{d}: loaded {d} layers, {d:.2} GiB of weights in {d:.1} s; device {d:.2} GiB allocated; context {d}; MemAvailable {d:.1} GiB\n", .{ pl.rank, pl.world, m.c.layers, gib(m.w.bytes), now(io) - t0, gib(cuda.usage(false).device), context, memAvailable(io) });
    if (pl.rank != 0) {
        try m.follow();
        std.debug.print("rank {d}: rank 0 stopped; FOLLOW DONE\n", .{pl.rank});
        return 0;
    }
    defer m.stop() catch |err| std.debug.print("stop: {s}\n", .{@errorName(err)});
    const v = m.c.vocab;
    const n = ids.items.len;
    if (std.mem.eql(u8, mode, "logits")) {
        const first = @min(try std.fmt.parseInt(usize, args[5], 10), n);
        const out = try a.alloc(f32, n * v);
        t0 = now(io);
        if (first > 0) try m.forward(ids.items[0..first], 0, rc.forward.max_rows, out[0 .. first * v], null);
        const tp = now(io) - t0;
        for (first..n) |i| try m.forward(ids.items[i..][0..1], i, 1, out[i * v ..][0..v], null);
        std.debug.print("logits: {d} prompt rows in {d:.3} s, {d} decode rows in {d:.3} s\n", .{ first, tp, n - first, now(io) - t0 - tp });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
    } else if (std.mem.eql(u8, mode, "greedy")) {
        const count = try std.fmt.parseInt(usize, args[5], 10);
        const out = try a.alloc(f32, n * v);
        try m.forward(ids.items, 0, rc.forward.max_rows, out, null);
        var gen: std.ArrayList(u8) = .empty;
        var last = argmax(out[(n - 1) * v ..][0..v]);
        for (0..count) |i| {
            try gen.print(a, "{d}{s}", .{ last, if (i + 1 < count) " " else "\n" });
            if (i + 1 == count) break;
            try m.forward(&.{last}, n + i, 1, out[0..v], null);
            last = argmax(out[0..v]);
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = gen.items });
    } else if (std.mem.eql(u8, mode, "suite")) {
        try suite(init, a, m, args[3], std.mem.indexOf(u8, args[4], "layers") != null);
    } else if (std.mem.eql(u8, mode, "long")) {
        try long(init, a, m, ids.items, args[4], try std.fmt.parseInt(usize, args[5], 10), try std.fmt.parseInt(usize, args[6], 10));
    } else if (std.mem.eql(u8, mode, "bench")) {
        try bench(init, a, m, ids.items, try std.fmt.parseInt(usize, args[4], 10));
    } else if (std.mem.eql(u8, mode, "layers")) {
        const out = try a.alloc(f32, n * v);
        const cap = try a.alloc(f32, m.c.layers * n * m.c.hidden);
        try m.forward(ids.items, 0, rc.forward.max_rows, out, cap);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(cap) });
    } else {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    return 0;
}

fn readIds(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.ArrayList(u32) {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)), ", \n");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    return ids;
}

fn write(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8, ext: []const u8, data: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.{s}", .{ dir, name, ext }), .data = data });
}

/// Every text of <dir>/texts.txt through the prompt path, the decode path (second half), greedy 32 and the layers.
fn suite(init: std.process.Init, a: std.mem.Allocator, m: *rc.Model, dir: []const u8, layers: bool) !void {
    const io = init.io;
    const v = m.c.vocab;
    const list = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/texts.txt", .{dir}), a, .limited(1 << 16));
    var lines = std.mem.tokenizeScalar(u8, list, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next() orelse continue;
        const ids = (try readIds(io, a, try std.fmt.allocPrint(a, "{s}/{s}.ids", .{ dir, name }))).items;
        const n = ids.len;
        const out = try a.alloc(f32, n * v);
        var t0 = now(io);
        try m.forward(ids, 0, rc.forward.max_rows, out, null);
        const tp = now(io) - t0;
        try write(io, a, dir, name, "prompt.f32", std.mem.sliceAsBytes(out));
        const half = n / 2;
        try m.forward(ids[0..half], 0, rc.forward.max_rows, out[0 .. half * v], null);
        t0 = now(io);
        for (half..n) |i| try m.forward(ids[i..][0..1], i, 1, out[i * v ..][0..v], null);
        const td = now(io) - t0;
        try write(io, a, dir, name, "decode.f32", std.mem.sliceAsBytes(out));
        try m.forward(ids, 0, rc.forward.max_rows, out, null);
        var gen: std.ArrayList(u8) = .empty;
        var last = argmax(out[(n - 1) * v ..][0..v]);
        for (0..32) |i| {
            try gen.print(a, "{d}{s}", .{ last, if (i + 1 < 32) " " else "\n" });
            if (i + 1 == 32) break;
            try m.forward(&.{last}, n + i, 1, out[0..v], null);
            last = argmax(out[0..v]);
        }
        try write(io, a, dir, name, "greedy.txt", gen.items);
        if (layers) {
            const cap = try a.alloc(f32, m.c.layers * n * m.c.hidden);
            try m.forward(ids, 0, rc.forward.max_rows, out, cap);
            try write(io, a, dir, name, "layers.f32", std.mem.sliceAsBytes(cap));
        }
        std.debug.print("{s}: {d} ids, prompt {d:.3} s, {d} decode rows {d:.3} s (eager, beside nothing measured)\n", .{ name, n, tp, n - half, td });
    }
}

fn off(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

fn env(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// The node's MemAvailable in GiB (the container reads the host's /proc/meminfo), 0 if unreadable.
fn memAvailable(io: std.Io) f64 {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return 0;
    const at = std.mem.indexOf(u8, text, "MemAvailable:") orelse return 0;
    var it = std.mem.tokenizeAny(u8, text[at + 13 ..], " \n");
    const kb = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return @as(f64, @floatFromInt(kb)) / (1 << 20);
}

/// Per row: -log p(next id), the argmax, and (kept rows) the logits, appended to the three outputs.
fn score(rows: []const f32, v: usize, first: usize, ids: []const u32, tail: usize, stride: usize, nll: *std.ArrayList(f32), top: *std.ArrayList(i32), kept: *std.ArrayList(f32), a: std.mem.Allocator) !void {
    const n = rows.len / v;
    for (0..n) |r| {
        const i = first + r;
        const row = rows[r * v ..][0..v];
        const best = argmax(row);
        try top.append(a, @intCast(best));
        if (i + 1 < ids.len) {
            var s: f64 = 0;
            for (row) |x| s += @exp(@as(f64, x - row[best]));
            try nll.append(a, @floatCast(@log(s) + row[best] - row[ids[i + 1]]));
        }
        if (i % stride == stride - 1 or i + tail >= ids.len) try kept.appendSlice(a, row);
    }
}

/// The long-prompt parity rows: prompt chunks to len - tail, then decode rows (see usage).
fn long(init: std.process.Init, a: std.mem.Allocator, m: *rc.Model, ids: []const u32, prefix: []const u8, tail: usize, stride: usize) !void {
    const io = init.io;
    const v = m.c.vocab;
    const n = ids.len;
    const R = rc.forward.max_rows;
    const out = try a.alloc(f32, R * v);
    var nll: std.ArrayList(f32) = .empty;
    var top: std.ArrayList(i32) = .empty;
    var kept: std.ArrayList(f32) = .empty;
    const t0 = now(io);
    var at: usize = 0;
    while (at < n - tail) : (at += R) {
        const c = @min(R, n - tail - at);
        try m.forward(ids[at..][0..c], at, R, out[0 .. c * v], null);
        try score(out[0 .. c * v], v, at, ids, tail, stride, &nll, &top, &kept, a);
    }
    const tp = now(io) - t0;
    for (n - tail..n) |i| {
        try m.forward(ids[i..][0..1], i, 1, out[0..v], null);
        try score(out[0..v], v, i, ids, tail, stride, &nll, &top, &kept, a);
    }
    const td = now(io) - t0 - tp;
    std.debug.print("long: {d} ids, {d} prompt rows in {d:.1} s ({d:.0} tok/s), {d} decode rows {d:.1} ms each; MemAvailable {d:.1} GiB\n", .{ n, n - tail, tp, @as(f64, @floatFromInt(n - tail)) / tp, tail, 1000 * td / @as(f64, @floatFromInt(tail)), memAvailable(io) });
    for ([_][]const u8{ "nll.f32", "top.i32", "rows.f32" }, [_][]const u8{ std.mem.sliceAsBytes(nll.items), std.mem.sliceAsBytes(top.items), std.mem.sliceAsBytes(kept.items) }) |ext, data|
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, ext }), .data = data });
}

/// Timed prefill of `ids` (128-row prompt chunks, last row's logits only) with depth marks, then greedy decode rows.
fn bench(init: std.process.Init, a: std.mem.Allocator, m: *rc.Model, ids: []const u32, decode: usize) !void {
    const io = init.io;
    const v = m.c.vocab;
    const n = ids.len;
    const R = rc.forward.max_rows;
    const out = try a.alloc(f32, v);
    const t0 = now(io);
    var mark: usize = 4096;
    var since = t0;
    var since_at: usize = 0;
    var at: usize = 0;
    while (at < n) {
        const c = @min(R, n - at);
        try m.forward(ids[at..][0..c], at, R, if (c > 1) out else out[0..v], null);
        at += c;
        if (at >= mark or at == n) {
            const t = now(io);
            std.debug.print("prefill {d}: {d:.1} s, {d:.0} tok/s overall, {d:.0} tok/s over [{d}, {d}); MemAvailable {d:.1} GiB\n", .{ at, t - t0, @as(f64, @floatFromInt(at)) / (t - t0), @as(f64, @floatFromInt(at - since_at)) / (t - since), since_at, at, memAvailable(io) });
            since = t;
            since_at = at;
            while (mark <= at) mark *= 2;
        }
    }
    var tok = argmax(out[0..v]);
    var gen: std.ArrayList(u8) = .empty;
    var td: f64 = 0;
    for (0..decode) |i| {
        try gen.print(a, "{d} ", .{tok});
        const t = now(io);
        try m.forward(&.{tok}, n + i, 1, out[0..v], null);
        if (i >= 2) td += now(io) - t; // the first two rows warm up
        tok = argmax(out[0..v]);
    }
    if (decode > 2) std.debug.print("decode at {d}: {d} rows, {d:.2} ms a row, {d:.2} tok/s (eager, after 2 warm-up rows); greedy {s}\n", .{ n, decode - 2, 1000 * td / @as(f64, @floatFromInt(decode - 2)), @as(f64, @floatFromInt(decode - 2)) / td, gen.items[0..@min(gen.items.len, 120)] });
}

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

/// The first largest logit (torch's argmax rule).
fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    return @intCast(best);
}

/// This process's rank of two (TF_RICURSIVE_RANK, 0 if unset).
fn place() !rc.forward.Place {
    const r = std.c.getenv("TF_RICURSIVE_RANK") orelse return .{};
    const rank = std.fmt.parseInt(usize, std.mem.span(r), 10) catch return error.BadSetting;
    if (rank > 1) return error.BadSetting;
    return .{ .rank = rank };
}
