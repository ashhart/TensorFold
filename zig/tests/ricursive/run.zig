//! Ripple on Metal on real weights: greedy tokens, every position's logits or forced steps, for the reference's check.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fr = tf.ricursive;

const usage =
    \\tf-ricursive-run <model> <tokens.json> greedy <n>                the prompt in chunks, then n one-row steps' tokens
    \\tf-ricursive-run <model> <tokens.json> logits <out>              16-row windows from 0; bf16 logits [tokens, vocab]
    \\tf-ricursive-run <model> <tokens.json> chunked <out>[,<rows>]    every row's logits through prompt chunks of rows
    \\                                                                 (2048): the tensor-unit path's, [tokens, vocab]
    \\tf-ricursive-run <model> <tokens.json> prompt <out>              the prompt in chunks; the last row's bf16 logits
    \\tf-ricursive-run <model> <tokens.json> forced <cont.json>,<out>  then cont's tokens a step; bf16 logits before each
    \\tf-ricursive-run <model> <tokens.json> exact <cont.json>[,<w>]   cont's tokens as one-row steps and as w-row windows
    \\                                                                 (16) after the prompt: every row's logits bit for bit
    \\tf-ricursive-run <model> <tokens.json> bench <n>                 the prompt, then n one-row steps: wall, encode, GPU ms
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedModelTokensModeArg;
    }
    const gpa = init.gpa;
    const io = init.io;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const prompt_json = try readIds(gpa, io, args[2]);
    defer prompt_json.deinit();
    const prompt = prompt_json.value;
    if (prompt.len == 0) return error.EmptyPrompt;
    const mode = args[3];
    const arg = args[4];
    const m = try fr.Model.load(gpa, io, args[1]);
    defer m.deinit();
    var small = try fr.state.Scratch.init(m.device, &m.config, fr.state.batch_rows);
    defer small.deinit();
    var wide = try fr.state.Scratch.init(m.device, &m.config, fr.state.prompt_rows);
    defer wide.deinit();
    const vocab: usize = m.config.vocab;
    var t0 = now();
    if (eql(mode, "logits") or eql(mode, "chunked")) {
        var rows: usize = fr.state.window_rows;
        var path: []const u8 = arg;
        if (eql(mode, "chunked")) {
            rows = fr.state.prompt_rows;
            if (std.mem.indexOfScalar(u8, arg, ',')) |comma| {
                path = arg[0..comma];
                rows = try std.fmt.parseInt(usize, arg[comma + 1 ..], 10);
            }
            if (rows <= fr.state.batch_rows or rows > fr.state.prompt_rows) return error.ChunkRowsOutOfRange;
        }
        var cache = try fr.state.Cache.init(m.device, &m.config, prompt.len + 1);
        defer cache.deinit();
        const out = try gpa.alloc(u16, prompt.len * vocab);
        defer gpa.free(out);
        var at: usize = 0;
        while (at < prompt.len) {
            const n = @min(prompt.len - at, rows);
            if (n <= fr.state.batch_rows) {
                try fr.forward.run(m, &small, &cache, prompt[at..][0..n], .{ .all = true });
                @memcpy(out[at * vocab ..][0 .. n * vocab], small.logits_all.slice(u16, n * vocab));
            } else {
                // a chunk's every row: the chunk on the prompt path, its rows' logits one 32-row head pass at a time
                try fr.forward.run(m, &wide, &cache, prompt[at..][0..n], .{});
                try headRows(m, &wide, &small, out[at * vocab ..][0 .. n * vocab]);
            }
            cache.len += n;
            at += n;
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(out) });
        std.debug.print("ricursive {s}: {d} rows in {d} ms\n", .{ mode, prompt.len, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (eql(mode, "exact")) return exact(gpa, io, m, &small, &wide, prompt, arg);
    var forced: ?std.json.Parsed([]u32) = null;
    defer if (forced) |f| f.deinit();
    var forced_out: []const u8 = "";
    if (eql(mode, "forced")) {
        const comma = std.mem.indexOfScalar(u8, arg, ',') orelse return error.ExpectedContCommaOut;
        forced = try readIds(gpa, io, arg[0..comma]);
        forced_out = arg[comma + 1 ..];
    }
    const steps: usize = if (eql(mode, "greedy") or eql(mode, "bench")) try std.fmt.parseInt(usize, arg, 10) else if (forced) |f| f.value.len else 0;
    var cache = try fr.state.Cache.init(m.device, &m.config, prompt.len + steps + fr.state.window_rows + 1);
    defer cache.deinit();
    try feed(m, &small, &wide, &cache, prompt);
    const ttft = now() - t0;
    if (eql(mode, "prompt")) {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = arg, .data = std.mem.sliceAsBytes(cache.logits.slice(u16, vocab)) });
        std.debug.print("ricursive prompt: {d} tokens in {d} ms ({d:.0} tok/s)\n", .{ prompt.len, ttft / std.time.ns_per_ms, @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(ttft)) });
        return;
    }
    if (forced) |f| {
        const out = try gpa.alloc(u16, f.value.len * vocab);
        defer gpa.free(out);
        for (f.value, 0..) |tok, i| {
            @memcpy(out[i * vocab ..][0..vocab], cache.logits.slice(u16, vocab));
            if (i + 1 == f.value.len) break;
            try fr.forward.run(m, &small, &cache, &.{tok}, .{});
            cache.len += 1;
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = forced_out, .data = std.mem.sliceAsBytes(out) });
        return;
    }
    if (eql(mode, "bench")) {
        var tm: fr.forward.Timing = .{};
        t0 = now();
        for (0..steps) |_| {
            try fr.forward.run(m, &small, &cache, &.{argmax(cache.logits.slice(u16, vocab))}, .{ .timing = &tm });
            cache.len += 1;
        }
        const wall = @as(f64, @floatFromInt(now() - t0)) / 1e6 / @as(f64, @floatFromInt(steps));
        const per = 1e3 / @as(f64, @floatFromInt(steps));
        std.debug.print("ricursive bench at {d}: {d:.3} ms a step wall, {d:.3} encode, {d:.3} GPU\n", .{ prompt.len, wall, tm.encode * per, tm.gpu * per });
        return;
    }
    if (!eql(mode, "greedy")) {
        std.debug.print("{s}", .{usage});
        return error.UnknownMode;
    }
    const picks = try gpa.alloc(u32, steps);
    defer gpa.free(picks);
    t0 = now();
    for (picks, 0..) |*p, i| {
        p.* = argmax(cache.logits.slice(u16, vocab));
        if (i + 1 == steps) break;
        try fr.forward.run(m, &small, &cache, &.{p.*}, .{});
        cache.len += 1;
    }
    const decode = now() - t0;
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try std.json.Stringify.value(picks, .{}, &w.writer);
    std.debug.print("{s}\n", .{w.written()});
    std.debug.print("ricursive greedy: prompt {d} tokens {d} ms, {d} steps {d:.1} tok/s\n", .{ prompt.len, ttft / std.time.ns_per_ms, steps, @as(f64, @floatFromInt(steps - 1)) * 1e9 / @as(f64, @floatFromInt(decode)) });
}

/// cont's tokens after the prompt as one-row steps and as w-row windows: every row's logits must agree bit for bit.
fn exact(gpa: std.mem.Allocator, io: std.Io, m: *fr.Model, small: *fr.state.Scratch, wide: *fr.state.Scratch, prompt: []const u32, arg: []const u8) !void {
    const vocab = m.config.vocab;
    var parts = std.mem.splitScalar(u8, arg, ',');
    const cont = try readIds(gpa, io, parts.first());
    defer cont.deinit();
    const toks = cont.value;
    const width: usize = if (parts.next()) |w| try std.fmt.parseInt(usize, w, 10) else fr.state.window_rows;
    if (width == 0 or width > fr.state.window_rows) return error.BadWindow;
    // both caches at one capacity: the decode blocks follow it
    var scache = try fr.state.Cache.init(m.device, &m.config, prompt.len + toks.len + 1);
    defer scache.deinit();
    var wcache = try fr.state.Cache.init(m.device, &m.config, prompt.len + toks.len + 1);
    defer wcache.deinit();
    try feed(m, small, wide, &scache, prompt);
    try feed(m, small, wide, &wcache, prompt);
    const one = try gpa.alloc(u16, toks.len * vocab);
    defer gpa.free(one);
    for (toks, 0..) |tok, i| {
        try fr.forward.run(m, small, &scache, &.{tok}, .{ .all = true });
        @memcpy(one[i * vocab ..][0..vocab], small.logits_all.slice(u16, vocab));
        scache.len += 1;
    }
    const wins = try gpa.alloc(u16, toks.len * vocab);
    defer gpa.free(wins);
    var at: usize = 0;
    while (at < toks.len) {
        const n = @min(toks.len - at, width);
        try fr.forward.run(m, small, &wcache, toks[at..][0..n], .{ .all = true });
        @memcpy(wins[at * vocab ..][0 .. n * vocab], small.logits_all.slice(u16, n * vocab));
        wcache.len += n;
        at += n;
    }
    var differ: usize = 0;
    for (0..toks.len) |r| {
        if (std.mem.eql(u16, one[r * vocab ..][0..vocab], wins[r * vocab ..][0..vocab])) continue;
        if (differ < 4) std.debug.print("row {d} (window row {d}) differs\n", .{ r, r % width });
        differ += 1;
    }
    std.debug.print("ricursive exact at {d}: {d} of {d} rows' logits differ between one-row steps and {d}-row windows\n", .{ prompt.len, differ, toks.len, width });
    if (differ > 0) return error.RowsDiffer;
}

/// Every row of the chunk just run on `wide`: its final-normed rows through the head, 32 at a time on `small`'s logits.
fn headRows(m: *const fr.Model, wide: *fr.state.Scratch, small: *fr.state.Scratch, out: []u16) !void {
    const vocab = m.config.vocab;
    const h = m.config.hidden;
    const n = out.len / vocab;
    var at: usize = 0;
    while (at < n) {
        const r = @min(n - at, fr.state.batch_rows);
        @memcpy(small.x.slice(u16, r * h), wide.x.slice(u16, (at + r) * h)[at * h ..]);
        try fr.forward.head(m, small, r);
        @memcpy(out[at * vocab ..][0 .. r * vocab], small.logits_all.slice(u16, r * vocab));
        at += r;
    }
}

/// The prompt in prompt_rows chunks (the tensor-unit path past 32 rows), as the backend feeds it.
fn feed(m: *const fr.Model, small: *fr.state.Scratch, wide: *fr.state.Scratch, cache: *fr.state.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, fr.state.prompt_rows);
        try fr.forward.run(m, if (n > fr.state.batch_rows) wide else small, cache, ids[at..][0..n], .{});
        cache.len += n;
        at += n;
    }
}

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed([]u32) {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    return std.json.parseFromSlice([]u32, gpa, bytes, .{});
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn argmax(l: []const u16) u32 {
    var best: usize = 0;
    for (l, 0..) |x, i| if (logit(x) > logit(l[best])) {
        best = i;
    };
    return @intCast(best);
}

fn logit(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}
