//! Spark2.5 on Metal on real weights: greedy tokens, every position's logits or forced steps, for a reference.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fl = tf.spark25_metal;

const usage =
    \\tf-spark-run <model> <tokens.json> greedy <n>                the prompt in chunks, then n one-row steps' tokens
    \\tf-spark-run <model> <tokens.json> logits <out>              16-row windows from 0; bf16 logits [tokens, vocab]
    \\tf-spark-run <model> <tokens.json> prompt <out>              the prompt in chunks; the last row's bf16 logits
    \\tf-spark-run <model> <tokens.json> forced <cont.json>,<out>  then cont's tokens a step; bf16 logits before each
    \\tf-spark-run <model> <tokens.json> exact <cont.json>        cont's tokens as one-row steps and as 16-row windows
    \\                                                             after the prompt: every row's logits bit for bit
    \\tf-spark-run <model> <tokens.json> bench <n>                 the prompt, then n one-row steps: wall, encode, GPU ms
    \\SPARK_LAYERS=N runs the first N layers only (a truncated reference's check)
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedModelTokensModeArg;
    }
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const ids = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer ids.deinit();
    const prompt = ids.value;
    if (prompt.len == 0) return error.EmptyPrompt;
    const m = try fl.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    // SPARK_LAYERS=N: the first N layers only, for a layer-by-layer check against a truncated reference
    if (std.c.getenv("SPARK_LAYERS")) |v| m.weights.layers = m.weights.layers[0..@min(m.weights.layers.len, try std.fmt.parseInt(usize, std.mem.span(v), 10))];
    const mode = args[3];
    var forced: ?std.json.Parsed([]u32) = null;
    defer if (forced) |f| f.deinit();
    var forced_out: []const u8 = "";
    if (std.mem.eql(u8, mode, "forced")) {
        const comma = std.mem.indexOfScalar(u8, args[4], ',') orelse return error.ExpectedContCommaOut;
        const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[4][0..comma], gpa, .limited(1 << 20));
        defer gpa.free(text);
        forced = try std.json.parseFromSlice([]u32, gpa, text, .{});
        forced_out = args[4][comma + 1 ..];
    }
    const steps: usize = if (std.mem.eql(u8, mode, "greedy") or std.mem.eql(u8, mode, "bench")) try std.fmt.parseInt(usize, args[4], 10) else if (forced) |f| f.value.len else 0;
    var cache = try fl.state.Cache.init(m.device, prompt.len + steps + 1);
    defer cache.deinit();
    var small = try fl.state.Scratch.init(m.device, fl.state.batch_rows);
    defer small.deinit();
    var wide = try fl.state.Scratch.init(m.device, fl.state.prompt_rows);
    defer wide.deinit();
    const vocab: usize = fl.config.vocab;
    var t0 = now();
    if (std.mem.eql(u8, mode, "logits")) {
        const out = try gpa.alloc(u16, prompt.len * vocab);
        defer gpa.free(out);
        var at: usize = 0;
        while (at < prompt.len) {
            const n = @min(prompt.len - at, fl.state.window_rows);
            try fl.forward.run(m, &small, &cache, prompt[at..][0..n], .{ .all = true });
            @memcpy(out[at * vocab ..][0 .. n * vocab], small.logits_all.slice(u16, n * vocab));
            cache.len += n;
            at += n;
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(out) });
        std.debug.print("spark logits: {d} rows in {d} ms\n", .{ prompt.len, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (std.mem.eql(u8, mode, "exact")) {
        const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[4], gpa, .limited(1 << 20));
        defer gpa.free(text);
        const cont = try std.json.parseFromSlice([]u32, gpa, text, .{});
        defer cont.deinit();
        const toks = cont.value;
        // both caches at one capacity: a full layer's decode blocks follow it
        var scache = try fl.state.Cache.init(m.device, prompt.len + toks.len + 1);
        defer scache.deinit();
        var wcache = try fl.state.Cache.init(m.device, prompt.len + toks.len + 1);
        defer wcache.deinit();
        try feed(m, &small, &wide, &scache, prompt);
        try feed(m, &small, &wide, &wcache, prompt);
        const one = try gpa.alloc(u16, toks.len * vocab);
        defer gpa.free(one);
        for (toks, 0..) |tok, i| {
            try fl.forward.run(m, &small, &scache, &.{tok}, .{ .all = true });
            @memcpy(one[i * vocab ..][0..vocab], small.logits_all.slice(u16, vocab));
            scache.len += 1;
        }
        // SPARK_WINDOW=W: windows of W rows (16 by default)
        const width: usize = if (std.c.getenv("SPARK_WINDOW")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else fl.state.window_rows;
        if (width == 0 or width > fl.state.window_rows) return error.BadWindow;
        var differ: usize = 0;
        var at: usize = 0;
        while (at < toks.len) {
            const n = @min(toks.len - at, width);
            try fl.forward.run(m, &small, &wcache, toks[at..][0..n], .{ .all = true });
            for (0..n) |r| {
                if (!std.mem.eql(u16, one[(at + r) * vocab ..][0..vocab], small.logits_all.slice(u16, n * vocab)[r * vocab ..][0..vocab])) {
                    differ += 1;
                    if (differ <= 4) std.debug.print("row {d} (window row {d}) differs\n", .{ at + r, r });
                }
            }
            wcache.len += n;
            at += n;
        }
        std.debug.print("spark exact at {d}: {d} of {d} rows' logits differ between one-row steps and windows\n", .{ prompt.len, differ, toks.len });
        return;
    }
    try feed(m, &small, &wide, &cache, prompt);
    const ttft = (now() - t0);
    if (std.mem.eql(u8, mode, "prompt")) {
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(cache.logits.slice(u16, vocab)) });
        std.debug.print("spark prompt: {d} tokens in {d} ms ({d:.0} tok/s)\n", .{ prompt.len, ttft / std.time.ns_per_ms, @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(ttft)) });
        return;
    }
    if (forced) |f| {
        const out = try gpa.alloc(u16, f.value.len * vocab);
        defer gpa.free(out);
        for (f.value, 0..) |tok, i| {
            @memcpy(out[i * vocab ..][0..vocab], cache.logits.slice(u16, vocab));
            if (i + 1 == f.value.len) break;
            try fl.forward.run(m, &small, &cache, &.{tok}, .{});
            cache.len += 1;
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = forced_out, .data = std.mem.sliceAsBytes(out) });
        return;
    }
    if (std.mem.eql(u8, mode, "bench")) {
        const n = try std.fmt.parseInt(usize, args[4], 10);
        var tm: fl.forward.Timing = .{};
        t0 = now();
        for (0..n) |_| {
            try fl.forward.run(m, &small, &cache, &.{argmax(cache.logits.slice(u16, vocab))}, .{ .timing = &tm });
            cache.len += 1;
        }
        const wall = @as(f64, @floatFromInt(now() - t0)) / 1e6 / @as(f64, @floatFromInt(n));
        const per = 1e3 / @as(f64, @floatFromInt(n));
        std.debug.print("spark bench at {d}: {d:.3} ms a step wall, {d:.3} encode, {d:.3} GPU\n", .{ prompt.len, wall, tm.encode * per, tm.gpu * per });
        return;
    }
    if (!std.mem.eql(u8, mode, "greedy")) {
        std.debug.print("{s}", .{usage});
        return error.UnknownMode;
    }
    const picks = try gpa.alloc(u32, steps);
    defer gpa.free(picks);
    t0 = now();
    for (picks, 0..) |*p, i| {
        p.* = argmax(cache.logits.slice(u16, vocab));
        if (i + 1 == steps) break;
        try fl.forward.run(m, &small, &cache, &.{p.*}, .{});
        cache.len += 1;
    }
    const decode = (now() - t0);
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try std.json.Stringify.value(picks, .{}, &w.writer);
    std.debug.print("{s}\n", .{w.written()});
    std.debug.print("spark greedy: prompt {d} tokens {d} ms, {d} steps {d:.1} tok/s\n", .{ prompt.len, ttft / std.time.ns_per_ms, steps, @as(f64, @floatFromInt(steps - 1)) * 1e9 / @as(f64, @floatFromInt(decode)) });
}

/// The prompt in prompt_rows chunks (the tensor-unit path past 16 rows), as the backend feeds it.
fn feed(m: *const fl.Model, small: *fl.state.Scratch, wide: *fl.state.Scratch, cache: *fl.state.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, fl.state.prompt_rows);
        try fl.forward.run(m, if (n > fl.state.batch_rows) wide else small, cache, ids[at..][0..n], .{});
        cache.len += n;
        at += n;
    }
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
