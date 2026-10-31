//! Llama on Metal on real weights: greedy tokens, every position's logits or forced steps, for a reference.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const fl = tf.llama;

const usage =
    \\tf-llama-run <model> <tokens.json> greedy <n>                the prompt in chunks, then n one-row steps' tokens
    \\tf-llama-run <model> <tokens.json> logits <out>              16-row windows from 0; bf16 logits [tokens, vocab]
    \\tf-llama-run <model> <tokens.json> prompt <out>              the prompt in chunks; the last row's bf16 logits
    \\tf-llama-run <model> <tokens.json> forced <cont.json>,<out>  then cont's tokens a step; bf16 logits before each
    \\tf-llama-run <model> <tokens.json> exact <cont.json>[,<w>]  cont's tokens as one-row steps and as w-row windows
    \\                                                             (16) after the prompt: every row's logits bit for bit,
    \\                                                             failing on any difference or a missed planted row
    \\tf-llama-run <model> <tokens.json> bench <n>                 the prompt, then n one-row steps: wall, encode, GPU ms
    \\tf-llama-run <model> <tokens.json> draft <drafter>,<n>       greedy's n tokens through the drafter's windows: the same
    \\                                                             tokens, with the rounds, drafts kept and tok/s
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
    var drafter_dir: []const u8 = "";
    if (std.mem.eql(u8, mode, "draft")) {
        const comma = std.mem.indexOfScalar(u8, args[4], ',') orelse return error.ExpectedDrafterCommaSteps;
        drafter_dir = args[4][0..comma];
    }
    const steps: usize = if (std.mem.eql(u8, mode, "greedy") or std.mem.eql(u8, mode, "bench")) try std.fmt.parseInt(usize, args[4], 10) else if (drafter_dir.len > 0) try std.fmt.parseInt(usize, args[4][drafter_dir.len + 1 ..], 10) else if (forced) |f| f.value.len else 0;
    var cache = try fl.state.Cache.init(m.device, &m.config, prompt.len + steps + fl.state.window_rows + 1);
    defer cache.deinit();
    var small = try fl.state.Scratch.init(m.device, &m.config, fl.state.batch_rows);
    defer small.deinit();
    var wide = try fl.state.Scratch.init(m.device, &m.config, fl.state.prompt_rows);
    defer wide.deinit();
    const vocab: usize = m.config.vocab;
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
        std.debug.print("llama logits: {d} rows in {d} ms\n", .{ prompt.len, (now() - t0) / std.time.ns_per_ms });
        return;
    }
    if (std.mem.eql(u8, mode, "exact")) {
        var parts = std.mem.splitScalar(u8, args[4], ',');
        const text = try std.Io.Dir.cwd().readFileAlloc(init.io, parts.first(), gpa, .limited(1 << 20));
        defer gpa.free(text);
        const cont = try std.json.parseFromSlice([]u32, gpa, text, .{});
        defer cont.deinit();
        const toks = cont.value;
        const width: usize = if (parts.next()) |w| try std.fmt.parseInt(usize, w, 10) else fl.state.window_rows;
        // both caches at one capacity: the decode blocks follow it
        var scache = try fl.state.Cache.init(m.device, &m.config, prompt.len + toks.len + 1);
        defer scache.deinit();
        var wcache = try fl.state.Cache.init(m.device, &m.config, prompt.len + toks.len + 1);
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
        if (width == 0 or width > fl.state.window_rows) return error.BadWindow;
        const wins = try gpa.alloc(u16, toks.len * vocab);
        defer gpa.free(wins);
        var at: usize = 0;
        while (at < toks.len) {
            const n = @min(toks.len - at, width);
            try fl.forward.run(m, &small, &wcache, toks[at..][0..n], .{ .all = true });
            @memcpy(wins[at * vocab ..][0 .. n * vocab], small.logits_all.slice(u16, n * vocab));
            wcache.len += n;
            at += n;
        }
        var first: [4]usize = undefined;
        const differ = fl.exact.differing(one, wins, vocab, &first);
        for (first[0..@min(differ, first.len)]) |r| std.debug.print("row {d} (window row {d}) differs\n", .{ r, r % width });
        std.debug.print("llama exact at {d}: {d} of {d} rows' logits differ between one-row steps and windows\n", .{ prompt.len, differ, toks.len });
        // the control: a row planted in a copy of the steps' logits must be the one difference reported
        const planted = try gpa.dupe(u16, one);
        defer gpa.free(planted);
        fl.exact.plant(planted, vocab, toks.len / 2);
        if (fl.exact.differing(one, planted, vocab, &first) != 1 or first[0] != toks.len / 2) return error.PlantedRowMissed;
        std.debug.print("llama exact: the planted row {d} is reported\n", .{toks.len / 2});
        if (differ > 0) return error.RowsDiffer;
        return;
    }
    try feed(m, &small, &wide, &cache, prompt);
    const ttft = (now() - t0);
    if (std.mem.eql(u8, mode, "prompt")) {
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(cache.logits.slice(u16, vocab)) });
        std.debug.print("llama prompt: {d} tokens in {d} ms ({d:.0} tok/s)\n", .{ prompt.len, ttft / std.time.ns_per_ms, @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(ttft)) });
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
        std.debug.print("llama bench at {d}: {d:.3} ms a step wall, {d:.3} encode, {d:.3} GPU\n", .{ prompt.len, wall, tm.encode * per, tm.gpu * per });
        return;
    }
    if (drafter_dir.len > 0) return draft(gpa, init.io, m, &small, &cache, drafter_dir, prompt, steps);
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
    std.debug.print("llama greedy: prompt {d} tokens {d} ms, {d} steps {d:.1} tok/s\n", .{ prompt.len, ttft / std.time.ns_per_ms, steps, @as(f64, @floatFromInt(steps - 1)) * 1e9 / @as(f64, @floatFromInt(decode)) });
}

/// Greedy decoding through the drafter's windows after the prompt: drafts kept while the target's rows agree.
fn draft(gpa: std.mem.Allocator, io: std.Io, m: *fl.Model, small: *fl.state.Scratch, cache: *fl.state.Cache, dir: []const u8, prompt: []const u32, steps: usize) !void {
    const vocab = m.config.vocab;
    const dm = try fl.Model.loadOn(gpa, io, dir, m.queue);
    const d = try fl.drafter.Drafter.init(gpa, dm, .{ .capacity = prompt.len + steps + 2 * fl.state.window_rows, .streams = 1 });
    defer d.deinit();
    try d.warm();
    const p = try fl.drafter.Drafter.make(d, gpa, prompt, m.config.eos[0..m.config.eos_count]);
    defer fl.drafter.Drafter.free(d, p);
    var context: std.ArrayList(u32) = .empty;
    defer context.deinit(gpa);
    try context.appendSlice(gpa, prompt);
    var picks: std.ArrayList(u32) = .empty;
    defer picks.deinit(gpa);
    var pending = argmax(cache.logits.slice(u16, vocab));
    try picks.append(gpa, pending);
    try context.append(gpa, pending);
    var rounds: usize = 0;
    var offered: usize = 0;
    var landed: usize = 0;
    var ids: [fl.state.window_rows]u32 = undefined;
    const t0 = now();
    while (picks.items.len < steps) {
        const drafts = try p.propose(context.items, fl.state.window_rows - 1);
        ids[0] = pending;
        @memcpy(ids[1..][0..drafts.len], drafts);
        const rows = 1 + drafts.len;
        try fl.forward.run(m, small, cache, ids[0..rows], .{ .all = true });
        const all = small.logits_all.slice(u16, rows * vocab);
        var keep: usize = 0;
        while (keep < drafts.len and argmax(all[keep * vocab ..][0..vocab]) == drafts[keep]) keep += 1;
        pending = argmax(all[keep * vocab ..][0..vocab]);
        cache.len += keep + 1;
        try picks.appendSlice(gpa, drafts[0..keep]);
        try picks.append(gpa, pending);
        try context.appendSlice(gpa, drafts[0..keep]);
        try context.append(gpa, pending);
        p.round(@intCast(rows), @intCast(keep));
        rounds += 1;
        offered += drafts.len;
        landed += keep;
    }
    const decode = now() - t0;
    picks.shrinkRetainingCapacity(steps);
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    try std.json.Stringify.value(picks.items, .{}, &w.writer);
    std.debug.print("{s}\n", .{w.written()});
    const per_round = @as(f64, @floatFromInt(rounds + landed)) / @as(f64, @floatFromInt(@max(rounds, 1)));
    std.debug.print("llama draft: prompt {d} tokens, {d} tokens in {d} rounds ({d:.2} a round), {d} of {d} drafts kept, {d:.1} tok/s; step {d:.2} ms, rounds {d:.1} ms at 4 rows, {d:.1} at 8\n", .{ prompt.len, steps, rounds, per_round, landed, offered, @as(f64, @floatFromInt(steps - 1)) * 1e9 / @as(f64, @floatFromInt(decode)), d.step_ms orelse 0, d.rounds.at(4), d.rounds.at(8) });
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
