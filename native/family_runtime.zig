//! Shared completion/verification driver for Metal families with chained MTP heads.
const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");
const Cache = @import("model.zig").Cache;
const Stopwatch = @import("vendor/io_util.zig").Stopwatch;
pub fn run(comptime M: type, init: std.process.Init, args: []const []const u8) !void {
    const a = init.gpa;
    const io = init.io;
    var prompt: []const u8 = "Write a short Python function that computes the Fibonacci sequence.";
    var token_list: ?[]const u8 = null;
    var max_tokens: usize = 32;
    var drafts: usize = 3;
    var settings = sampling.Sampling{};
    var seed_set = false;
    var report: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var exact = false;
    var warm = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const key = args[i];
        if (std.mem.eql(u8, key, "--metal-sampling")) {
            settings.metal = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--metal-simd")) {
            mx.force_simd = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--no-drafts")) {
            drafts = 0;
            continue;
        }
        if (std.mem.eql(u8, key, "--warmup")) {
            warm = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-exact")) {
            exact = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        const val = args[i + 1];
        if (std.mem.eql(u8, key, "--prompt")) prompt = val else if (std.mem.eql(u8, key, "--tokens")) token_list = val else if (std.mem.eql(u8, key, "--max-tokens")) max_tokens = try std.fmt.parseInt(usize, val, 10) else if (std.mem.eql(u8, key, "--mtp-drafts")) drafts = try std.fmt.parseInt(usize, val, 10) else if (std.mem.eql(u8, key, "--temperature")) settings.temperature = try std.fmt.parseFloat(f64, val) else if (std.mem.eql(u8, key, "--top-k")) settings.top_k = try std.fmt.parseInt(usize, val, 10) else if (std.mem.eql(u8, key, "--top-p")) settings.top_p = try std.fmt.parseFloat(f64, val) else if (std.mem.eql(u8, key, "--seed")) {
            settings.seed = try std.fmt.parseInt(u64, val, 10);
            seed_set = true;
        } else if (std.mem.eql(u8, key, "--report")) report = val else if (std.mem.eql(u8, key, "--dump-logits")) dump = val else return error.UnknownArgument;
        i += 1;
    }
    if (drafts > 15) return error.InvalidDraftBudget;
    try settings.validate();
    try mx.init();
    defer mx.shutdown();
    var m = try M.init(io, args[2], drafts > 0 and !exact);
    defer m.deinit();
    if (exact) {
        try check(M, &m);
        return;
    }
    if (warm) {
        var p = try m.forward(&.{42});
        defer p.deinit();
        try m.commit(&p, 1);
        m.reset();
    }
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, args[2], a);
    defer a.free(path);
    var tok = try @import("vendor/tokenizer.zig").loadTokenizer(io, a, path);
    defer tok.deinit();
    var tokens: std.ArrayList(i32) = .empty;
    defer tokens.deinit(a);
    if (token_list) |list| {
        var split = std.mem.splitScalar(u8, list, ',');
        while (split.next()) |v| try tokens.append(a, try std.fmt.parseInt(i32, v, 10));
    } else {
        const ids = try tok.encode(a, prompt);
        defer a.free(ids);
        for (ids) |id| try tokens.append(a, @intCast(id));
    }
    if (tokens.items.len == 0) return error.EmptyPrompt;
    if (tokens.items.len > 262144 or max_tokens > 262144 - tokens.items.len) return error.ContextLimitExceeded;
    for (tokens.items) |id| if (id < 0 or id >= M.vocab) {
        return error.InvalidToken;
    };
    if (!seed_set) settings.seed = sampling.seedFor(tokens.items);
    var head_cache = M.DraftCache{};
    defer head_cache.deinit();
    var last = mx.empty;
    defer mx.free(last);
    var timer = Stopwatch.init(io);
    var pending: i32 = 0;
    var off: usize = 0;
    while (off < tokens.items.len) {
        const n = @min(16, tokens.items.len - off);
        var p = try m.forward(tokens.items[off..][0..n]);
        defer p.deinit();
        if (m.mtp) for (0..n) |j| {
            if (last.ctx != null) _ = try m.draftStep(&p.scope, last, tokens.items[off + j], &head_cache);
            try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(j), @intCast(j + 1)));
        };
        if (!m.mtp) try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(n - 1), @intCast(n)));
        const ids = try sampling.rows(&m.kernels, &p.scope, try p.scope.slice(p.logits, 0, @intCast(n - 1), @intCast(n)), &.{@intCast(off + n)}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        if (dump) |file| {
            const z = try a.dupeSentinel(u8, file, 0);
            defer a.free(z);
            const f = try p.scope.cast(p.logits, mx.f32t);
            try mx.eval(f);
            try mx.check(mx.c.mlx_save(z, f));
        }
        try m.commit(&p, n);
        off += n;
    }
    const prefill = @as(f64, @floatFromInt(timer.read())) / 1e9;
    std.debug.print("Prefill {d} tokens in {d:.3}s\n", .{ tokens.items.len, prefill });
    timer.reset();
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);
    var history: std.ArrayList(i32) = .empty;
    defer history.deinit(a);
    var accepted: usize = 0;
    var rounds: usize = 0;
    if (max_tokens > 0) try generated.append(a, @intCast(pending));
    while (generated.items.len < max_tokens and !M.eos(pending)) {
        var window: [16]i32 = undefined;
        window[0] = pending;
        var n: usize = 1;
        if (m.mtp) {
            history.clearRetainingCapacity();
            try history.appendSlice(a, tokens.items);
            for (generated.items) |id| try history.append(a, @intCast(id));
            const budget = @min(drafts, max_tokens - generated.items.len);
            const copy = @import("copy.zig").propose(history.items, budget);
            if (copy.len == budget and budget > 0) {
                @memcpy(window[1..][0..budget], copy.tokens[0..budget]);
                n += budget;
            } else {
                var scope = mx.Scope{};
                defer scope.deinit();
                var dc = try head_cache.clone();
                defer dc.deinit();
                var dh = last;
                for (0..budget) |j| {
                    dh = try m.draftStep(&scope, dh, window[j], &dc);
                    const ids = try sampling.rows(&m.kernels, &scope, try m.draftHead(&scope, dh), &.{m.position + @as(i32, @intCast(j)) + 2}, settings);
                    defer mx.allocator.free(ids);
                    window[n] = ids[0];
                    n += 1;
                    if (M.eos(ids[0])) break;
                }
            }
        }
        var p = try m.forward(window[0..n]);
        defer p.deinit();
        var positions: [16]i32 = undefined;
        for (0..n) |j| positions[j] = m.position + @as(i32, @intCast(j)) + 1;
        const ids = try sampling.rows(&m.kernels, &p.scope, p.logits, positions[0..n], settings);
        defer mx.allocator.free(ids);
        var keep: usize = 1;
        var stop = false;
        while (keep < n and ids[keep - 1] == window[keep]) {
            try generated.append(a, @intCast(window[keep]));
            accepted += 1;
            keep += 1;
            if (M.eos(window[keep - 1]) or generated.items.len == max_tokens) {
                stop = true;
                break;
            }
        }
        if (!stop) {
            pending = ids[keep - 1];
            try generated.append(a, @intCast(pending));
        }
        if (m.mtp) for (0..keep) |j| {
            _ = try m.draftStep(&p.scope, last, window[j], &head_cache);
            try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(j), @intCast(j + 1)));
        };
        if (!m.mtp) try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(keep - 1), @intCast(keep)));
        try m.commit(&p, keep);
        rounds += 1;
        if (stop) break;
    }
    const seconds = @as(f64, @floatFromInt(timer.read())) / 1e9;
    const text = try tok.decode(a, generated.items, false);
    defer a.free(text);
    var buffer: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buffer);
    try out.interface.writeAll(text);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(generated.items), &digest, .{});
    std.debug.print("Generated {d} tokens in {d:.3}s ({d:.2} tok/s), {d} rounds, {d} accepted drafts\nSHA-256: {s}\n", .{ generated.items.len, seconds, @as(f64, @floatFromInt(generated.items.len)) / seconds, rounds, accepted, std.fmt.bytesToHex(digest, .lower) });
    if (report) |file| {
        const bytes = try std.json.Stringify.valueAlloc(a, .{ .prompt_tokens = tokens.items, .tokens = generated.items, .text = text, .seed = settings.seed, .temperature = settings.temperature, .top_k = settings.top_k, .top_p = settings.top_p, .metal_sampling = settings.metal, .prefill_seconds = prefill, .decode_seconds = seconds, .rounds = rounds, .accepted_drafts = accepted, .token_sha256 = std.fmt.bytesToHex(digest, .lower) }, .{});
        defer a.free(bytes);
        const f = try std.Io.Dir.cwd().createFile(io, file, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, bytes);
    }
}
pub fn check(comptime M: type, m: *M) !void {
    var prefix: [33]i32 = undefined;
    for (&prefix, 0..) |*v, i| v.* = @intCast(1000 + i * 37);
    var serial: [5]mx.Array = undefined;
    var cached: @TypeOf(m.cache) = @splat(.{});
    defer for (&cached) |*c| c.deinit();
    var scope = mx.Scope{};
    defer scope.deinit();
    for (0..2) |run_id| {
        m.reset();
        var offset: usize = 0;
        while (offset < prefix.len) {
            const count = @min(16, prefix.len - offset);
            var p = try m.forward(prefix[offset..][0..count]);
            defer p.deinit();
            try m.commit(&p, count);
            offset += count;
        }
        if (run_id == 0) {
            for ([_]i32{ 23, 41, 59, 83, 97 }, 0..) |token, j| {
                var p = try m.forward(&.{token});
                defer p.deinit();
                serial[j] = try scope.own(try mx.retain(p.logits));
                try m.commit(&p, 1);
            }
            for (m.cache, &cached) |c, *saved| saved.* = try c.clone();
        } else {
            var p = try m.forward(&.{ 23, 41, 59, 83, 97, 101, 103, 107 });
            defer p.deinit();
            for (0..5) |j| try equal(&scope, serial[j], try p.scope.slice(p.logits, 0, @intCast(j), @intCast(j + 1)));
            try m.commit(&p, 4);
            var next = try m.forward(&.{97});
            defer next.deinit();
            try equal(&scope, serial[4], next.logits);
            try m.commit(&next, 1);
            for (m.cache, cached) |actual, expected| {
                inline for (.{ "a", "b", "raw", "pooled", "ple" }) |field| {
                    if (@hasField(@TypeOf(actual), field)) {
                        const x = @field(actual, field);
                        const y = @field(expected, field);
                        if ((x.ctx == null) != (y.ctx == null)) return error.CachePresenceMismatch;
                        if (x.ctx != null) try @import("sampling_checks.zig").equal(&scope, x, y);
                    }
                }
                if (@hasField(@TypeOf(actual), "offset")) {
                    if (actual.offset != expected.offset or !std.mem.eql(i32, &actual.history, &expected.history)) return error.CacheMetadataMismatch;
                }
            }
        }
    }
    std.debug.print("PASS: every verified row, partial-commit continuation and all {d} layer caches match serial bit for bit.\n", .{m.cache.len});
}
fn equal(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    const x = try s.cast(a, mx.f32t);
    const y = try s.cast(b, mx.f32t);
    try mx.evalMany(&.{ x, y }, false);
    const n = mx.c.mlx_array_size(x);
    if (n != mx.c.mlx_array_size(y) or !std.mem.eql(u8, std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(x)[0..n]), std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(y)[0..n]))) return error.ExactnessMismatch;
}
