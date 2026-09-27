const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const tokenizer = @import("vendor/tokenizer.zig");
const Stopwatch = @import("vendor/io_util.zig").Stopwatch;
const Draft = @import("drafter.zig").Drafter;
const sampling = @import("sampling.zig");
const lanes = @import("lanes.zig");
fn eos(id: i32) bool {
    return id == 248044 or id == 248046;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-sampling")) return @import("sampling_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-vocab")) return @import("draft_vocab_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-sparse")) return @import("flash.zig").Model.checkAttention(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-attention")) return @import("attention_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-checkpoint-files")) return @import("safetensors.zig").checkFiles(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-ple")) return @import("ple_tables.zig").Tables.check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-ple_norm")) return @import("flash.zig").Model.checkPleNorm(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-variants")) return @import("variant_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-allocation-failures")) return @import("failure_checks.zig").check(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-model-schema")) return @import("schema.zig").checkCheckpoint(std.meta.stringToEnum(@import("schema.zig").Kind, args[2]) orelse return error.UnsupportedModel, io, args[3]);
    if (args.len < 3 or !std.mem.eql(u8, args[1], "run")) {
        std.debug.print("Nemotron/Flash MTP options: --full-draft-vocab, --no-queued-drafts\n", .{});
        std.debug.print("Usage: tensorfold run MODEL_DIR [--prompt TEXT] [--tokens ID,ID,...] [--max-tokens N]\n  [--drafter DIR] [--mtp-drafts N] [--no-drafts] [--no-copy] [--metal-simd] [--metal-sampling]\n  [--temperature T] [--seed N] [--top-k N] [--top-p P] [--warmup]\n  [--report PATH] [--dump-logits PATH] [--check-exact] [--check-cache-stress] [--check-long-cache]\n  [--trace-dir EXISTING_DIR (Flash only)]\n  tensorfold check-sampling|check-sparse|check-attention FIXTURE_DIR\n  tensorfold check-model-schema qwen|dflash|nemotron|flash MODEL_DIR\n", .{});
        return;
    }
    {
        var pathbuf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/config.json", .{args[2]}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
        defer cfg.deinit();
        if (cfg.value == .object) if (cfg.value.object.get("model_type")) |kind| {
            if (kind == .string and std.mem.eql(u8, kind.string, "nemotron_h")) return @import("family_runtime.zig").run(@import("nemotron.zig").Model, init, args);
            if (kind == .string and std.mem.eql(u8, kind.string, "qwen4_exp")) return @import("family_runtime.zig").run(@import("flash.zig").Model, init, args);
        };
    }
    var prompt: []const u8 = "Write a short Python function that computes the Fibonacci sequence.";
    var max_tokens: usize = 32;
    var token_list: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var settings = sampling.Sampling{};
    var explicit_seed = false;
    var exact = false;
    var cache_stress = false;
    var long_cache = false;
    var warmup = false;
    var copy_enabled = true;
    var report: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--check-long-cache")) {
            long_cache = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--no-copy")) {
            copy_enabled = false;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--check-cache-stress")) {
            cache_stress = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--metal-sampling")) {
            settings.metal = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--metal-simd")) {
            mx.force_simd = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--check-exact")) {
            exact = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--warmup")) {
            warmup = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--report")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            report = args[i + 1];
            i += 1;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        if (std.mem.eql(u8, args[i], "--seed")) explicit_seed = true;
        if (std.mem.eql(u8, args[i], "--prompt")) prompt = args[i + 1] else if (std.mem.eql(u8, args[i], "--max-tokens")) max_tokens = try std.fmt.parseInt(usize, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--tokens")) token_list = args[i + 1] else if (std.mem.eql(u8, args[i], "--dump-logits")) dump = args[i + 1] else if (std.mem.eql(u8, args[i], "--drafter")) draft_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--temperature")) settings.temperature = try std.fmt.parseFloat(f64, args[i + 1]) else if (std.mem.eql(u8, args[i], "--seed")) settings.seed = try std.fmt.parseInt(u64, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--top-k")) settings.top_k = try std.fmt.parseInt(usize, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--top-p")) settings.top_p = try std.fmt.parseFloat(f64, args[i + 1]) else return error.UnknownArgument;
        i += 1;
    }
    try settings.validate();
    try mx.init();
    defer mx.shutdown();
    var timer = Stopwatch.init(io);
    var m = try model.Model.init(io, args[2]);
    defer m.deinit();
    if (long_cache) return @import("cache_checks.zig").checkLong(model.Model, &m);
    if (cache_stress) return @import("cache_checks.zig").check(model.Model, &m);
    if (exact) {
        try @import("verification.zig").check(&m);
        return;
    }
    var draft: ?Draft = if (draft_dir) |path| try Draft.init(io, path, &m) else null;
    defer if (draft) |*d| d.deinit();
    if (warmup) {
        std.debug.print("Warming Metal variants...\n", .{});
        for ([_]usize{ 1, 16, 32 }) |n| {
            const fake: [32]i32 = @splat(42);
            var parents: [32]i32 = undefined;
            for (0..n) |j| parents[j] = if (j == 0) -1 else @intCast((j - 1) / 2);
            var p = try m.forward(fake[0..n], parents[0..n]);
            defer p.deinit();
            try m.commit(&p, &.{0});
            if (draft) |*d| {
                try d.absorb(&m, &p, &.{0});
                _ = try d.propose(&m, 42, 15, settings);
            }
        }
        m.reset();
        if (draft) |*d| d.reset();
    }
    // The borrowed tokenizer accepts absolute paths.
    const dir = try std.Io.Dir.cwd().realPathFileAlloc(io, args[2], allocator);
    defer allocator.free(dir);
    var tok = try tokenizer.loadTokenizer(io, allocator, dir);
    defer tok.deinit();
    var tokens: std.ArrayList(i32) = .empty;
    defer tokens.deinit(allocator);
    if (token_list) |list| {
        var parts = std.mem.splitScalar(u8, list, ',');
        while (parts.next()) |part| try tokens.append(allocator, try std.fmt.parseInt(i32, part, 10));
    } else {
        const ids = try tok.encode(allocator, prompt);
        defer allocator.free(ids);
        for (ids) |id| try tokens.append(allocator, @intCast(id));
    }
    if (tokens.items.len == 0) return error.EmptyPrompt;
    if (tokens.items.len > 262144 or max_tokens > 262144 - tokens.items.len) return error.ContextLimitExceeded;
    for (tokens.items) |id| if (id < 0 or id >= 248320) return error.InvalidToken;
    if (!explicit_seed) settings.seed = sampling.seedFor(tokens.items);
    std.debug.print("Loaded target in {d:.2}s; prompt {d} tokens\n", .{ @as(f64, @floatFromInt(timer.read())) / 1e9, tokens.items.len });
    timer.reset();
    var pending: i32 = 0;
    var off: usize = 0;
    while (off < tokens.items.len) {
        const n = @min(128, tokens.items.len - off);
        var parents: [128]i32 = undefined;
        var rows: [128]i32 = undefined;
        for (0..n) |j| {
            parents[j] = @as(i32, @intCast(j)) - 1;
            rows[j] = @intCast(j);
        }
        var p = try m.forward(tokens.items[off..][0..n], parents[0..n]);
        defer p.deinit();
        var positions: [128]i32 = undefined;
        for (0..n) |j| positions[j] = m.position + @as(i32, @intCast(j)) + 1;
        const ids = try sampling.rows(&m.kernels, &p.scope, p.logits, positions[0..n], settings);
        defer mx.allocator.free(ids);
        pending = ids[n - 1];
        if (dump) |path| if (off + n == tokens.items.len) {
            const z = try allocator.dupeSentinel(u8, path, 0);
            defer allocator.free(z);
            const f = try p.scope.cast(p.logits, mx.f32t);
            try mx.eval(f);
            try mx.check(mx.c.mlx_save(z, f));
        };
        try m.commit(&p, rows[0..n]);
        if (draft) |*d| try d.absorb(&m, &p, rows[0..n]);
        off += n;
    }
    const prefill_seconds = @as(f64, @floatFromInt(timer.read())) / 1e9;
    std.debug.print("Prefill {d:.2}s\n", .{prefill_seconds});
    timer.reset();
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(allocator);
    var history: std.ArrayList(i32) = .empty;
    defer history.deinit(allocator);
    try history.appendSlice(allocator, tokens.items);
    var rounds: usize = 0;
    var accepted: usize = 0;
    var draft_ns: u64 = 0;
    var forward_ns: u64 = 0;
    var commit_ns: u64 = 0;
    if (max_tokens > 0) try generated.append(allocator, @intCast(pending));
    while (generated.items.len < max_tokens and pending != 248044 and pending != 248046) {
        var stage = Stopwatch.init(io);
        history.shrinkRetainingCapacity(tokens.items.len);
        for (generated.items) |id| try history.append(allocator, @intCast(id));
        const copy = if (draft != null and copy_enabled) @import("copy.zig").propose(history.items, max_tokens - generated.items.len) else @import("drafter.zig").Proposal{};
        const proposal = if (copy.len >= @min(15, max_tokens - generated.items.len)) copy else if (draft) |*d| try d.propose(&m, pending, @min(15, max_tokens - generated.items.len), settings) else @import("drafter.zig").Proposal{};
        draft_ns += stage.read();
        stage.reset();
        var window: [32]i32 = undefined;
        var parents: [32]i32 = undefined;
        window[0] = pending;
        parents[0] = -1;
        for (0..proposal.len) |j| {
            window[j + 1] = proposal.tokens[j];
            parents[j + 1] = proposal.parents[j] + 1;
        }
        const n = proposal.len + 1;
        const tree = try lanes.Tree.init(parents[0..n]);
        var p = try m.forward(window[0..n], parents[0..n]);
        defer p.deinit();
        var positions: [32]i32 = undefined;
        for (0..n) |j| positions[j] = m.position + tree.depths[j] + 1;
        const ids = try sampling.rows(&m.kernels, &p.scope, p.logits, positions[0..n], settings);
        forward_ns += stage.read();
        stage.reset();
        defer mx.allocator.free(ids);
        const result = try @import("acceptance.zig").select(window[0..n], parents[0..n], ids, max_tokens - generated.items.len, eos);
        try generated.appendSlice(allocator, result.tokens[0..result.count]);
        accepted += result.accepted;
        pending = result.pending;
        try m.commit(&p, result.path[0..result.kept]);
        if (draft) |*d| try d.absorb(&m, &p, result.path[0..result.kept]);
        commit_ns += stage.read();
        rounds += 1;
        if (result.stop) break;
    }
    const seconds = @as(f64, @floatFromInt(timer.read())) / 1e9;
    const text = try tok.decode(allocator, generated.items, false);
    defer allocator.free(text);
    var outbuf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &outbuf);
    try out.interface.writeAll(text);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(generated.items), &digest, .{});
    std.debug.print("Token SHA-256: {s}\n", .{std.fmt.bytesToHex(digest, .lower)});
    std.debug.print("Stage totals: draft {d:.3}s, target+sample {d:.3}s, commit+absorb {d:.3}s\n", .{ @as(f64, @floatFromInt(draft_ns)) / 1e9, @as(f64, @floatFromInt(forward_ns)) / 1e9, @as(f64, @floatFromInt(commit_ns)) / 1e9 });
    std.debug.print("Generated {d} tokens in {d:.3}s ({d:.2} tok/s), {d} rounds, {d} accepted drafts\nIDs: {any}\n", .{ generated.items.len, seconds, @as(f64, @floatFromInt(generated.items.len)) / seconds, rounds, accepted, generated.items });
    if (report) |path| {
        var peak: usize = 0;
        var active: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        try mx.check(mx.c.mlx_get_active_memory(&active));
        const content = try std.json.Stringify.valueAlloc(allocator, .{ .prompt_tokens = tokens.items, .tokens = generated.items, .text = text, .seed = settings.seed, .temperature = settings.temperature, .top_k = settings.top_k, .top_p = settings.top_p, .metal_sampling = settings.metal, .context_copy = copy_enabled, .prefill_seconds = prefill_seconds, .decode_seconds = seconds, .rounds = rounds, .accepted_drafts = accepted, .warmed = warmup, .peak_mlx_bytes = peak, .active_mlx_bytes = active, .token_sha256 = std.fmt.bytesToHex(digest, .lower) }, .{});
        defer allocator.free(content);
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, content);
    }
}

test {
    _ = @import("draft_vocab.zig");
    _ = @import("lanes.zig");
    _ = @import("sampling.zig");
    _ = @import("config.zig");
    _ = @import("copy.zig");
    _ = @import("ngram.zig");
    _ = @import("acceptance.zig");
    _ = @import("safetensors.zig");
    _ = @import("ple_tables.zig");
    _ = @import("schema.zig");
}
