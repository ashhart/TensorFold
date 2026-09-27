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
    var trace_dir: ?[]const u8 = null;
    var trace_gdn: ?usize = null;
    var exact = false;
    var cache_stress = false;
    var long_cache = false;
    var warm = false;
    var copy_enabled = true;
    var reduced_vocab = true;
    var queued_drafts = true;
    var early_mtp = true;
    var check_mtp_state = false;
    var adaptive_drafts = true;
    var gpu_handoff = true;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const key = args[i];
        if (std.mem.eql(u8, key, "--no-gpu-handoff")) {
            gpu_handoff = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--fixed-drafts")) {
            adaptive_drafts = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-mtp-state")) {
            check_mtp_state = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--no-early-mtp")) {
            early_mtp = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--full-draft-vocab")) {
            reduced_vocab = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--no-queued-drafts")) {
            queued_drafts = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-long-cache")) {
            long_cache = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--no-copy")) {
            copy_enabled = false;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-cache-stress")) {
            cache_stress = true;
            continue;
        }
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
        } else if (std.mem.eql(u8, key, "--report")) report = val else if (std.mem.eql(u8, key, "--dump-logits")) dump = val else if (std.mem.eql(u8, key, "--trace-dir")) trace_dir = val else if (std.mem.eql(u8, key, "--trace-gdn")) trace_gdn = try std.fmt.parseInt(usize, val, 10) else return error.UnknownArgument;
        i += 1;
    }
    if (drafts > 15) return error.InvalidDraftBudget;
    if (check_mtp_state and drafts == 0) return error.InvalidDraftBudget;
    try settings.validate();
    try mx.init();
    defer mx.shutdown();
    var m = try M.init(io, args[2], drafts > 0 and !exact and !cache_stress and !long_cache);
    defer m.deinit();
    if (trace_dir != null and !@hasField(M, "trace_dir")) return error.UnsupportedTrace;
    if (trace_gdn) |layer_index| {
        if (trace_dir == null or layer_index >= 48 or layer_index % 4 == 3) return error.InvalidTraceLayer;
        if (@hasField(M, "trace_gdn")) m.trace_gdn = layer_index else return error.UnsupportedTrace;
    }
    if (long_cache) return @import("cache_checks.zig").checkLong(M, &m);
    if (cache_stress) return @import("cache_checks.zig").check(M, &m);
    if (exact) {
        try check(M, &m);
        return;
    }
    if (m.mtp and reduced_vocab) try @import("draft_vocab.zig").install(&m.weights, M.draft_vocabulary, M.vocab, 8);
    const draft_ids = m.weights.arrays.get("draft_ids");
    if (check_mtp_state) return @import("mtp_checks.zig").check(M, &m);
    var depth = try @import("draft_depth.zig").Adaptive.init(drafts, M.draft_prior);
    var calibration_seconds: f64 = 0;
    if (adaptive_drafts and m.mtp) {
        const calibration_timer = Stopwatch.init(io);
        try @import("mtp_calibration.zig").measure(M, &m, io, &depth, settings);
        calibration_seconds = @as(f64, @floatFromInt(calibration_timer.read())) / 1e9;
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
        if (@hasField(M, "trace_dir")) m.trace_dir = if (trace_gdn != null or off + n == tokens.items.len) trace_dir else null;
        var p = try m.forward(tokens.items[off..][0..n]);
        if (@hasField(M, "trace_dir")) m.trace_dir = null;
        defer p.deinit();
        if (m.mtp) for (0..n) |j| {
            if (last.ctx != null) _ = try m.draftStep(&p.scope, last, tokens.items[off + j], &head_cache);
            try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(j), @intCast(j + 1)));
        };
        if (!m.mtp) try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(n - 1), @intCast(n)));
        const ids = try sampling.rows(&m.kernels, &p.scope, try p.scope.slice(p.logits, 0, @intCast(n - 1), @intCast(n)), &.{@intCast(off + n)}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        if (dump) |file| if (off + n == tokens.items.len) {
            const z = try a.dupeSentinel(u8, file, 0);
            defer a.free(z);
            const f = try p.scope.cast(p.logits, mx.f32t);
            try mx.eval(f);
            try mx.check(mx.c.mlx_save(z, f));
        };
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
    var handoff_rounds: usize = 0;
    var depth_counts: [16]usize = @splat(0);
    var proposal_hash = std.crypto.hash.sha2.Sha256.init(.{});
    const Pipeline = @import("mtp_pipeline.zig").Pipeline(M);
    var pipeline = Pipeline{};
    defer pipeline.deinit();
    const early = early_mtp and m.mtp and settings.metal;
    if (early and max_tokens > 1 and !M.eos(pending)) {
        var s = mx.Scope{};
        defer s.deinit();
        pipeline = try Pipeline.prepare(&m, &s, head_cache, last, pending, m.position, settings);
    }
    if (max_tokens > 0) try generated.append(a, @intCast(pending));
    while (generated.items.len < max_tokens and !M.eos(pending)) {
        const round_timer = Stopwatch.init(io);
        var neural_proposed: usize = 0;
        var stage: []const u8 = "draft proposals";
        var proposal: usize = 0;
        errdefer std.debug.print("Decode failed at round {d}, position {d}, stage {s}, proposal {d}\n", .{ rounds, m.position, stage, proposal });
        var window: [16]i32 = undefined;
        window[0] = pending;
        var n: usize = 1;
        var proposal_scope = mx.Scope{};
        defer proposal_scope.deinit();
        var gpu_chain: ?mx.Array = null;
        const handoff = gpu_handoff and queued_drafts and settings.metal and @hasDecl(M, "forwardArray");
        if (m.mtp) {
            history.clearRetainingCapacity();
            try history.appendSlice(a, tokens.items);
            for (generated.items) |id| try history.append(a, @intCast(id));
            const remaining = max_tokens - generated.items.len;
            // Python chooses the initial depth before committing the first token.
            const budget = if (adaptive_drafts) depth.choose(remaining + @intFromBool(rounds == 0)) else @min(drafts, remaining);
            const copy = @import("copy.zig").propose(history.items, budget);
            if (copy_enabled and copy.len == budget and budget > 0) {
                @memcpy(window[1..][0..budget], copy.tokens[0..budget]);
                n += budget;
            } else if (early) {
                const chain = try pipeline.propose(&m, &proposal_scope, budget, m.position, settings, queued_drafts);
                if (handoff) {
                    gpu_chain = chain;
                    n += budget;
                } else {
                    try mx.eval(chain);
                    n += try @import("acceptance.zig").copyChain(window[n..], mx.c.mlx_array_data_uint32(chain)[0..budget], M.eos);
                }
                neural_proposed = n - 1;
            } else {
                const scope = &proposal_scope;
                var dc = try head_cache.clone();
                defer dc.deinit();
                var dh = last;
                if (queued_drafts and settings.metal) {
                    var proposed: [15]mx.Array = undefined;
                    var token = try scope.ints(&.{pending});
                    for (0..budget) |j| {
                        proposal = j;
                        dh = try m.draftStepArray(scope, dh, token, &dc, true);
                        token = try @import("gpu_sampling.zig").sample(&m.kernels, scope, try m.draftHead(scope, dh), &.{m.position + @as(i32, @intCast(j)) + 1}, settings, draft_ids);
                        proposed[j] = token;
                    }
                    const chain = try scope.cat(proposed[0..budget], 0);
                    if (handoff) {
                        gpu_chain = chain;
                        n += budget;
                    } else {
                        try mx.eval(chain);
                        n += try @import("acceptance.zig").copyChain(window[n..], mx.c.mlx_array_data_uint32(chain)[0..budget], M.eos);
                    }
                } else for (0..budget) |j| {
                    proposal = j;
                    dh = try m.draftStep(scope, dh, window[j], &dc);
                    const ids = try sampling.rowsMapped(&m.kernels, scope, try m.draftHead(scope, dh), &.{m.position + @as(i32, @intCast(j)) + 1}, settings, draft_ids);
                    defer mx.allocator.free(ids);
                    window[n] = ids[0];
                    n += 1;
                    if (M.eos(ids[0])) break;
                }
                neural_proposed = n - 1;
            }
        }
        stage = "target verification";
        var p = if (@hasDecl(M, "forwardArray")) blk: {
            if (gpu_chain) |chain| {
                const first = try proposal_scope.cast(try proposal_scope.ints(&.{pending}), mx.c.MLX_UINT32);
                break :blk try m.forwardArray(try proposal_scope.cat(&.{ first, chain }, 0));
            }
            break :blk try m.forwardQueued(window[0..n]);
        } else try m.forwardQueued(window[0..n]);
        defer p.deinit();
        var positions: [16]i32 = undefined;
        for (0..n) |j| positions[j] = m.position + @as(i32, @intCast(j)) + 1;
        var speculation: ?Pipeline.Speculation = null;
        defer if (speculation) |*spec| spec.deinit();
        const ids = if (early or gpu_chain != null) blk: {
            const target = try @import("gpu_sampling.zig").sample(&m.kernels, &p.scope, p.logits, positions[0..n], settings, null);
            var draws: [3]mx.Array = undefined;
            draws[0] = target;
            var draw_count: usize = 1;
            if (early) {
                speculation = try pipeline.speculate(&m, &p.scope, p.hidden, target, m.position, settings);
                draws[draw_count] = speculation.?.firsts;
                draw_count += 1;
            }
            if (gpu_chain) |chain| {
                draws[draw_count] = chain;
                draw_count += 1;
            }
            // Target, next drafts and incoming proposals share one host read.
            const values = try p.scope.cat(draws[0..draw_count], 0);
            try mx.eval(values);
            const out_ids = try mx.allocator.alloc(i32, n);
            errdefer mx.allocator.free(out_ids);
            for (out_ids, 0..) |*id, j| id.* = @intCast(mx.c.mlx_array_data_uint32(values)[j]);
            if (gpu_chain != null) {
                const physical_rows = n;
                const start = physical_rows * (1 + @as(usize, @intFromBool(early)));
                // The GPU can evaluate beyond a proposed EOS. Acceptance and cache
                // commit see only its logical prefix, exactly as in host handoff.
                n = 1 + try @import("acceptance.zig").copyChain(window[1..], mx.c.mlx_array_data_uint32(values)[start..][0 .. physical_rows - 1], M.eos);
                neural_proposed = n - 1;
                handoff_rounds += 1;
            }
            break :blk out_ids;
        } else try sampling.rows(&m.kernels, &p.scope, p.logits, positions[0..n], settings);
        defer mx.allocator.free(ids);
        const width = [_]u8{@intCast(n)};
        proposal_hash.update(&width);
        proposal_hash.update(std.mem.sliceAsBytes(window[0..n]));
        var parents: [16]i32 = undefined;
        for (0..n) |j| parents[j] = @as(i32, @intCast(j)) - 1;
        const result = try @import("acceptance.zig").select(window[0..n], parents[0..n], ids[0..n], max_tokens - generated.items.len, M.eos);
        const keep = result.kept;
        try generated.appendSlice(a, result.tokens[0..result.count]);
        accepted += result.accepted;
        pending = result.pending;
        stage = "draft cache commit";
        if (early) {
            if (!result.stop) try pipeline.settle(&p.scope, speculation.?, keep);
            try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(keep - 1), @intCast(keep)));
        } else if (m.mtp) for (0..keep) |j| {
            _ = try m.draftStep(&p.scope, last, window[j], &head_cache);
            try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(j), @intCast(j + 1)));
        };
        if (!m.mtp) try mx.replace(&last, try p.scope.slice(p.hidden, 0, @intCast(keep - 1), @intCast(keep)));
        stage = "target cache commit";
        try m.commit(&p, keep);
        if (neural_proposed > 0) {
            depth_counts[neural_proposed] += 1;
            if (adaptive_drafts) try depth.observe(neural_proposed, result.accepted, @as(f64, @floatFromInt(round_timer.read())) / 1e6);
        }
        rounds += 1;
        if (result.stop) break;
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
    var proposal_digest: [32]u8 = undefined;
    proposal_hash.final(&proposal_digest);
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(generated.items), &digest, .{});
    std.debug.print("Generated {d} tokens in {d:.3}s ({d:.2} tok/s), {d} rounds, {d} accepted drafts\nSHA-256: {s}\n", .{ generated.items.len, seconds, @as(f64, @floatFromInt(generated.items.len)) / seconds, rounds, accepted, std.fmt.bytesToHex(digest, .lower) });
    if (report) |file| {
        var peak: usize = 0;
        var active: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        try mx.check(mx.c.mlx_get_active_memory(&active));
        const bytes = try std.json.Stringify.valueAlloc(a, .{
            .prompt_tokens = tokens.items,
            .tokens = generated.items,
            .text = text,
            .seed = settings.seed,
            .temperature = settings.temperature,
            .top_k = settings.top_k,
            .top_p = settings.top_p,
            .metal_sampling = settings.metal,
            .context_copy = copy_enabled,
            .draft_vocab_size = if (draft_ids) |ids| mx.c.mlx_array_size(ids) else @as(usize, M.vocab),
            .queued_drafts = m.mtp and queued_drafts and settings.metal,
            .early_mtp = early,
            .gpu_handoff_rounds = handoff_rounds,
            .adaptive_drafts = adaptive_drafts and m.mtp,
            .draft_depth_counts = depth_counts,
            .calibration_seconds = calibration_seconds,
            .proposal_sha256 = std.fmt.bytesToHex(proposal_digest, .lower),
            .prefill_seconds = prefill,
            .decode_seconds = seconds,
            .rounds = rounds,
            .accepted_drafts = accepted,
            .peak_mlx_bytes = peak,
            .active_mlx_bytes = active,
            .token_sha256 = std.fmt.bytesToHex(digest, .lower),
        }, .{});
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
