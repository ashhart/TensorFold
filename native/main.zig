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
    @import("bonsai.zig").memory_limit = init.environ_map.get("TENSORFOLD_MEMORY_LIMIT_GB");
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-flash-checkpoint")) return @import("flash_names.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-flash-weights")) return @import("flash_ops.zig").checkWeights(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-bonsai-pack")) return @import("bonsai.zig").check(io, args[2], args[3]);
    if (args.len == 2 and std.mem.eql(u8, args[1], "check-request-state")) return @import("request_state_checks.zig").check();
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-session-images")) return @import("session_checks.zig").checkImages(io, args[2], args[3]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-session-rounds")) return @import("session_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-memory-budget")) return @import("memory_budget.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-prompt-cache")) return @import("prompt_cache.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-prefill-plan")) return @import("prefill_plan.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-memory-runtime")) return @import("memory_runtime.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-allocation")) return @import("draft_allocation.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-capture")) return @import("draft_capture.zig").check(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "fit-draft-calibration")) return @import("draft_calibration.zig").fitFile(io, args[2], args[3]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-calibration")) return @import("draft_calibration.zig").check(io, args[2]);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "convert-drafter")) return @import("convert_drafter.zig").run(io, args[2..]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-drafter-conversion")) return @import("convert_drafter.zig").check(io, args[2]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "serve")) return @import("server.zig").run(init, args);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-chat")) return @import("chat.zig").check(io, args[2], args[3]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-tool-calls")) return @import("tool_calls.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-tool-stream")) return @import("tool_stream.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-image-http")) return @import("image_http.zig").check(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-image-url")) return @import("image_http.zig").fetchCheck(io, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-gemma-prefill")) return @import("gemma_prefill.zig").check(io, args[2], args[3]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "check-gemma-draft")) return @import("gemma.zig").checkDraft(io, args[2], args[3], args[4]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "check-dflash")) return @import("dflash.zig").check(io, args[2], args[3], try std.fmt.parseInt(usize, args[4], 10));
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-dspark")) return @import("deepseek.zig").checkDspark(io, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-deepseek-model")) return @import("deepseek.zig").checkModel(io, args[2], args[3]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-deepseek-dense")) return @import("deepseek_dense_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-runtime")) return @import("runtime_checks.zig").check(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-glm-model")) return @import("glm.zig").checkModel(io, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-gemma-model")) return @import("gemma.zig").checkModel(io, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-vision")) return @import("vision.zig").check(io, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-image")) return @import("vision.zig").checkImage(io, args[2], args[3]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-prefill-math")) return @import("prefill_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-sampling")) return @import("sampling_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-vocab")) return @import("draft_vocab_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-draft-depth")) return @import("draft_depth.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-mtp-positions")) return @import("mtp_position_checks.zig").check(io, args[2]);
    if (args.len == 2 and std.mem.eql(u8, args[1], "check-serial-pipeline")) return @import("serial_pipeline_checks.zig").check();
    if (args.len == 2 and std.mem.eql(u8, args[1], "check-kv-buffer")) return @import("kv_buffer_checks.zig").check();
    if (args.len == 2 and std.mem.eql(u8, args[1], "check-ngram-gpu")) return @import("ngram.zig").checkGpu();
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-ple-resident")) return @import("ple_tables.zig").Tables.checkResident(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-sparse")) return @import("flash.zig").Model.checkAttention(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-attention")) return @import("attention_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-checkpoint-files")) return @import("safetensors.zig").checkFiles(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-ple")) return @import("ple_tables.zig").Tables.check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-ple_norm")) return @import("flash.zig").Model.checkPleNorm(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-variants")) return @import("variant_checks.zig").check(io, args[2]);
    if (args.len == 3 and std.mem.eql(u8, args[1], "check-allocation-failures")) return @import("failure_checks.zig").check(io, args[2]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "check-model-schema")) return @import("schema.zig").checkCheckpoint(std.meta.stringToEnum(@import("schema.zig").Kind, args[2]) orelse return error.UnsupportedModel, io, args[3]);
    if (args.len < 3 or !std.mem.eql(u8, args[1], "run")) {
        std.debug.print("Convert DeepSeek drafters: tensorfold convert-drafter mtp SHARD... OUTDIR [--layer 0]\n  tensorfold convert-drafter dspark SHARD... OUTDIR\n", .{});
        std.debug.print("DFlash2 calibration: built-in upstream tables; --draft-calibration FILE overrides them.\n  tensorfold fit-draft-calibration SAMPLES_JSON OUTPUT_JSON\n", .{});
        std.debug.print("HTTP: tensorfold serve MODEL_DIR [--host 127.0.0.1] [--port 8080] [--served-model-name NAME]\n  [--temperature T] [--top-k K] [--top-p P] [--max-tokens 4096]\n  [--thinking | --no-thinking] [--reasoning-effort medium] [--thinking-budget 0]\n  [--request-timeout-seconds 0] [--shutdown-grace-seconds 5]\n  [--vision-urls] Raw/chat completions with SSE; Qwen accepts data URLs and opt-in public HTTPS images.\n", .{});
        std.debug.print("Qwen images: --image LOCAL_FILE (up to four); optional explicit <|vision_start|><|image_pad|><|vision_end|> markers in --prompt.\n", .{});
        std.debug.print("Text families: Qwen/Bonsai, Nemotron, Flash Next, Gemma, GLM, DeepSeek.\nGLM/DeepSeek MTP: --mtp-drafts 0..15 or --no-drafts. DeepSeek: --drafter DIR with model.safetensors and config.json (deepseek_v4_mtp or deepseek_v4_dspark). Gemma: --drafter DIR [--drafter-bits 8|4|0].\nNemotron/Flash MTP options: --full-draft-vocab, --no-queued-drafts, --no-early-mtp, --no-gpu-handoff, --fixed-drafts, --check-mtp-state\n", .{});
        std.debug.print("Flash resident PLE: --resident-ple [--no-ple-wiring], --check-ple-state [--check-long-cache]\nDiagnostics: tensorfold check-ngram-gpu; tensorfold check-ple-resident MODEL_DIR\n", .{});
        std.debug.print("Qwen prefill: regular 2048-token chunks; --lane-prefill selects the 128-row diagnostic.\nQwen/Flash traces: --trace-dir EXISTING_DIR. Math oracle: tensorfold check-prefill-math FIXTURE_DIR\n", .{});
        std.debug.print("Usage: tensorfold run MODEL_DIR [--prompt TEXT] [--tokens ID,ID,...] [--max-tokens N]\n  [--drafter DIR] [--mtp-drafts N] [--no-drafts] [--no-copy] [--metal-simd] [--metal-sampling]\n  [--no-serial-pipeline] [--check-serial-state] [--temperature T] [--seed N] [--top-k N] [--top-p P] [--warmup]\n  [--no-kv-buffers] [--check-kv-buffers] [--check-kv-reuse]\n  [--report PATH] [--dump-logits PATH] [--check-exact] [--check-cache-stress] [--check-long-cache]\n  [--trace-dir EXISTING_DIR (Qwen/Flash)]\n  tensorfold check-kv-buffer\n  tensorfold check-sampling|check-sparse|check-attention FIXTURE_DIR\n  tensorfold check-model-schema qwen|dflash|nemotron|flash|gemma|glm|deepseek MODEL_DIR\n", .{});
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
            if (kind == .string and std.mem.eql(u8, kind.string, "gemma4")) return @import("serial_runtime.zig").run(@import("gemma.zig").Model, init, args);
            if (kind == .string and std.mem.eql(u8, kind.string, "glm5_next")) return @import("serial_runtime.zig").run(@import("glm.zig").Model, init, args);
            if (kind == .string and std.mem.eql(u8, kind.string, "deepseek_v4")) return @import("serial_runtime.zig").run(@import("deepseek.zig").Model, init, args);
        };
    }
    var prompt: []const u8 = "Write a short Python function that computes the Fibonacci sequence.";
    var image_paths: std.ArrayList([]const u8) = .empty;
    defer image_paths.deinit(allocator);
    var max_tokens: usize = 32;
    var token_list: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var calibration_path: ?[]const u8 = null;
    var capture_dir: ?[]const u8 = init.environ_map.get("TF_DRAFT_CAPTURE");
    if (capture_dir != null and capture_dir.?.len == 0) capture_dir = null;
    var drafts_enabled = true;
    var settings = sampling.Sampling{};
    var explicit_seed = false;
    var exact = false;
    var cache_stress = false;
    var long_cache = false;
    var warmup = false;
    var lane_prefill = false;
    var trace_dir: ?[]const u8 = null;
    var copy_enabled = true;
    var serial_pipeline = true;
    var check_serial = false;
    var check_buffers = false;
    var check_reuse = false;
    var report: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--no-drafts")) {
            drafts_enabled = false;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--image")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            try image_paths.append(allocator, args[i + 1]);
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--lane-prefill")) {
            lane_prefill = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--check-kv-buffers")) {
            check_buffers = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--check-kv-reuse")) {
            check_reuse = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--no-kv-buffers")) {
            @import("kv_buffer.zig").enabled = false;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--check-serial-state")) {
            check_serial = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--no-serial-pipeline")) {
            serial_pipeline = false;
            continue;
        }
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
        if (std.mem.eql(u8, args[i], "--bonsai-form")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            @import("bonsai.zig").form_override = try @import("bonsai.zig").Form.parse(args[i + 1]);
            i += 1;
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
        if (std.mem.eql(u8, args[i], "--trace-dir")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            trace_dir = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--draft-calibration")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            calibration_path = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--draft-capture")) {
            if (i + 1 >= args.len) return error.MissingArgument;
            capture_dir = args[i + 1];
            i += 1;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        if (std.mem.eql(u8, args[i], "--seed")) explicit_seed = true;
        if (std.mem.eql(u8, args[i], "--prompt")) prompt = args[i + 1] else if (std.mem.eql(u8, args[i], "--max-tokens")) max_tokens = try std.fmt.parseInt(usize, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--tokens")) token_list = args[i + 1] else if (std.mem.eql(u8, args[i], "--dump-logits")) dump = args[i + 1] else if (std.mem.eql(u8, args[i], "--drafter")) draft_dir = args[i + 1] else if (std.mem.eql(u8, args[i], "--temperature")) settings.temperature = try std.fmt.parseFloat(f64, args[i + 1]) else if (std.mem.eql(u8, args[i], "--seed")) settings.seed = try std.fmt.parseInt(u64, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--top-k")) settings.top_k = try std.fmt.parseInt(usize, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--top-p")) settings.top_p = try std.fmt.parseFloat(f64, args[i + 1]) else return error.UnknownArgument;
        i += 1;
    }
    try settings.validate();
    const startup_timer = Stopwatch.init(io);
    try mx.init();
    defer mx.shutdown();
    var timer = Stopwatch.init(io);
    var m = try model.Model.init(io, args[2]);
    m.trace_dir = trace_dir;
    defer m.deinit();
    if (check_reuse) return @import("cache_checks.zig").checkBufferReuse(model.Model, &m);
    if (check_buffers) return @import("cache_checks.zig").checkBuffered(model.Model, &m, long_cache);
    if (check_serial) return @import("cache_checks.zig").checkSerial(model.Model, &m, long_cache);
    if (long_cache) return @import("cache_checks.zig").checkLong(model.Model, &m);
    if (cache_stress) return @import("cache_checks.zig").check(model.Model, &m);
    if (exact) {
        try @import("verification.zig").check(&m);
        return;
    }
    var draft: ?Draft = if (drafts_enabled) (if (draft_dir) |path| try Draft.init(io, path, &m) else null) else null;
    defer if (draft) |*d| d.deinit();
    if (calibration_path) |path| {
        if (draft) |*d| try d.loadCalibration(io, path) else return error.CalibrationRequiresDrafter;
    }
    const load_seconds = @as(f64, @floatFromInt(timer.read())) / 1e9;
    const warm_timer = Stopwatch.init(io);
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
                try d.absorb(&m, &p, &.{0}, fake[0..n]);
                _ = try d.propose(&m, 42, 15, settings);
            }
        }
        m.reset();
        if (draft) |*d| d.reset();
    }
    const warmup_seconds = if (warmup) @as(f64, @floatFromInt(warm_timer.read())) / 1e9 else 0;
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
    if (tokens.items.len > 262144) return error.ContextLimitExceeded;
    for (tokens.items) |id| if (id < 0 or id >= 248320) return error.InvalidToken;
    if (image_paths.items.len > 0 and lane_prefill) return error.ImageRequiresRegularPrefill;
    var image_prompt: ?@import("vision.zig").Prompt = if (image_paths.items.len > 0) try @import("vision.zig").Prompt.prepare(io, args[2], image_paths.items, &tokens, allocator, &m.weights) else null;
    defer if (image_prompt) |*p| p.deinit();
    if (tokens.items.len > 262144 or max_tokens > 262144 - tokens.items.len) return error.ContextLimitExceeded;
    for (tokens.items) |id| if (id < 0 or id >= 248320) return error.InvalidToken;
    if (!explicit_seed) settings.seed = sampling.seedFor(tokens.items);
    if (capture_dir) |folder| {
        if (draft) |*d| {
            d.capture = @import("draft_capture.zig").Writer.init(mx.allocator, io, folder, settings, 0, 5120) catch |err| blk: {
                std.debug.print("DFlash capture disabled: {s}\n", .{@errorName(err)});
                break :blk null;
            };
            if (d.capture) |writer| std.debug.print("DFlash capture: {s}\n", .{writer.base});
        } else return error.CaptureRequiresDrafter;
    }
    const startup_seconds = @as(f64, @floatFromInt(startup_timer.read())) / 1e9;
    std.debug.print("Loaded and prepared target in {d:.2}s; prompt {d} tokens\n", .{ startup_seconds, tokens.items.len });
    timer.reset();
    var pending: i32 = 0;
    var off: usize = 0;
    while (off < tokens.items.len) {
        const n = @min(if (lane_prefill) @as(usize, 128) else 2048, tokens.items.len - off);
        var parents: [2048]i32 = undefined;
        var rows: [2048]i32 = undefined;
        for (0..n) |j| {
            parents[j] = @as(i32, @intCast(j)) - 1;
            rows[j] = @intCast(j);
        }
        var image_scope = mx.Scope{};
        defer image_scope.deinit();
        var p = if (image_prompt) |*image| try m.prefillImage(tokens.items[off..][0..n], try image_scope.slice(image.embeddings, 1, @intCast(off), @intCast(off + n)), try image.positions.chunk(&image_scope, off, off + n), image.positions.delta) else if (lane_prefill) try m.forward(tokens.items[off..][0..n], parents[0..n]) else try m.prefill(tokens.items[off..][0..n]);
        defer p.deinit();
        const last_logits = try p.scope.slice(p.logits, 1, mx.dim(p.logits, 1) - 1, mx.dim(p.logits, 1));
        const ids = try sampling.rows(&m.kernels, &p.scope, last_logits, &.{m.position + @as(i32, @intCast(n))}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        if (dump) |path| if (off + n == tokens.items.len) {
            const z = try allocator.dupeSentinel(u8, path, 0);
            defer allocator.free(z);
            const f = try p.scope.cast(p.logits, mx.f32t);
            try mx.eval(f);
            try mx.check(mx.c.mlx_save(z, f));
        };
        try m.commit(&p, rows[0..n]);
        if (trace_dir != null) for (m.cache, 0..) |cache, layer| {
            try m.trace(&p.scope, p.start, layer, "cache0", cache.a);
            try m.trace(&p.scope, p.start, layer, "cache1", cache.b);
        };
        if (draft) |*d| {
            var begin: usize = 0;
            while (begin < n) {
                const end = @min(begin + 128, n);
                try d.absorb(&m, &p, rows[begin..end], tokens.items[off..][0..n]);
                begin = end;
            }
            if (off + n == tokens.items.len and max_tokens > 0) {
                if (mx.dim(p.logits, 1) == 1) {
                    d.captureTarget(&m, &p, &.{0}, &.{m.position});
                } else {
                    for (0..n) |j| parents[j] = p.start + @as(i32, @intCast(j)) + 1;
                    d.captureTarget(&m, &p, &.{@intCast(n - 1)}, parents[0..n]);
                }
            }
        }
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
    const use_serial_pipeline = serial_pipeline and settings.metal and draft == null;
    var queued_serial_steps: usize = 0;
    if (use_serial_pipeline) {
        const result = try @import("serial_pipeline.zig").generate(model.Model, &m, allocator, &generated, max_tokens, settings, eos, null);
        rounds = result.rounds;
        queued_serial_steps = result.queued_ahead;
        forward_ns = timer.read();
    }
    while (!use_serial_pipeline and generated.items.len < max_tokens and pending != 248044 and pending != 248046) {
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
        if (draft) |*d| {
            d.captureTarget(&m, &p, result.path[0..result.count], positions[0..n]);
            try d.absorb(&m, &p, result.path[0..result.kept], window[0..n]);
        }
        commit_ns += stage.read();
        rounds += 1;
        if (result.stop) break;
    }
    const seconds = @as(f64, @floatFromInt(timer.read())) / 1e9;
    const capture_base: ?[]const u8 = if (draft) |d| (if (d.capture) |writer| writer.base else null) else null;
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
    if (use_serial_pipeline) {
        std.debug.print("Pipelined target/sample/cache: {d:.3}s; {d} steps queued ahead\n", .{ @as(f64, @floatFromInt(forward_ns)) / 1e9, queued_serial_steps });
    } else std.debug.print("Stage totals: draft {d:.3}s, target+sample {d:.3}s, commit+absorb {d:.3}s\n", .{ @as(f64, @floatFromInt(draft_ns)) / 1e9, @as(f64, @floatFromInt(forward_ns)) / 1e9, @as(f64, @floatFromInt(commit_ns)) / 1e9 });
    std.debug.print("Generated {d} tokens in {d:.3}s ({d:.2} tok/s), {d} rounds, {d} accepted drafts\nIDs: {any}\n", .{ generated.items.len, seconds, @as(f64, @floatFromInt(generated.items.len)) / seconds, rounds, accepted, generated.items });
    if (report) |path| {
        var version = mx.c.mlx_string_new();
        defer _ = mx.c.mlx_string_free(version);
        try mx.check(mx.c.mlx_version(&version));
        var peak: usize = 0;
        var active: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        try mx.check(mx.c.mlx_get_active_memory(&active));
        const bonsai_form = if (m.weights.bonsai_form) |form| try form.name(init.arena.allocator()) else null;
        const content = try std.json.Stringify.valueAlloc(allocator, .{ .mlx_version = std.mem.span(mx.c.mlx_string_data(version)), .bonsai_form = bonsai_form, .prompt_tokens = tokens.items, .tokens = generated.items, .text = text, .seed = settings.seed, .temperature = settings.temperature, .top_k = settings.top_k, .top_p = settings.top_p, .metal_sampling = settings.metal, .context_copy = copy_enabled, .draft_capture = capture_base, .serial_pipeline = use_serial_pipeline, .kv_buffers = @import("kv_buffer.zig").enabled, .queued_serial_steps = queued_serial_steps, .load_seconds = load_seconds, .warmup_seconds = warmup_seconds, .startup_seconds = startup_seconds, .calibration_seconds = @as(f64, 0), .prefill_mode = if (lane_prefill) "lane" else "regular", .metal_backend = if (mx.tensor_units) "tensor" else "simd", .prefill_seconds = prefill_seconds, .decode_seconds = seconds, .rounds = rounds, .accepted_drafts = accepted, .warmed = warmup, .peak_mlx_bytes = peak, .active_mlx_bytes = active, .token_sha256 = std.fmt.bytesToHex(digest, .lower) }, .{});
        defer allocator.free(content);
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, content);
    }
}

test {
    _ = @import("draft_allocation.zig");
    _ = @import("memory_budget.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prefill_plan.zig");
    _ = @import("draft_capture.zig");
    _ = @import("draft_calibration.zig");
    _ = @import("tool_stream.zig");
    _ = @import("tool_calls.zig");
    _ = @import("chat.zig");
    _ = @import("deepseek_prompts.zig");
    _ = @import("image_source.zig");
    _ = @import("request_options.zig");
    _ = @import("thinking_budget.zig");
    _ = @import("image_http.zig");
    _ = @import("server_control.zig");
    _ = @import("reply_text.zig");
    _ = @import("vision_positions.zig");
    _ = @import("image_input.zig");
    _ = @import("gemma_ops.zig");
    _ = @import("large_family_ops.zig");
    _ = @import("deepseek_dense.zig");
    _ = @import("deepseek.zig");
    _ = @import("deepseek_dspark.zig");
    _ = @import("draft_depth.zig");
    _ = @import("dflash.zig");
    _ = @import("bonsai.zig");
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
