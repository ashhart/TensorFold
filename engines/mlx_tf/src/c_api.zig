//! TensorFold `mlx_tf` C ABI exported by `libmlxtf.dylib`.
//! Allows Python, Swift, Rust, and C applications to drive bare-Metal LLM inference directly
//! with zero IPC and zero Apple MLX dependency.
const std = @import("std");
const mtl = @import("metal4.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const spec_mod = @import("speculate.zig");
const draft_mod = @import("drafter.zig");
const cross_mod = @import("cross_draft.zig");
const samp_mod = @import("sampler.zig");
const batch_mod = @import("batch.zig");
const srv_mod = @import("server.zig");
const conv_mod = @import("convert.zig");
const tfq3_conv = @import("tfq3_convert.zig");
const vision_mod = @import("vision.zig");

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

pub export fn mlxtf_version() callconv(.c) [*:0]const u8 {
    return "0.4.0";
}

const Model = model_mod.Model;
const Tokenizer = tok_mod.Tokenizer;
const nowMs = model_mod.nowMs;

pub const MlxTfGenOptions = extern struct {
    max_tokens: u32 = 256,
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: f32 = 0.0,
    seed: u64 = 42,
    enable_thinking: u8 = 0,
    use_chat_template: u8 = 0,
    stop_on_eos: u8 = 1,
    _pad: u8 = 0,
    repetition_penalty: f32 = 1.0,
    frequency_penalty: f32 = 0.0,
    presence_penalty: f32 = 0.0,
    _reserved: u32 = 0,
};

pub const MlxTfStats = extern struct {
    prompt_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    rounds: u32 = 0,
    drafts_accepted: u32 = 0,
    drafts_proposed: u32 = 0,
    cached_tokens: u32 = 0,
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    total_ms: f64 = 0,
    tok_per_sec: f64 = 0,
};

pub const MlxTfTokenCb = ?*const fn (user_data: ?*anyopaque, token_id: u32, utf8: [*]const u8, utf8_len: usize) callconv(.c) u8;
pub const MlxTfWaveTokenCb = ?*const fn (user_data: ?*anyopaque, stream_idx: u32, token_id: u32, utf8: [*]const u8, utf8_len: usize) callconv(.c) u8;

pub const MlxTfEngine = struct {
    a: std.mem.Allocator,
    ctx: mtl.Context,
    tok: Tokenizer,
    model: Model,
    dflash: ?draft_mod.Drafter = null,
    dflash_dec: ?draft_mod.HybridDFlashDecoder = null,
    cross_model: ?Model = null,
    cross_dec: ?cross_mod.CrossModelDecoder = null,
    copy_dec: spec_mod.AdaptiveSpecDecoder,
    fork_engine: ?batch_mod.ForkBatchEngine = null,
    vision: ?vision_mod.VisionEncoder = null,
    max_streams: usize,
    device_cstr: [128]u8 = @splat(0),
    arch_cstr: [64]u8 = @splat(0),
};

fn cSliceOpt(ptr: ?[*:0]const u8) ?[]const u8 {
    const p = ptr orelse return null;
    const s = std.mem.span(p);
    return if (s.len == 0) null else s;
}

pub export fn mlxtf_load(
    model_dir_c: [*:0]const u8,
    dflash_dir_c: ?[*:0]const u8,
    cross_draft_dir_c: ?[*:0]const u8,
    max_context: u32,
    max_streams: u32,
) callconv(.c) ?*MlxTfEngine {
    return mlxtf_load_ex(model_dir_c, dflash_dir_c, cross_draft_dir_c, max_context, max_streams, 0, 0);
}

pub export fn mlxtf_load_ex(
    model_dir_c: [*:0]const u8,
    dflash_dir_c: ?[*:0]const u8,
    cross_draft_dir_c: ?[*:0]const u8,
    max_context: u32,
    max_streams: u32,
    force_simd: u8,
    keep_bf16: u8,
) callconv(.c) ?*MlxTfEngine {
    const a = std.heap.smp_allocator;
    const model_dir = std.mem.span(model_dir_c);
    const dflash_dir = cSliceOpt(dflash_dir_c);
    const cross_dir = cSliceOpt(cross_draft_dir_c);

    // Scoped MLX_TF_BF16 toggle when caller explicitly requests lossless BF16
    var restore_bf16 = false;
    var prev_bf16_copy: ?[:0]u8 = null;
    if (keep_bf16 != 0) {
        if (getenv("MLX_TF_BF16")) |prev| {
            prev_bf16_copy = a.dupeSentinel(u8, std.mem.span(prev), 0) catch null;
        }
        _ = setenv("MLX_TF_BF16", "1", 1);
        restore_bf16 = true;
    }
    defer if (restore_bf16) {
        if (prev_bf16_copy) |pc| {
            _ = setenv("MLX_TF_BF16", pc.ptr, 1);
            a.free(pc);
        } else {
            _ = unsetenv("MLX_TF_BF16");
        }
    };

    const eng = a.create(MlxTfEngine) catch return null;
    errdefer a.destroy(eng);

    var ctx = mtl.Context.init(1024) catch return null;
    if (force_simd != 0) ctx.force_simd = true;
    eng.* = .{
        .a = a,
        .ctx = ctx,
        .tok = undefined,
        .model = undefined,
        .copy_dec = undefined,
        .max_streams = @max(1, @min(@as(usize, if (max_streams == 0) 16 else max_streams), batch_mod.max_streams)),
    };
    errdefer eng.ctx.deinit();

    const dname = eng.ctx.deviceName();
    const dn = @min(dname.len, eng.device_cstr.len - 1);
    @memcpy(eng.device_cstr[0..dn], dname[0..dn]);

    eng.tok = Tokenizer.load(a, model_dir) catch return null;
    errdefer eng.tok.deinit();

    const ctx_len = if (max_context == 0) 8192 else max_context;
    eng.model = Model.init(a, &eng.ctx, model_dir, .{
        .max_context = ctx_len,
        .capture_taps = (dflash_dir != null),
    }) catch return null;
    errdefer eng.model.deinit();

    const aname = @tagName(eng.model.c.arch);
    const an = @min(aname.len, eng.arch_cstr.len - 1);
    @memcpy(eng.arch_cstr[0..an], aname[0..an]);

    eng.copy_dec = spec_mod.AdaptiveSpecDecoder.init(a, 127) catch return null;
    errdefer eng.copy_dec.deinit();

    if (dflash_dir) |dd| {
        eng.dflash = draft_mod.Drafter.init(a, &eng.ctx, &eng.model, dd) catch return null;
        eng.dflash_dec = draft_mod.HybridDFlashDecoder.init(a, &eng.dflash.?, 15) catch return null;
    } else if (cross_dir) |cd| {
        eng.cross_model = Model.init(a, &eng.ctx, cd, .{ .max_context = ctx_len }) catch return null;
        eng.cross_dec = cross_mod.CrossModelDecoder.init(a, &eng.model, &eng.cross_model.?, 5) catch return null;
    }

    eng.vision = vision_mod.VisionEncoder.tryInit(a, &eng.ctx, model_dir) catch null;

    return eng;
}

pub export fn mlxtf_free(eng: ?*MlxTfEngine) callconv(.c) void {
    const e = eng orelse return;
    if (e.vision) |*v| v.deinit();
    if (e.fork_engine) |*fe| fe.deinit();
    if (e.dflash_dec) |*dd| dd.deinit();
    if (e.dflash) |*d| d.deinit();
    if (e.cross_dec) |*cd| cd.deinit();
    if (e.cross_model) |*cm| cm.deinit();
    e.copy_dec.deinit();
    e.model.deinit();
    e.tok.deinit();
    e.ctx.deinit();
    e.a.destroy(e);
}

pub export fn mlxtf_device_name(eng: ?*MlxTfEngine) callconv(.c) [*:0]const u8 {
    const e = eng orelse return "Apple GPU";
    return @ptrCast(&e.device_cstr);
}

pub export fn mlxtf_arch_name(eng: ?*MlxTfEngine) callconv(.c) [*:0]const u8 {
    const e = eng orelse return "unknown";
    return @ptrCast(&e.arch_cstr);
}

pub export fn mlxtf_uses_m5_tensor_ops(eng: ?*MlxTfEngine) callconv(.c) u8 {
    const e = eng orelse return 0;
    return if (e.ctx.has_tensor_ops) 1 else 0;
}

pub export fn mlxtf_has_vision(eng: ?*MlxTfEngine) callconv(.c) u8 {
    const e = eng orelse return 0;
    return if (e.vision != null) 1 else 0;
}

pub export fn mlxtf_encode_image(
    eng: ?*MlxTfEngine,
    path_c: [*:0]const u8,
    out_f32: ?[*]f32,
    max_elems: usize,
    out_num_tokens: ?*u32,
    out_hidden: ?*u32,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    const ve = if (e.vision) |*v| v else return -2;
    const enc = ve.encodeImage(e.a, std.mem.span(path_c)) catch return -3;
    defer enc.deinit(e.a);
    if (out_num_tokens) |p| p.* = enc.num_tokens;
    if (out_hidden) |p| p.* = enc.hidden_size;
    if (out_f32) |dst| {
        if (max_elems < enc.embs.len) return -4;
        for (enc.embs, 0..) |b, i| dst[i] = @bitCast(@as(u32, b) << 16);
    }
    return 0;
}

pub export fn mlxtf_encode_patches(
    eng: ?*MlxTfEngine,
    patches_bf16: [*]const u16,
    num_elems: usize,
    grid_h: u32,
    grid_w: u32,
    out_f32: ?[*]f32,
    max_elems: usize,
    out_num_tokens: ?*u32,
    out_hidden: ?*u32,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    const ve = if (e.vision) |*v| v else return -2;
    const enc = ve.encodePatches(e.a, patches_bf16[0..num_elems], grid_h, grid_w) catch return -3;
    defer enc.deinit(e.a);
    if (out_num_tokens) |p| p.* = enc.num_tokens;
    if (out_hidden) |p| p.* = enc.hidden_size;
    if (out_f32) |dst| {
        if (max_elems < enc.embs.len) return -4;
        for (enc.embs, 0..) |b, i| dst[i] = @bitCast(@as(u32, b) << 16);
    }
    return 0;
}

pub export fn mlxtf_generate_with_images(
    eng: ?*MlxTfEngine,
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    image_paths: ?[*]const [*:0]const u8,
    num_images: u32,
    opts_ptr: ?*const MlxTfGenOptions,
    cb: MlxTfTokenCb,
    user_data: ?*anyopaque,
    out_stats: ?*MlxTfStats,
) callconv(.c) i32 {
    if (num_images == 0 or image_paths == null) {
        return mlxtf_generate(eng, prompt_ptr, prompt_len, opts_ptr, cb, user_data, out_stats);
    }
    const e = eng orelse return -1;
    const ve = if (e.vision) |*v| v else return -10;
    const opts = if (opts_ptr) |p| p.* else MlxTfGenOptions{};
    const raw_prompt = prompt_ptr[0..prompt_len];
    const t_all0 = nowMs();

    var vis_embs: std.ArrayList(u16) = .empty;
    defer vis_embs.deinit(e.a);
    var img_counts: std.ArrayList(u32) = .empty;
    defer img_counts.deinit(e.a);

    const paths = image_paths.?[0..num_images];
    for (paths) |p_c| {
        const enc = ve.encodeImage(e.a, std.mem.span(p_c)) catch return -11;
        defer enc.deinit(e.a);
        vis_embs.appendSlice(e.a, enc.embs) catch return -12;
        img_counts.append(e.a, enc.num_tokens) catch return -12;
    }

    const prompt_ids = vision_mod.buildMultimodalPrompt(
        e.a,
        &e.tok,
        ve.c,
        raw_prompt,
        img_counts.items,
        opts.use_chat_template != 0,
        opts.enable_thinking != 0,
    ) catch return -3;
    defer e.a.free(prompt_ids);
    if (prompt_ids.len == 0) return -5;
    if (prompt_ids.len >= e.model.max_context) return -6;
    const max_toks: u32 = @intCast(@min(@as(usize, opts.max_tokens), @as(usize, e.model.max_context) - prompt_ids.len));

    const stop_eos = (opts.stop_on_eos != 0);
    const has_penalties = (opts.repetition_penalty != 1.0 and opts.repetition_penalty > 0.0) or
        opts.frequency_penalty != 0.0 or opts.presence_penalty != 0.0;
    const need_logits = (opts.temperature > 1e-6) or has_penalties;

    const sp = samp_mod.SamplingParams{
        .temperature = opts.temperature,
        .top_p = opts.top_p,
        .top_k = opts.top_k,
        .min_p = opts.min_p,
        .seed = opts.seed,
        .repetition_penalty = if (opts.repetition_penalty > 0.0) opts.repetition_penalty else 1.0,
        .frequency_penalty = opts.frequency_penalty,
        .presence_penalty = opts.presence_penalty,
    };
    var rng = samp_mod.Rng.init(opts.seed);
    var counts = std.AutoHashMap(u32, u32).init(e.a);
    defer counts.deinit();
    if (has_penalties) {
        for (prompt_ids) |pid| {
            const gop = counts.getOrPut(pid) catch return -4;
            gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
        }
    }

    e.model.reset();
    const tp0 = nowMs();
    var next_tok = e.model.prefillWithEmbeddings(prompt_ids, vis_embs.items, ve.c.image_token_id, 128) catch return -4;
    const prefill_ms = nowMs() - tp0;
    if (need_logits) {
        next_tok = e.model.sampleRowWithPenalties(0, sp, &rng, if (has_penalties) &counts else null) catch return -4;
    }

    var stats = spec_mod.SpecStats{};
    const td0 = nowMs();
    while (stats.tokens_generated < max_toks) {
        if (stop_eos and e.model.c.isEos(next_tok)) break;
        if (has_penalties) {
            const gop = counts.getOrPut(next_tok) catch return -4;
            gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
        }
        if (cb) |f| {
            const piece = e.tok.decodeToken(next_tok);
            if (f(user_data, next_tok, piece.ptr, piece.len) == 0) break;
        }
        stats.tokens_generated += 1;
        stats.rounds += 1;
        if (stats.tokens_generated >= max_toks) break;
        if (need_logits) {
            _ = e.model.stepWithLogits(next_tok) catch return -4;
            next_tok = e.model.sampleRowWithPenalties(0, sp, &rng, if (has_penalties) &counts else null) catch return -4;
        } else {
            next_tok = e.model.step(next_tok) catch return -4;
        }
    }
    stats.decode_ms = nowMs() - td0;

    if (out_stats) |os| {
        os.* = .{
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = @intCast(stats.tokens_generated),
            .rounds = @intCast(stats.rounds),
            .drafts_accepted = 0,
            .drafts_proposed = 0,
            .cached_tokens = 0,
            .prefill_ms = prefill_ms,
            .decode_ms = stats.decode_ms,
            .total_ms = nowMs() - t_all0,
            .tok_per_sec = stats.tokPerSec(),
        };
    }
    return 0;
}

pub export fn mlxtf_generate(
    eng: ?*MlxTfEngine,
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    opts_ptr: ?*const MlxTfGenOptions,
    cb: MlxTfTokenCb,
    user_data: ?*anyopaque,
    out_stats: ?*MlxTfStats,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    const opts = if (opts_ptr) |p| p.* else MlxTfGenOptions{};
    const raw_prompt = prompt_ptr[0..prompt_len];
    const t_all0 = nowMs();

    const prompt_text = if (opts.use_chat_template != 0)
        (e.tok.formatChat(e.a, raw_prompt, opts.enable_thinking != 0) catch return -2)
    else
        (e.a.dupe(u8, raw_prompt) catch return -2);
    defer e.a.free(prompt_text);

    const prompt_ids = e.tok.encode(e.a, prompt_text) catch return -3;
    defer e.a.free(prompt_ids);
    if (prompt_ids.len == 0) return -5;
    if (prompt_ids.len >= e.model.max_context) return -6;
    const max_toks: u32 = @intCast(@min(@as(usize, opts.max_tokens), @as(usize, e.model.max_context) - prompt_ids.len));

    const Bridge = struct {
        tok: *Tokenizer,
        cb: MlxTfTokenCb,
        ud: ?*anyopaque,
        fn onTok(raw: ?*anyopaque, tok_id: u32) void {
            const b: *@This() = @ptrCast(@alignCast(raw.?));
            if (b.cb) |f| {
                const piece = b.tok.decodeToken(tok_id);
                _ = f(b.ud, tok_id, piece.ptr, piece.len);
            }
        }
    };
    var bridge = Bridge{ .tok = &e.tok, .cb = cb, .ud = user_data };
    const zig_cb: ?*const fn (?*anyopaque, u32) void = if (cb != null) Bridge.onTok else null;

    var stats = spec_mod.SpecStats{};
    var prefill_ms: f64 = 0;
    const stop_eos = (opts.stop_on_eos != 0);
    const has_penalties = (opts.repetition_penalty != 1.0 and opts.repetition_penalty > 0.0) or
        opts.frequency_penalty != 0.0 or opts.presence_penalty != 0.0;

    if (opts.temperature <= 1e-6 and !has_penalties) {
        if (e.dflash_dec != null and max_toks > 1) {
            const tp0 = nowMs();
            stats = e.dflash_dec.?.generate(prompt_ids, max_toks, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
            prefill_ms = @max(0.0, (nowMs() - tp0) - stats.decode_ms);
        } else if (e.cross_dec != null and max_toks > 1) {
            const tp0 = nowMs();
            stats = e.cross_dec.?.generate(prompt_ids, max_toks, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
            prefill_ms = @max(0.0, (nowMs() - tp0) - stats.decode_ms);
        } else if (e.model.c.arch == .nemotron_h or prompt_ids.len < 96 or max_toks <= 1) {
            e.model.reset();
            const tp0 = nowMs();
            var next_tok = e.model.prefill(prompt_ids, 128) catch return -4;
            prefill_ms = nowMs() - tp0;
            const td0 = nowMs();
            while (stats.tokens_generated < max_toks) {
                if (stop_eos and e.model.c.isEos(next_tok)) break;
                if (cb) |f| {
                    const piece = e.tok.decodeToken(next_tok);
                    if (f(user_data, next_tok, piece.ptr, piece.len) == 0) break;
                }
                stats.tokens_generated += 1;
                stats.rounds += 1;
                if (stats.tokens_generated >= max_toks) break;
                next_tok = e.model.step(next_tok) catch return -4;
            }
            stats.decode_ms = nowMs() - td0;
        } else {
            e.model.reset();
            const tp0 = nowMs();
            stats = e.copy_dec.generate(&e.model, prompt_ids, max_toks, stop_eos, null, e.a, zig_cb, &bridge) catch return -4;
            prefill_ms = @max(0.0, (nowMs() - tp0) - stats.decode_ms);
        }
    } else {
        const sp = samp_mod.SamplingParams{
            .temperature = opts.temperature,
            .top_p = opts.top_p,
            .top_k = opts.top_k,
            .min_p = opts.min_p,
            .seed = opts.seed,
            .repetition_penalty = if (opts.repetition_penalty > 0.0) opts.repetition_penalty else 1.0,
            .frequency_penalty = opts.frequency_penalty,
            .presence_penalty = opts.presence_penalty,
        };
        if (e.cross_dec != null and !has_penalties and opts.temperature > 1e-6 and max_toks > 1) {
            const tp0 = nowMs();
            stats = e.cross_dec.?.generateSampled(prompt_ids, max_toks, stop_eos, sp, null, e.a, zig_cb, &bridge) catch return -4;
            prefill_ms = @max(0.0, (nowMs() - tp0) - stats.decode_ms);
        } else {
            var rng = samp_mod.Rng.init(opts.seed);
            var counts = std.AutoHashMap(u32, u32).init(e.a);
            defer counts.deinit();
            if (has_penalties) {
                for (prompt_ids) |pid| {
                    const gop = counts.getOrPut(pid) catch return -4;
                    gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
                }
            }
            e.model.reset();
            const tp0 = nowMs();
            _ = e.model.prefill(prompt_ids, 128) catch return -4;
            prefill_ms = nowMs() - tp0;
            var tok_id = e.model.sampleRowWithPenalties(0, sp, &rng, if (has_penalties) &counts else null) catch return -4;
            const td0 = nowMs();
            while (stats.tokens_generated < max_toks) {
                if (stop_eos and e.model.c.isEos(tok_id)) break;
                if (has_penalties) {
                    const gop = counts.getOrPut(tok_id) catch return -4;
                    gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
                }
                if (cb) |f| {
                    const piece = e.tok.decodeToken(tok_id);
                    if (f(user_data, tok_id, piece.ptr, piece.len) == 0) break;
                }
                stats.tokens_generated += 1;
                stats.rounds += 1;
                if (stats.tokens_generated >= max_toks) break;
                _ = e.model.stepWithLogits(tok_id) catch return -4;
                tok_id = e.model.sampleRowWithPenalties(0, sp, &rng, if (has_penalties) &counts else null) catch return -4;
            }
            stats.decode_ms = nowMs() - td0;
        }
    }

    if (out_stats) |os| {
        os.* = .{
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = @intCast(stats.tokens_generated),
            .rounds = @intCast(stats.rounds),
            .drafts_accepted = @intCast(stats.drafts_accepted),
            .drafts_proposed = @intCast(stats.drafts_proposed),
            .cached_tokens = 0,
            .prefill_ms = prefill_ms,
            .decode_ms = stats.decode_ms,
            .total_ms = nowMs() - t_all0,
            .tok_per_sec = stats.tokPerSec(),
        };
    }
    return 0;
}

pub export fn mlxtf_stream_parallel(
    eng: ?*MlxTfEngine,
    shared_ptr: [*]const u8,
    shared_len: usize,
    n_streams: u32,
    deltas_ptrs: [*]const [*]const u8,
    delta_lens: [*]const usize,
    max_tokens_arr: [*]const u32,
    temps_arr: [*]const f32,
    cb: MlxTfWaveTokenCb,
    user_data: ?*anyopaque,
    out_stats: ?*MlxTfStats,
) callconv(.c) i32 {
    const e = eng orelse return -1;
    if (n_streams == 0 or n_streams > e.max_streams) return -2;
    const t_all0 = nowMs();

    if (e.fork_engine == null) {
        e.fork_engine = batch_mod.ForkBatchEngine.init(e.a, &e.model, e.max_streams, 2048) catch return -3;
    }

    const shared_ids = e.tok.encode(e.a, shared_ptr[0..shared_len]) catch return -4;
    defer e.a.free(shared_ids);

    const n: usize = n_streams;
    const specs = e.a.alloc(batch_mod.StreamSpec, n) catch return -5;
    defer e.a.free(specs);
    const delta_bufs = e.a.alloc([]u32, n) catch return -5;
    for (delta_bufs) |*d| d.* = &.{};
    defer {
        for (delta_bufs) |d| if (d.len > 0) e.a.free(d);
        e.a.free(delta_bufs);
    }

    for (0..n) |i| {
        const d_txt = deltas_ptrs[i][0..delta_lens[i]];
        delta_bufs[i] = e.tok.encode(e.a, d_txt) catch return -4;
        specs[i] = .{
            .delta_tokens = delta_bufs[i],
            .max_tokens = max_tokens_arr[i],
            .sampling = .{ .temperature = temps_arr[i], .seed = 42 + i },
        };
    }

    const WaveBridge = struct {
        tok: *Tokenizer,
        cb: MlxTfWaveTokenCb,
        ud: ?*anyopaque,
        fn onTok(raw: ?*anyopaque, sidx: usize, tok_id: u32) void {
            const b: *@This() = @ptrCast(@alignCast(raw.?));
            if (b.cb) |f| {
                const piece = b.tok.decodeToken(tok_id);
                _ = f(b.ud, @intCast(sidx), tok_id, piece.ptr, piece.len);
            }
        }
    };
    var wb = WaveBridge{ .tok = &e.tok, .cb = cb, .ud = user_data };
    const zig_cb: ?batch_mod.StreamTokenCallback = if (cb != null) WaveBridge.onTok else null;

    const ws = e.fork_engine.?.runWave(shared_ids, specs, null, e.a, zig_cb, &wb) catch return -6;

    if (out_stats) |os| {
        os.* = .{
            .prompt_tokens = @intCast(ws.prefix_tokens),
            .completion_tokens = @intCast(ws.total_generated),
            .rounds = @intCast(ws.decode_steps),
            .drafts_accepted = 0,
            .drafts_proposed = 0,
            .cached_tokens = @intCast(ws.cached_tokens),
            .decode_ms = ws.decode_ms,
            .total_ms = nowMs() - t_all0,
            .tok_per_sec = ws.decodeTokPerSec(),
        };
    }
    return 0;
}

pub export fn mlxtf_serve(eng: ?*MlxTfEngine, port: u16, model_name_c: ?[*:0]const u8) callconv(.c) i32 {
    return mlxtf_serve_host(eng, "127.0.0.1", port, model_name_c);
}

pub export fn mlxtf_serve_host(eng: ?*MlxTfEngine, host_c: ?[*:0]const u8, port: u16, model_name_c: ?[*:0]const u8) callconv(.c) i32 {
    const e = eng orelse return -1;
    const host = cSliceOpt(host_c) orelse "127.0.0.1";
    const mname = cSliceOpt(model_name_c) orelse "tensorfold-model";
    const dd_ptr: ?*draft_mod.HybridDFlashDecoder = if (e.dflash_dec) |*dd| dd else null;
    const cd_ptr: ?*cross_mod.CrossModelDecoder = if (e.cross_dec) |*cd| cd else null;
    const ve_ptr: ?*vision_mod.VisionEncoder = if (e.vision) |*v| v else null;

    var srv = srv_mod.ServerEngine.init(e.a, &e.ctx, &e.model, &e.tok, mname, dd_ptr, cd_ptr, ve_ptr, e.max_streams) catch return -2;
    defer srv.deinit();
    srv.serveHost(host, port) catch return -3;
    return 0;
}

pub export fn mlxtf_convert(
    src_dir_c: [*:0]const u8,
    dst_dir_c: [*:0]const u8,
    bits: u32,
    group_size: u32,
) callconv(.c) i32 {
    const a = std.heap.smp_allocator;
    const src_dir = std.mem.span(src_dir_c);
    const dst_dir = std.mem.span(dst_dir_c);
    const stats = conv_mod.convertModel(a, src_dir, dst_dir, .{
        .bits = if (bits == 0) 4 else bits,
        .group_size = if (group_size == 0) 64 else group_size,
    }) catch return -1;
    std.debug.print("TensorFold mlx_tf convert: {d} tensors ({d} quantized) -> {d} tensors | {d:.1} MiB -> {d:.1} MiB in {d:.0} ms\n", .{
        stats.tensors_in,
        stats.tensors_quantized,
        stats.tensors_out,
        @as(f64, @floatFromInt(stats.bytes_in)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(stats.bytes_out)) / (1024.0 * 1024.0),
        stats.elapsed_ms,
    });
    return 0;
}

pub export fn mlxtf_convert_tfq3(
    src_dir_c: [*:0]const u8,
    dst_dir_c: [*:0]const u8,
    bits: u32,
    profile_c: ?[*:0]const u8,
    head_bits: u32,
    quantize_head: u8,
) callconv(.c) i32 {
    const a = std.heap.smp_allocator;
    const src_dir = std.mem.span(src_dir_c);
    const dst_dir = std.mem.span(dst_dir_c);
    const profile = cSliceOpt(profile_c) orelse "";
    const stats = tfq3_conv.convertModelTfq3(a, src_dir, dst_dir, .{
        .bits = if (bits == 0) 4 else bits,
        .profile = profile,
        .head_bits = head_bits,
        .quantize_head = (quantize_head != 0),
    }) catch return -1;
    std.debug.print("TensorFold mlx_tf convert (TFQ3 v3): {d} TFQ3 ({d:.2} dB) + {d} BF16 across {d} shard(s) | {d:.1} MiB -> {d:.1} MiB in {d:.0} ms\n", .{
        stats.tensors_tfq3,
        stats.mean_snr_db,
        stats.tensors_residual,
        stats.shards_written,
        @as(f64, @floatFromInt(stats.bytes_in)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(stats.bytes_out)) / (1024.0 * 1024.0),
        stats.elapsed_ms,
    });
    return 0;
}
