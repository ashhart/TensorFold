//! m4lm CLI: clean-room Metal + Zig inference engine replacing Apple MLX / mlx-lm.
//!
//! Commands:
//!   m4lm inspect      MODEL_DIR
//!   m4lm generate     MODEL_DIR --prompt "..." [--max-tokens N] [--draft DFLASH] [--cross-draft DRAFT] [--temp F] [--top-p F] [--top-k N] [--min-p F] [--seed N] [--no-chat] [--think] [--no-spec]
//!   m4lm serve        MODEL_DIR [--draft DFLASH] [--cross-draft DRAFT] [--port 8080] [--name NAME] [--max-streams 16]
//!   m4lm bench        MODEL_DIR [--prompt-len N] [--tokens N] [--warmup N]
//!   m4lm wave-bench   MODEL_DIR [--streams 16] [--tokens 128]
//!   m4lm verify       MODEL_DIR REF_JSON KEY
//!   m4lm golden       MODEL_DIR PROMPT_JSON EXPECTED_JSON [--limit N]
//!   m4lm spec-check   MODEL_DIR
//!   m4lm cross-check  TARGET_DIR DRAFT_DIR [--k 5]
//!   m4lm dflash-check TARGET_DIR DRAFTER_DIR
//!   m4lm timeline     MODEL_DIR
const std = @import("std");
const mtl = @import("metal4.zig");
const cfg = @import("config.zig");
const weights = @import("weights.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const spec_mod = @import("speculate.zig");
const draft_mod = @import("drafter.zig");
const cross_mod = @import("cross_draft.zig");
const samp_mod = @import("sampler.zig");
const batch_mod = @import("batch.zig");
const srv_mod = @import("server.zig");
const pcache_mod = @import("prefix_cache.zig");
const rewind_mod = @import("dual_rewind.zig");
const conv_mod = @import("convert.zig");
const tfq3_conv = @import("tfq3_convert.zig");
const vision_mod = @import("vision.zig");
const st = @import("safetensors.zig");

const Model = model_mod.Model;
const Tokenizer = tok_mod.Tokenizer;
const nowMs = model_mod.nowMs;

extern "c" fn write(fd: c_int, buf: [*]const u8, count: usize) isize;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub const version_str = "0.4.0";

fn writeStdout(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = write(1, bytes.ptr + off, bytes.len - off);
        if (n <= 0) break;
        off += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len >= 2) {
        if (std.mem.eql(u8, args[1], "--version") or std.mem.eql(u8, args[1], "-V") or std.mem.eql(u8, args[1], "version")) {
            writeStdout("TensorFold mlx_tf " ++ version_str ++ "\n");
            return 0;
        }
        if (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h") or std.mem.eql(u8, args[1], "help")) {
            printUsage();
            return 0;
        }
    }

    if (args.len < 3) {
        printUsage();
        return 2;
    }

    const cmd = args[1];
    const model_dir = args[2];

    // Global runtime flags that can appear anywhere in args[3..]
    for (args[3..]) |arg| {
        if (std.mem.eql(u8, arg, "--bf16")) _ = setenv("MLX_TF_BF16", "1", 1);
    }

    if (std.mem.eql(u8, cmd, "convert")) {
        if (args.len < 4) {
            std.debug.print("usage: mlx_tf convert SRC_DIR DST_DIR [--format affine|tfq3] [--bits 2..8] [--group-size 32|64|128] [--profile 3b|3.5b|4b|5b]\n", .{});
            return 2;
        }
        return cmdConvert(a, model_dir, args[3], args[4..]);
    }

    var ctx = try mtl.Context.init(1024);
    defer ctx.deinit();

    for (args[3..]) |arg| {
        if (std.mem.eql(u8, arg, "--force-simd")) ctx.force_simd = true;
    }

    if (std.mem.eql(u8, cmd, "inspect") or std.mem.eql(u8, cmd, "pack")) {
        return cmdInspect(a, &ctx, model_dir);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        if (args.len < 5) {
            std.debug.print("usage: m4lm verify MODEL_DIR REF_JSON KEY\n", .{});
            return 2;
        }
        return cmdVerify(a, &ctx, model_dir, args[3], args[4]);
    } else if (std.mem.eql(u8, cmd, "generate")) {
        return cmdGenerate(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "serve")) {
        return cmdServe(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        return cmdBench(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "wave-bench")) {
        return cmdWaveBench(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "cache-bench")) {
        return cmdCacheBench(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "roofline")) {
        return cmdRoofline(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "tf-rewind") or std.mem.eql(u8, cmd, "dual-rewind")) {
        return cmdDualRewind(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "tf-rewind-demo") or std.mem.eql(u8, cmd, "needle-demo")) {
        return cmdNeedleDemo(a, &ctx, model_dir, args[3..]);
    } else if (std.mem.eql(u8, cmd, "golden")) {
        if (args.len < 5) {
            std.debug.print("usage: m4lm golden MODEL_DIR PROMPT_JSON EXPECTED_JSON [--limit N]\n", .{});
            return 2;
        }
        return cmdGolden(a, &ctx, model_dir, args[3], args[4], args[5..]);
    } else if (std.mem.eql(u8, cmd, "spec-check")) {
        return cmdSpecCheck(a, &ctx, model_dir);
    } else if (std.mem.eql(u8, cmd, "cross-check")) {
        if (args.len < 4) {
            std.debug.print("usage: m4lm cross-check TARGET_DIR DRAFT_DIR [--k 5]\n", .{});
            return 2;
        }
        return cmdCrossCheck(a, &ctx, model_dir, args[3], args[4..]);
    } else if (std.mem.eql(u8, cmd, "dflash-check")) {
        if (args.len < 4) {
            std.debug.print("usage: m4lm dflash-check TARGET_DIR DRAFTER_DIR\n", .{});
            return 2;
        }
        return cmdDflashCheck(a, &ctx, model_dir, args[3]);
    } else if (std.mem.eql(u8, cmd, "timeline")) {
        return cmdTimeline(a, &ctx, model_dir, args[3..]);
    }

    printUsage();
    return 2;
}

fn printUsage() void {
    std.debug.print(
        \\TensorFold mlx_tf {s} — Bare-Metal Apple Silicon LLM & VLM Engine (Zero Apple MLX)
        \\
        \\Commands:
        \\  mlx_tf inspect      MODEL_DIR [--bf16]
        \\  mlx_tf convert      SRC_DIR DST_DIR [--format affine|tfq3] [--bits 2..8] [--group-size 32|64|128]
        \\                      [--profile 3b|3.5b|4b|5b] [--head-bits 3|4|5] [--keep-head-bf16]
        \\  mlx_tf generate     MODEL_DIR --prompt "..." [--image PATH ...] [--max-tokens N] [--max-context N]
        \\                      [--draft DFLASH] [--cross-draft DRAFT] [--temp F] [--top-p F] [--top-k N] [--min-p F]
        \\                      [--seed N] [--no-chat] [--think] [--no-spec] [--bf16] [--force-simd]
        \\  mlx_tf serve        MODEL_DIR [--host 127.0.0.1] [--port 8080] [--name NAME] [--max-context 8192]
        \\                      [--max-streams 16] [--draft DFLASH] [--cross-draft DRAFT] [--bf16] [--force-simd]
        \\  mlx_tf bench        MODEL_DIR [--prompt-len N] [--tokens N] [--warmup N] [--no-qmv] [--no-fused] [--bf16] [--force-simd]
        \\  mlx_tf roofline     MODEL_DIR [--tokens 64] [--prefill 512]
        \\  mlx_tf wave-bench   MODEL_DIR [--streams 16] [--tokens 128]
        \\  mlx_tf cache-bench  MODEL_DIR [--repeats 12]
        \\  mlx_tf tf-rewind    MODEL_DIR [--window 64] [--gen 64]
        \\  mlx_tf tf-rewind-demo MODEL_DIR [--repeats 14]
        \\  mlx_tf verify       MODEL_DIR REF_JSON KEY
        \\  mlx_tf golden       MODEL_DIR PROMPT_JSON EXPECTED_JSON [--limit N]
        \\  mlx_tf spec-check   MODEL_DIR [--force-simd]
        \\  mlx_tf cross-check  TARGET_DIR DRAFT_DIR [--k 5]
        \\  mlx_tf dflash-check TARGET_DIR DRAFTER_DIR
        \\  mlx_tf timeline     MODEL_DIR
        \\  mlx_tf --version | --help
        \\
    , .{version_str});
}

fn cmdConvert(a: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var format: []const u8 = "affine";
    var bits: usize = 4;
    var gs: usize = 64;
    var profile: []const u8 = "";
    var head_bits: usize = 0;
    var quantize_head = true;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--format") and i + 1 < rest.len) {
            i += 1;
            format = rest[i];
        } else if (std.mem.eql(u8, rest[i], "--bits") and i + 1 < rest.len) {
            i += 1;
            bits = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--group-size") and i + 1 < rest.len) {
            i += 1;
            gs = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--profile") and i + 1 < rest.len) {
            i += 1;
            profile = rest[i];
        } else if (std.mem.eql(u8, rest[i], "--head-bits") and i + 1 < rest.len) {
            i += 1;
            head_bits = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--keep-head-bf16")) {
            quantize_head = false;
        }
    }

    if (std.mem.eql(u8, format, "tfq3")) {
        const stats = try tfq3_conv.convertModelTfq3(a, src_dir, dst_dir, .{
            .bits = bits,
            .profile = profile,
            .head_bits = head_bits,
            .quantize_head = quantize_head,
        });
        std.debug.print("Converted {s} -> {s} (TFQ3 v3)\n", .{ src_dir, dst_dir });
        std.debug.print("  Tensors: {d} TFQ3 ({s}{d}-bit, mean SNR {d:.2} dB) + {d} BF16 residual across {d} shard(s)\n", .{
            stats.tensors_tfq3,
            if (profile.len > 0) profile else "",
            bits,
            stats.mean_snr_db,
            stats.tensors_residual,
            stats.shards_written,
        });
        std.debug.print("  Size   : {d:.1} MiB -> {d:.1} MiB ({d:.2}x compression) in {d:.0} ms\n", .{
            @as(f64, @floatFromInt(stats.bytes_in)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(stats.bytes_out)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(stats.bytes_in)) / @max(1.0, @as(f64, @floatFromInt(stats.bytes_out))),
            stats.elapsed_ms,
        });
        return 0;
    }

    const stats = try conv_mod.convertModel(a, src_dir, dst_dir, .{ .bits = bits, .group_size = gs });
    std.debug.print("Converted {s} -> {s}\n", .{ src_dir, dst_dir });
    std.debug.print("  Tensors: {d} in ({d} quantized to {d}-bit g{d}) -> {d} out\n", .{
        stats.tensors_in, stats.tensors_quantized, bits, gs, stats.tensors_out,
    });
    std.debug.print("  Size   : {d:.1} MiB -> {d:.1} MiB ({d:.2}x compression) in {d:.0} ms\n", .{
        @as(f64, @floatFromInt(stats.bytes_in)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(stats.bytes_out)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(stats.bytes_in)) / @max(1.0, @as(f64, @floatFromInt(stats.bytes_out))),
        stats.elapsed_ms,
    });
    return 0;
}

fn cmdInspect(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8) !u8 {
    const t0 = nowMs();
    const c = try cfg.ModelConfig.load(a, model_dir);
    var w = try weights.ModelWeights.load(a, ctx, c, model_dir);
    defer w.deinit();
    ctx.commitResidency();
    std.debug.print("Device: {s} (metal4={}, m5_tensor_ops={})\n", .{ ctx.deviceName(), ctx.has_metal4, ctx.has_tensor_ops });
    if (c.arch == .nemotron_h) {
        std.debug.print("Config: arch=nemotron_h hidden={d} layers={d} ({d} Mamba-2 + {d} MoE[{d}x top-{d}] + {d} NoPE-GQA) vocab={d}\n", .{
            c.hidden_size, c.num_layers, c.numMambaLayers(), c.numMoeLayers(), c.n_routed_experts, c.num_experts_per_tok, c.numFullLayers(), c.vocab_size,
        });
    } else {
        std.debug.print("Config: arch={s} hidden={d} inter={d} layers={d} ({d} GDN + {d} GQA) vocab={d} tied={}\n", .{
            @tagName(c.arch), c.hidden_size, c.intermediate_size, c.num_layers, c.numLinearLayers(), c.numFullLayers(), c.vocab_size, c.tie_word_embeddings,
        });
    }
    std.debug.print("Packed {d:.1} MiB into {d} Metal buffers (tfq3={}, shards={d}) in {d:.1} ms.\n", .{
        @as(f64, @floatFromInt(w.total_bytes)) / (1024.0 * 1024.0),
        w.buffers.items.len,
        w.tfq3_store != null,
        if (w.tfq3_store) |ts| ts.shards.items.len else 0,
        nowMs() - t0,
    });
    if (try vision_mod.VisionConfig.load(a, model_dir)) |vc| {
        std.debug.print("Vision : kind={s} hidden={d} depth={d} heads={d} patch={d} out_hidden={d}\n", .{
            @tagName(vc.kind), vc.hidden_size, vc.depth, vc.num_heads, vc.patch_size, vc.out_hidden_size,
        });
    }
    return 0;
}


fn cmdVerify(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, ref_path: []const u8, key: []const u8) !u8 {
    const t0 = nowMs();
    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    const tok_ms = nowMs() - t0;

    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 4096 });
    defer m.deinit();
    const load_ms = nowMs() - t0;
    std.debug.print("Loaded {s} on {s} in {d:.0} ms (tokenizer {d:.0} ms, weights {d:.1} MiB)\n", .{
        key,
        ctx.deviceName(),
        load_ms,
        tok_ms,
        @as(f64, @floatFromInt(m.w.total_bytes)) / (1024.0 * 1024.0),
    });

    const ref_bytes = try st.mapFile(a, ref_path);
    defer st.unmapFile(ref_bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, ref_bytes, .{});
    defer parsed.deinit();
    const entry = parsed.value.object.get(key).?.object;

    // 1. Check native BPE tokenizer parity against HuggingFace/mlx_lm prompt_ids
    const rendered = entry.get("rendered_prompt").?.string;
    const ref_prompt_arr = entry.get("prompt_ids").?.array.items;
    var prompt_ids = try a.alloc(u32, ref_prompt_arr.len);
    defer a.free(prompt_ids);
    for (ref_prompt_arr, 0..) |v, i| prompt_ids[i] = @intCast(v.integer);

    const enc_ids = try tok.encode(a, rendered);
    defer a.free(enc_ids);
    var tok_exact = (enc_ids.len == prompt_ids.len);
    if (tok_exact) {
        for (enc_ids, prompt_ids) |got, want| {
            if (got != want) {
                tok_exact = false;
                break;
            }
        }
    }
    std.debug.print("Tokenizer parity: {s} (got {d} vs ref {d})\n", .{
        if (tok_exact) "EXACT" else "DIFF",
        enc_ids.len,
        prompt_ids.len,
    });
    if (!tok_exact) {
        std.debug.print("  got : {any}\n  want: {any}\n", .{ enc_ids, prompt_ids });
    }

    const ref_out_arr = entry.get("out_ids").?.array.items;
    var ref_out = try a.alloc(u32, ref_out_arr.len);
    defer a.free(ref_out);
    for (ref_out_arr, 0..) |v, i| ref_out[i] = @intCast(v.integer);

    // Warmup run (16 tokens)
    _ = try m.prefill(prompt_ids, 128);
    var w_tok = m.out_tok_buf.slice(u32, 1)[0];
    for (0..15) |_| w_tok = try m.step(w_tok);

    // Timed verification run
    m.reset();
    const tp0 = nowMs();
    var next_tok = try m.prefill(prompt_ids, 128);
    const pp_ms = nowMs() - tp0;

    // Print top-3 first-step logits vs mlx_lm reference
    const ref_top_idx = entry.get("first_step_top8_idx").?.array.items;
    const ref_top_val = entry.get("first_step_top8_val").?.array.items;
    std.debug.print("First-step logits vs mlx_lm: ", .{});
    for (0..@min(3, ref_top_idx.len)) |k| {
        const id: usize = @intCast(ref_top_idx[k].integer);
        const ref_v: f64 = switch (ref_top_val[k]) {
            .float => |f| f,
            .integer => |iv| @floatFromInt(iv),
            else => 0,
        };
        std.debug.print("[id={d} got={d:.3} ref={d:.3}] ", .{ id, m.readLogit(id), ref_v });
    }
    std.debug.print("\n", .{});

    var matched: usize = 0;
    var first_diff: ?usize = null;
    var got_first_diff: u32 = 0;
    var want_first_diff: u32 = 0;
    const n_gen = ref_out.len;
    const td0 = nowMs();
    var step_ms_sum: f64 = 0;
    var encode_ms_sum: f64 = 0;

    for (0..n_gen) |step_i| {
        if (next_tok == ref_out[step_i]) {
            matched += 1;
        } else if (first_diff == null) {
            first_diff = step_i;
            got_first_diff = next_tok;
            want_first_diff = ref_out[step_i];
        }
        if (step_i + 1 < n_gen) {
            next_tok = try m.step(next_tok);
            step_ms_sum += m.last_timing.wall_ms;
            encode_ms_sum += m.last_timing.encode_ms;
        }
    }
    const tg_ms = nowMs() - td0;
    const pp_tps = @as(f64, @floatFromInt(prompt_ids.len)) / (pp_ms / 1e3);
    const tg_tps = @as(f64, @floatFromInt(n_gen - 1)) / (tg_ms / 1e3);
    const avg_step = step_ms_sum / @as(f64, @floatFromInt(n_gen - 1));
    const avg_enc = encode_ms_sum / @as(f64, @floatFromInt(n_gen - 1));

    std.debug.print("Token match: {d}/{d}", .{ matched, n_gen });
    if (first_diff) |fd| {
        std.debug.print(" (first diff at step {d}: got {d} '{s}', want {d} '{s}')", .{
            fd,
            got_first_diff,
            tok.decodeToken(got_first_diff),
            want_first_diff,
            tok.decodeToken(want_first_diff),
        });
    }
    std.debug.print("\n", .{});
    std.debug.print("m4lm Prefill: {d} tok in {d:.2} ms ({d:.1} tok/s)\n", .{ prompt_ids.len, pp_ms, pp_tps });
    std.debug.print("m4lm Decode : {d} tok in {d:.2} ms ({d:.1} tok/s, {d:.2} ms/tok, host encode {d:.3} ms)\n", .{
        n_gen - 1, tg_ms, tg_tps, avg_step, avg_enc,
    });
    return if (matched == n_gen) 0 else 1;
}

fn streamTokCb(raw_ctx: ?*anyopaque, tok_id: u32) void {
    const tok: *const Tokenizer = @ptrCast(@alignCast(raw_ctx.?));
    writeStdout(tok.decodeToken(tok_id));
}

fn cmdGenerate(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var prompt: []const u8 = "Explain how a GPU tensor core multiplies two matrices in clear, concise technical terms.";
    var max_tokens: usize = 256;
    var max_context: u32 = 8192;
    var use_chat = true;
    var think = false;
    var use_spec = true;
    var draft_dir: ?[]const u8 = null;
    var cross_draft_dir: ?[]const u8 = null;
    var sp = samp_mod.SamplingParams{};
    var image_paths: std.ArrayList([]const u8) = .empty;
    defer image_paths.deinit(a);

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if ((std.mem.eql(u8, rest[i], "--prompt") or std.mem.eql(u8, rest[i], "-p")) and i + 1 < rest.len) {
            i += 1;
            prompt = rest[i];
        } else if ((std.mem.eql(u8, rest[i], "--image") or std.mem.eql(u8, rest[i], "-i")) and i + 1 < rest.len) {
            i += 1;
            try image_paths.append(a, rest[i]);
        } else if ((std.mem.eql(u8, rest[i], "--max-tokens") or std.mem.eql(u8, rest[i], "-m")) and i + 1 < rest.len) {
            i += 1;
            max_tokens = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--max-context") and i + 1 < rest.len) {
            i += 1;
            max_context = try std.fmt.parseInt(u32, rest[i], 10);
        } else if ((std.mem.eql(u8, rest[i], "--draft") or std.mem.eql(u8, rest[i], "-d")) and i + 1 < rest.len) {
            i += 1;
            draft_dir = rest[i];
        } else if (std.mem.eql(u8, rest[i], "--cross-draft") and i + 1 < rest.len) {
            i += 1;
            cross_draft_dir = rest[i];
        } else if ((std.mem.eql(u8, rest[i], "--temp") or std.mem.eql(u8, rest[i], "--temperature")) and i + 1 < rest.len) {
            i += 1;
            sp.temperature = try std.fmt.parseFloat(f32, rest[i]);
        } else if (std.mem.eql(u8, rest[i], "--top-p") and i + 1 < rest.len) {
            i += 1;
            sp.top_p = try std.fmt.parseFloat(f32, rest[i]);
        } else if (std.mem.eql(u8, rest[i], "--top-k") and i + 1 < rest.len) {
            i += 1;
            sp.top_k = try std.fmt.parseInt(u32, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--min-p") and i + 1 < rest.len) {
            i += 1;
            sp.min_p = try std.fmt.parseFloat(f32, rest[i]);
        } else if (std.mem.eql(u8, rest[i], "--seed") and i + 1 < rest.len) {
            i += 1;
            sp.seed = try std.fmt.parseInt(u64, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--no-chat")) {
            use_chat = false;
        } else if (std.mem.eql(u8, rest[i], "--think")) {
            think = true;
        } else if (std.mem.eql(u8, rest[i], "--no-spec")) {
            use_spec = false;
        }
    }

    const t0 = nowMs();
    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = max_context, .capture_taps = (draft_dir != null and sp.temperature <= 1e-6 and image_paths.items.len == 0) });
    defer m.deinit();

    if (image_paths.items.len > 0) {
        var ve = (try vision_mod.VisionEncoder.tryInit(a, ctx, model_dir)) orelse {
            std.debug.print("Error: model at '{s}' has no vision encoder weights.\n", .{model_dir});
            return 2;
        };
        defer ve.deinit();
        const load_ms = nowMs() - t0;

        const tv0 = nowMs();
        var vis_embs: std.ArrayList(u16) = .empty;
        defer vis_embs.deinit(a);
        var img_counts: std.ArrayList(u32) = .empty;
        defer img_counts.deinit(a);
        for (image_paths.items) |ip| {
            const enc_img = try ve.encodeImage(a, ip);
            defer enc_img.deinit(a);
            try vis_embs.appendSlice(a, enc_img.embs);
            try img_counts.append(a, enc_img.num_tokens);
        }
        const vis_ms = nowMs() - tv0;

        const prompt_ids = try vision_mod.buildMultimodalPrompt(a, &tok, ve.c, prompt, img_counts.items, use_chat, think);
        defer a.free(prompt_ids);
        if (prompt_ids.len >= m.max_context) {
            std.debug.print("Error: multimodal prompt ({d} tokens) exceeds --max-context ({d}). Increase --max-context.\n", .{ prompt_ids.len, m.max_context });
            return 2;
        }
        max_tokens = @min(max_tokens, @as(usize, m.max_context) - prompt_ids.len);

        var rng = samp_mod.Rng.init(sp.seed);
        const tp0 = nowMs();
        var next_tok = try m.prefillWithEmbeddings(prompt_ids, vis_embs.items, ve.c.image_token_id, 128);
        if (sp.temperature > 1e-6) next_tok = try m.sampleRow(0, sp, &rng);
        const pp_ms = nowMs() - tp0;

        var generated: usize = 0;
        const td0 = nowMs();
        while (generated < max_tokens) {
            if (m.c.isEos(next_tok)) break;
            writeStdout(tok.decodeToken(next_tok));
            generated += 1;
            if (generated >= max_tokens) break;
            if (sp.temperature > 1e-6) {
                _ = try m.stepWithLogits(next_tok);
                next_tok = try m.sampleRow(0, sp, &rng);
            } else {
                next_tok = try m.step(next_tok);
            }
        }
        const tg_ms = nowMs() - td0;
        writeStdout("\n");

        const pp_tps = @as(f64, @floatFromInt(prompt_ids.len)) / @max(pp_ms / 1e3, 1e-6);
        const tg_tps = if (generated > 1) @as(f64, @floatFromInt(generated - 1)) / @max(tg_ms / 1e3, 1e-6) else 0;
        std.debug.print("==========\nLoad: {d:.0} ms | Vision ({d} img, {d} tok): {d:.1} ms | Prompt: {d} tok ({d:.1} tok/s) | Generation: {d} tok ({d:.1} tok/s)\n", .{
            load_ms, image_paths.items.len, vis_embs.items.len / ve.c.out_hidden_size, vis_ms, prompt_ids.len, pp_tps, generated, tg_tps,
        });
        return 0;
    }

    const full_prompt = if (use_chat) try tok.formatChat(a, prompt, think) else try a.dupe(u8, prompt);
    defer a.free(full_prompt);
    const prompt_ids = try tok.encode(a, full_prompt);
    defer a.free(prompt_ids);
    if (prompt_ids.len == 0) {
        std.debug.print("Error: prompt encoded to 0 tokens.\n", .{});
        return 2;
    }
    if (prompt_ids.len >= m.max_context) {
        std.debug.print("Error: prompt ({d} tokens) exceeds --max-context ({d}). Increase --max-context.\n", .{ prompt_ids.len, m.max_context });
        return 2;
    }
    max_tokens = @min(max_tokens, @as(usize, m.max_context) - prompt_ids.len);

    if (cross_draft_dir) |cdir| {
        var dm = try Model.init(a, ctx, cdir, .{ .max_context = 8192 });
        defer dm.deinit();
        const load_ms = nowMs() - t0;
        var dec = try cross_mod.CrossModelDecoder.init(a, &m, &dm, 5);
        defer dec.deinit();
        const tp0 = nowMs();
        const stats = if (sp.temperature <= 1e-6)
            try dec.generate(prompt_ids, max_tokens, true, null, a, streamTokCb, &tok)
        else
            try dec.generateSampled(prompt_ids, max_tokens, true, sp, null, a, streamTokCb, &tok);
        const total_ms = nowMs() - tp0;
        writeStdout("\n");
        std.debug.print("==========\nLoad (target+cross-draft): {d:.0} ms | Prompt: {d} tok | Gen: {d} tok ({d:.1} tok/s, {d:.2} tok/rnd, {d}/{d} drafts accepted, temp={d:.2}, wall {d:.0} ms)\n", .{
            load_ms, prompt_ids.len, stats.tokens_generated, stats.tokPerSec(), stats.avgTokensPerRound(), stats.drafts_accepted, stats.drafts_proposed, sp.temperature, total_ms,
        });
        return 0;
    }

    if (sp.temperature <= 1e-6) {
        if (draft_dir) |ddir| {
            var dr = try draft_mod.Drafter.init(a, ctx, &m, ddir);
            defer dr.deinit();
            const load_ms = nowMs() - t0;
            var dec = try draft_mod.HybridDFlashDecoder.init(a, &dr, 15);
            defer dec.deinit();
            const tp0 = nowMs();
            const stats = try dec.generate(prompt_ids, max_tokens, true, null, a, streamTokCb, &tok);
            const total_ms = nowMs() - tp0;
            writeStdout("\n");
            std.debug.print("==========\nLoad (target+DFlash2): {d:.0} ms | Prompt: {d} tok | Gen: {d} tok ({d:.1} tok/s, {d:.2} tok/rnd, {d}/{d} drafts accepted, wall {d:.0} ms)\n", .{
                load_ms, prompt_ids.len, stats.tokens_generated, stats.tokPerSec(), stats.avgTokensPerRound(), stats.drafts_accepted, stats.drafts_proposed, total_ms,
            });
            return 0;
        }

        const load_ms = nowMs() - t0;
        if (use_spec and m.c.arch != .nemotron_h) {
            var dec = try spec_mod.AdaptiveSpecDecoder.init(a, 127);
            defer dec.deinit();
            const tp0 = nowMs();
            const stats = try dec.generate(&m, prompt_ids, max_tokens, true, null, a, streamTokCb, &tok);
            const total_ms = nowMs() - tp0;
            writeStdout("\n");
            std.debug.print("==========\nLoad: {d:.0} ms | Prompt: {d} tok | Gen: {d} tok ({d:.1} tok/s, {d:.2} tok/rnd, {d}/{d} drafts accepted, wall {d:.0} ms)\n", .{
                load_ms, prompt_ids.len, stats.tokens_generated, stats.tokPerSec(), stats.avgTokensPerRound(), stats.drafts_accepted, stats.drafts_proposed, total_ms,
            });
            return 0;
        }
    }

    const load_ms = nowMs() - t0;
    var rng = samp_mod.Rng.init(sp.seed);

    const tp0 = nowMs();
    var next_tok = try m.prefill(prompt_ids, 128);
    if (sp.temperature > 1e-6) next_tok = try m.sampleRow(0, sp, &rng);
    const pp_ms = nowMs() - tp0;

    var generated: usize = 0;
    const td0 = nowMs();
    while (generated < max_tokens) {
        if (m.c.isEos(next_tok)) break;
        writeStdout(tok.decodeToken(next_tok));
        generated += 1;
        if (generated >= max_tokens) break;
        if (sp.temperature > 1e-6) {
            _ = try m.stepWithLogits(next_tok);
            next_tok = try m.sampleRow(0, sp, &rng);
        } else {
            next_tok = try m.step(next_tok);
        }
    }
    const tg_ms = nowMs() - td0;
    writeStdout("\n");

    const pp_tps = @as(f64, @floatFromInt(prompt_ids.len)) / (pp_ms / 1e3);
    const tg_tps = if (generated > 1) @as(f64, @floatFromInt(generated - 1)) / (tg_ms / 1e3) else 0;
    std.debug.print("==========\nLoad: {d:.0} ms | Prompt: {d} tok ({d:.1} tok/s) | Generation: {d} tok ({d:.1} tok/s, temp={d:.2}, top_p={d:.2}, top_k={d}, min_p={d:.2}, seed={d})\n", .{
        load_ms, prompt_ids.len, pp_tps, generated, tg_tps, sp.temperature, sp.top_p, sp.top_k, sp.min_p, sp.seed,
    });
    return 0;
}


fn cmdBench(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var prompt_len: usize = 128;
    var gen_tokens: usize = 128;
    var warmup: usize = 16;
    var sweep_apis = false;

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--prompt-len") and i + 1 < rest.len) {
            i += 1;
            prompt_len = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--tokens") and i + 1 < rest.len) {
            i += 1;
            gen_tokens = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--warmup") and i + 1 < rest.len) {
            i += 1;
            warmup = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--sweep-apis")) {
            sweep_apis = true;
        } else if (std.mem.eql(u8, rest[i], "--no-qmv")) {
            Model.use_qmv_1 = false;
        } else if (std.mem.eql(u8, rest[i], "--no-fused")) {
            Model.use_fused_1 = false;
        }
    }

    var m = try Model.init(a, ctx, model_dir, .{ .max_context = @intCast(prompt_len + gen_tokens + 256) });
    defer m.deinit();

    const prompt = try a.alloc(u32, prompt_len);
    defer a.free(prompt);
    for (prompt, 0..) |*t, idx| t.* = @intCast((idx * 131 + 42) % 30000);

    if (sweep_apis) {
        m.auto_api = false;
        const apis = [_]mtl.BackendApi{ .metal4_device, .metal4_none, .metal3_serial, .metal3_concurrent };
        for (apis) |api| {
            ctx.api = api;
            m.reset();
            var t_w = try m.prefill(prompt, 128);
            for (0..8) |_| t_w = try m.step(t_w);
            m.reset();
            const tp0 = nowMs();
            var tok = try m.prefill(prompt, 128);
            const pp_ms = nowMs() - tp0;
            const td0 = nowMs();
            for (0..gen_tokens) |_| tok = try m.step(tok);
            const tg_ms = nowMs() - td0;
            std.debug.print("API {s:<18}: pp {d:6.1} tok/s ({d:5.2} ms) | tg {d:6.1} tok/s ({d:5.2} ms/tok) last={d}\n", .{
                @tagName(api),
                @as(f64, @floatFromInt(prompt_len)) / (pp_ms / 1e3),
                pp_ms,
                @as(f64, @floatFromInt(gen_tokens)) / (tg_ms / 1e3),
                tg_ms / @as(f64, @floatFromInt(gen_tokens)),
                tok,
            });
        }
        return 0;
    }

    // Warmup
    var tok = try m.prefill(prompt, 128);
    for (0..warmup) |_| tok = try m.step(tok);

    // Timed run
    m.reset();
    const tp0 = nowMs();
    tok = try m.prefill(prompt, 128);
    const pp_ms = nowMs() - tp0;

    const td0 = nowMs();
    for (0..gen_tokens) |_| tok = try m.step(tok);
    const tg_ms = nowMs() - td0;

    std.debug.print("Device : {s} ({d:.1} MiB weights)\n", .{ ctx.deviceName(), @as(f64, @floatFromInt(m.w.total_bytes)) / (1024.0 * 1024.0) });
    std.debug.print("Prefill: {d} tok in {d:.2} ms ({d:.1} tok/s)\n", .{ prompt_len, pp_ms, @as(f64, @floatFromInt(prompt_len)) / (pp_ms / 1e3) });
    std.debug.print("Decode : {d} tok in {d:.2} ms ({d:.1} tok/s, {d:.2} ms/tok)\n", .{
        gen_tokens, tg_ms, @as(f64, @floatFromInt(gen_tokens)) / (tg_ms / 1e3), tg_ms / @as(f64, @floatFromInt(gen_tokens)),
    });
    return 0;
}


fn readU32Json(a: std.mem.Allocator, path: []const u8) ![]u32 {
    const bytes = try st.mapFile(a, path);
    defer st.unmapFile(bytes);
    const parsed = try std.json.parseFromSlice([]u32, a, bytes, .{});
    defer parsed.deinit();
    return a.dupe(u32, parsed.value);
}

fn cmdGolden(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, prompt_path: []const u8, expected_path: []const u8, rest: []const [:0]const u8) !u8 {
    var limit: usize = 1024;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--limit") and i + 1 < rest.len) {
            i += 1;
            limit = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--no-qmv")) {
            Model.use_qmv_1 = false;
        } else if (std.mem.eql(u8, rest[i], "--no-fused")) {
            Model.use_fused_1 = false;
        }
    }

    const prompt = try readU32Json(a, prompt_path);
    defer a.free(prompt);
    const expected = try readU32Json(a, expected_path);
    defer a.free(expected);
    const n_check = @min(limit, expected.len);

    const t0 = nowMs();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = @intCast(prompt.len + n_check + 256) });
    defer m.deinit();
    std.debug.print("Loaded on {s} ({d:.1} MiB) in {d:.0} ms\n", .{
        ctx.deviceName(),
        @as(f64, @floatFromInt(m.w.total_bytes)) / (1024.0 * 1024.0),
        nowMs() - t0,
    });

    const tp0 = nowMs();
    var tok = try m.prefill(prompt, 128);
    const pp_ms = nowMs() - tp0;

    var matched: usize = 0;
    var first_diff: ?usize = null;
    const td0 = nowMs();
    for (0..n_check) |step_i| {
        if (tok == expected[step_i]) {
            matched += 1;
        } else if (first_diff == null) {
            first_diff = step_i;
            std.debug.print("First diff at step {d}: got {d}, expected {d}\n", .{ step_i, tok, expected[step_i] });
        }
        if (step_i + 1 < n_check) tok = try m.step(tok);
    }
    const tg_ms = nowMs() - td0;
    std.debug.print("Golden match: {d}/{d}\n", .{ matched, n_check });
    std.debug.print("Prefill: {d} tok in {d:.1} ms ({d:.1} tok/s)\n", .{
        prompt.len, pp_ms, @as(f64, @floatFromInt(prompt.len)) / (pp_ms / 1e3),
    });
    std.debug.print("Decode : {d} tok in {d:.1} ms ({d:.2} tok/s, {d:.2} ms/tok)\n", .{
        n_check - 1, tg_ms, @as(f64, @floatFromInt(n_check - 1)) / (tg_ms / 1e3), tg_ms / @as(f64, @floatFromInt(n_check - 1)),
    });
    return if (matched == n_check) 0 else 1;
}

fn cmdSpecCheck(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8) !u8 {
    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 4096 });
    defer m.deinit();

    // Test 1: Non-contiguous branching tree verification parity vs plain c=1 decode
    // Use qk_16 for c=1 reference so both paths use identical tensor-op fp32 reduction order.
    const prev_qmv = Model.use_qmv_1;
    Model.use_qmv_1 = false;
    defer Model.use_qmv_1 = prev_qmv;

    const tree_prompt_str = try tok.formatChat(a, "Count from 1 to 20 separated by commas.", false);
    defer a.free(tree_prompt_str);
    const tree_prompt = try tok.encode(a, tree_prompt_str);
    defer a.free(tree_prompt);

    var ref_toks: [24]u32 = undefined;
    ref_toks[0] = try m.prefill(tree_prompt, 128);
    for (1..24) |i| ref_toks[i] = try m.step(ref_toks[i - 1]);

    // Replay with a 12-node branching tree where the true path jumps across non-contiguous nodes [0, 2, 5, 9]
    m.reset();
    var pending = try m.prefill(tree_prompt, 128);
    var tree_ok: usize = 0;
    if (pending == ref_toks[0]) tree_ok += 1;

    var cursor: usize = 1;
    while (cursor + 4 <= 24) {
        // Tree topology:
        //   0 (pending)
        //   ├─ 1 (decoy), 2 (TRUE ref[cursor]), 3 (decoy)
        //   │  ├─ 4 (decoy), 5 (TRUE ref[cursor+1]), 6 (decoy)
        //   │  │  ├─ 7 (decoy), 8 (decoy), 9 (TRUE ref[cursor+2])
        //   │  │  │  └─ 10 (decoy), 11 (TRUE ref[cursor+3])
        const parents = [12]i32{ -1, 0, 0, 0, 2, 2, 2, 5, 5, 5, 9, 9 };
        const nodes = [12]u32{
            pending,
            100,
            ref_toks[cursor],
            101,
            102,
            ref_toks[cursor + 1],
            103,
            104,
            105,
            ref_toks[cursor + 2],
            106,
            ref_toks[cursor + 3],
        };
        const picks = try m.verifyTree(&nodes, &parents, true);
        const keep = [4]u32{ 0, 2, 5, 9 };
        if (picks[0] == ref_toks[cursor]) tree_ok += 1;
        if (picks[2] == ref_toks[cursor + 1]) tree_ok += 1;
        if (picks[5] == ref_toks[cursor + 2]) tree_ok += 1;
        if (picks[9] == ref_toks[cursor + 3]) tree_ok += 1;
        try m.commitKeep(&keep);
        pending = picks[9];
        cursor += 4;
    }
    std.debug.print("Test 1 (Branching 12-node tree + non-contiguous KV/GDN keep): {d}/{d} EXACT\n", .{ tree_ok, cursor });

    // Test 2: Adaptive Suffix/Copy Lookahead on a realistic verbatim-copy / refactor prompt!
    Model.use_qmv_1 = prev_qmv;
    const copy_prompt_raw =
        \\Repeat the following technical passage word for word without changing a single word or punctuation mark:
        \\
        \\Modern Apple Silicon GPUs execute matrix multiplications on dedicated hardware tensor units arranged in sixteen-row blocks across forty compute cores. When a language model decodes one token at a time, fifteen of those sixteen hardware rows remain completely idle while the entire weight matrix streams from unified memory. By verifying sixteen to sixty-four candidate tokens simultaneously in a single forward pass, a bare Metal 4 engine converts wasted hardware rows directly into lossless token throughput without altering a single output bit.
    ;
    const copy_chat = try tok.formatChat(a, copy_prompt_raw, false);
    defer a.free(copy_chat);
    const copy_ids = try tok.encode(a, copy_chat);
    defer a.free(copy_ids);

    // Warmup
    m.reset();
    _ = try m.prefill(copy_ids, 128);

    // Plain c=1 reference (with qmv_1 disabled for exact fp32 parity comparison, plus timed with qmv_1 enabled!)
    m.reset();
    Model.use_qmv_1 = true;
    var plain_ids: std.ArrayList(u32) = .empty;
    defer plain_ids.deinit(a);
    const tp_plain0 = nowMs();
    var ptok = try m.prefill(copy_ids, 128);
    const td_plain0 = nowMs();
    try plain_ids.append(a, ptok);
    for (1..110) |_| {
        if (m.c.isEos(ptok)) break;
        ptok = try m.step(ptok);
        if (m.c.isEos(ptok)) break;
        try plain_ids.append(a, ptok);
    }
    const plain_ms = nowMs() - td_plain0;
    const plain_total = nowMs() - tp_plain0;
    const plain_tps = @as(f64, @floatFromInt(plain_ids.items.len)) / (plain_ms / 1e3);

    // Speculative Adaptive Copy Lookahead run
    m.reset();
    var dec = try spec_mod.AdaptiveSpecDecoder.init(a, 127);
    defer dec.deinit();
    var spec_ids: std.ArrayList(u32) = .empty;
    defer spec_ids.deinit(a);
    const spec_stats = try dec.generate(&m, copy_ids, 110, true, &spec_ids, a, null, null);

    var exact_match = (plain_ids.items.len == spec_ids.items.len);
    var match_cnt: usize = 0;
    for (0..@min(plain_ids.items.len, spec_ids.items.len)) |k| {
        if (plain_ids.items[k] == spec_ids.items[k]) match_cnt += 1 else exact_match = false;
    }

    std.debug.print("Test 2 (Adaptive Copy Lookahead vs Plain c=1 on copy passage):\n", .{});
    std.debug.print("  Token identity : {d}/{d} ({s})\n", .{ match_cnt, plain_ids.items.len, if (exact_match) "100% BIT-EXACT" else "DIFF" });
    std.debug.print("  Plain c=1      : {d} tok in {d:.2} ms ({d:.1} tok/s, wall {d:.1} ms)\n", .{ plain_ids.items.len, plain_ms, plain_tps, plain_total });
    std.debug.print("  m4lm Speculative: {d} tok in {d:.2} ms ({d:.1} tok/s, {d:.2} tok/rnd, {d}/{d} drafts accepted) -> {d:.2}x speedup!\n", .{
        spec_stats.tokens_generated,
        spec_stats.decode_ms,
        spec_stats.tokPerSec(),
        spec_stats.avgTokensPerRound(),
        spec_stats.drafts_accepted,
        spec_stats.drafts_proposed,
        spec_stats.tokPerSec() / plain_tps,
    });

    return if (tree_ok == cursor and exact_match) 0 else 1;
}

fn cmdDflashCheck(a: std.mem.Allocator, ctx: *mtl.Context, target_dir: []const u8, drafter_dir: []const u8) !u8 {
    const t0 = nowMs();
    var tok = try Tokenizer.load(a, target_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, target_dir, .{ .max_context = 4096, .capture_taps = true });
    defer m.deinit();
    const t_target = nowMs() - t0;

    const td_load0 = nowMs();
    var dr = try draft_mod.Drafter.init(a, ctx, &m, drafter_dir);
    defer dr.deinit();
    const t_draft = nowMs() - td_load0;
    std.debug.print("Loaded 27B ({d:.0} ms) + DFlash2 4-bit ({d:.0} ms) on {s}\n", .{ t_target, t_draft, ctx.deviceName() });

    var dec = try draft_mod.HybridDFlashDecoder.init(a, &dr, 15);
    defer dec.deinit();

    const Case = struct { name: []const u8, prompt: []const u8, max_toks: usize, check_exact: bool };
    const cases = [_]Case{
        .{
            .name = "open_prose",
            .prompt = "Explain how a GPU tensor core multiplies two matrices in clear, concise technical terms.",
            .max_toks = 96,
            .check_exact = true,
        },
        .{
            .name = "greenfield_code",
            .prompt = "Write a clean, idiomatic TypeScript ring buffer class with push, pop, peek, length, and iterator support.",
            .max_toks = 128,
            .check_exact = false,
        },
        .{
            .name = "copy_passage",
            .prompt = "Repeat the following technical passage word for word without changing a single word or punctuation mark:\n\nModern Apple Silicon GPUs execute matrix multiplications on dedicated hardware tensor units arranged in sixteen-row blocks across forty compute cores. When a language model decodes one token at a time, fifteen of those sixteen hardware rows remain completely idle while the entire weight matrix streams from unified memory. By verifying sixteen to sixty-four candidate tokens simultaneously in a single forward pass, a bare Metal engine converts wasted hardware rows directly into lossless token throughput without altering a single output bit.",
            .max_toks = 110,
            .check_exact = true,
        },
    };

    var all_ok = true;
    for (cases) |cs| {
        const chat = try Tokenizer.formatChatPrompt(a, cs.prompt, false);
        defer a.free(chat);
        const ids = try tok.encode(a, chat);
        defer a.free(ids);

        var spec_ids: std.ArrayList(u32) = .empty;
        defer spec_ids.deinit(a);
        const stats = try dec.generate(ids, cs.max_toks, true, &spec_ids, a, null, null);

        if (cs.check_exact) {
            // Compare against plain c=1 using qk_16 for bit-exact reduction order
            const prev_qmv = Model.use_qmv_1;
            Model.use_qmv_1 = false;
            defer Model.use_qmv_1 = prev_qmv;

            m.reset();
            var ref_ids: std.ArrayList(u32) = .empty;
            defer ref_ids.deinit(a);
            const t_p0 = nowMs();
            var cur = try m.prefill(ids, 128);
            const t_d0 = nowMs();
            if (!m.c.isEos(cur)) {
                try ref_ids.append(a, cur);
                while (ref_ids.items.len < cs.max_toks) {
                    cur = try m.step(cur);
                    if (m.c.isEos(cur)) break;
                    try ref_ids.append(a, cur);
                }
            }
            const ref_ms = nowMs() - t_d0;
            _ = t_p0;
            const ref_tps = @as(f64, @floatFromInt(ref_ids.items.len)) / (ref_ms / 1e3);

            var exact = (ref_ids.items.len == spec_ids.items.len);
            var match_cnt: usize = 0;
            for (0..@min(ref_ids.items.len, spec_ids.items.len)) |k| {
                if (ref_ids.items[k] == spec_ids.items[k]) match_cnt += 1 else exact = false;
            }
            if (!exact) all_ok = false;
            std.debug.print("  [{s:<16}] DFlash2+Copy: {d:6.1} tok/s ({d:.2} tok/rnd, {d} rnds, last draft {d:.2} ms) vs Plain: {d:5.1} tok/s -> {d:.2}x | Parity: {d}/{d} ({s})\n", .{
                cs.name,
                stats.tokPerSec(),
                stats.avgTokensPerRound(),
                stats.rounds,
                dr.last_draft_ms,
                ref_tps,
                stats.tokPerSec() / ref_tps,
                match_cnt,
                ref_ids.items.len,
                if (exact) "100% BIT-EXACT" else "DIFF",
            });
        } else {
            std.debug.print("  [{s:<16}] DFlash2+Copy: {d:6.1} tok/s ({d:.2} tok/rnd, {d} rnds, last draft {d:.2} ms, {d} tok)\n", .{
                cs.name,
                stats.tokPerSec(),
                stats.avgTokensPerRound(),
                stats.rounds,
                dr.last_draft_ms,
                stats.tokens_generated,
            });
        }
    }
    return if (all_ok) 0 else 1;
}

fn cmdTimeline(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, _: []const [:0]const u8) !u8 {
    ctx.api = .metal4_device;
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 2048 });
    defer m.deinit();
    m.auto_api = false;

    var prompt: [32]u32 = undefined;
    for (&prompt, 0..) |*t, idx| t.* = @intCast(100 + idx);
    var tok = try m.prefill(&prompt, 32);
    for (0..10) |_| tok = try m.step(tok);

    var tl = try mtl.Timeline.init(a, ctx, 2048);
    defer tl.deinit(a);
    ctx.timeline = &tl;
    defer ctx.timeline = null;

    // Aggregate over 10 steps
    const reps = 10;
    var bucket_names: std.ArrayList([]const u8) = .empty;
    defer bucket_names.deinit(a);
    var bucket_us: std.ArrayList(f64) = .empty;
    defer bucket_us.deinit(a);
    var bucket_cnt: std.ArrayList(u32) = .empty;
    defer bucket_cnt.deinit(a);

    const freq = tl.frequencyHz();
    const ticks = try a.alloc(u64, 2048);
    defer a.free(ticks);

    for (0..reps) |_| {
        tl.reset();
        tok = try m.step(tok);
        tl.readTicks(ticks);
        for (1..tl.count) |i| {
            const name = std.mem.sliceTo(&tl.names[i], 0);
            const dt_us = @as(f64, @floatFromInt(ticks[i] - ticks[i - 1])) * 1e6 / freq;
            var found: ?usize = null;
            for (bucket_names.items, 0..) |bn, bi| {
                if (std.mem.eql(u8, bn, name)) {
                    found = bi;
                    break;
                }
            }
            if (found) |bi| {
                bucket_us.items[bi] += dt_us;
                bucket_cnt.items[bi] += 1;
            } else {
                try bucket_names.append(a, try a.dupe(u8, name));
                try bucket_us.append(a, dt_us);
                try bucket_cnt.append(a, 1);
            }
        }
    }

    var total_us: f64 = 0;
    std.debug.print("GPU Timeline (averaged over {d} decode steps, {d} stamps/step):\n", .{ reps, tl.count });
    for (bucket_names.items, 0..) |bn, bi| {
        const ms = (bucket_us.items[bi] / reps) / 1000.0;
        const cnt = bucket_cnt.items[bi] / reps;
        total_us += bucket_us.items[bi] / reps;
        std.debug.print("  {s:<22} x{d:<4}  {d:6.3} ms  ({d:5.1} us/stage)\n", .{ bn, cnt, ms, (bucket_us.items[bi] / @as(f64, @floatFromInt(bucket_cnt.items[bi]))) });
    }
    std.debug.print("  TOTAL GPU CLOCK:               {d:6.3} ms\n", .{total_us / 1000.0});
    return 0;
}


fn cmdServe(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var host: []const u8 = "127.0.0.1";
    var port: u16 = 8080;
    var model_name: []const u8 = std.fs.path.basename(model_dir);
    var draft_dir: ?[]const u8 = null;
    var cross_draft_dir: ?[]const u8 = null;
    var max_streams: usize = 16;
    var max_context: u32 = 8192;

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--host") and i + 1 < rest.len) {
            i += 1;
            host = rest[i];
        } else if ((std.mem.eql(u8, rest[i], "--port") or std.mem.eql(u8, rest[i], "-p")) and i + 1 < rest.len) {
            i += 1;
            port = try std.fmt.parseInt(u16, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--name") and i + 1 < rest.len) {
            i += 1;
            model_name = rest[i];
        } else if ((std.mem.eql(u8, rest[i], "--draft") or std.mem.eql(u8, rest[i], "-d")) and i + 1 < rest.len) {
            i += 1;
            draft_dir = rest[i];
        } else if (std.mem.eql(u8, rest[i], "--cross-draft") and i + 1 < rest.len) {
            i += 1;
            cross_draft_dir = rest[i];
        } else if (std.mem.eql(u8, rest[i], "--max-streams") and i + 1 < rest.len) {
            i += 1;
            max_streams = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--max-context") and i + 1 < rest.len) {
            i += 1;
            max_context = try std.fmt.parseInt(u32, rest[i], 10);
        }
    }

    const t0 = nowMs();
    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = max_context, .capture_taps = (draft_dir != null) });
    defer m.deinit();

    var dr: ?draft_mod.Drafter = null;
    var dr_dec: ?draft_mod.HybridDFlashDecoder = null;
    var cross_m: ?Model = null;
    var cross_dec: ?cross_mod.CrossModelDecoder = null;

    defer {
        if (dr_dec) |*x| x.deinit();
        if (dr) |*x| x.deinit();
        if (cross_dec) |*x| x.deinit();
        if (cross_m) |*x| x.deinit();
    }

    if (draft_dir) |dd| {
        dr = try draft_mod.Drafter.init(a, ctx, &m, dd);
        dr_dec = try draft_mod.HybridDFlashDecoder.init(a, &dr.?, 15);
    } else if (cross_draft_dir) |cd| {
        cross_m = try Model.init(a, ctx, cd, .{ .max_context = max_context });
        cross_dec = try cross_mod.CrossModelDecoder.init(a, &m, &cross_m.?, 5);
    }

    var ve = vision_mod.VisionEncoder.tryInit(a, ctx, model_dir) catch null;
    defer if (ve) |*v| v.deinit();

    std.debug.print("Loaded model on {s} in {d:.0} ms ({d:.1} MiB weights, vision={})\n", .{
        ctx.deviceName(), nowMs() - t0, @as(f64, @floatFromInt(m.w.total_bytes)) / (1024.0 * 1024.0), ve != null,
    });

    const dd_ptr: ?*draft_mod.HybridDFlashDecoder = if (dr_dec) |*x| x else null;
    const cd_ptr: ?*cross_mod.CrossModelDecoder = if (cross_dec) |*x| x else null;
    const ve_ptr: ?*vision_mod.VisionEncoder = if (ve) |*v| v else null;
    var srv = try srv_mod.ServerEngine.init(a, ctx, &m, &tok, model_name, dd_ptr, cd_ptr, ve_ptr, max_streams);
    defer srv.deinit();
    try srv.serveHost(host, port);
    return 0;
}


fn cmdCrossCheck(a: std.mem.Allocator, ctx: *mtl.Context, target_dir: []const u8, draft_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var k_steps: usize = 5;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--k") and i + 1 < rest.len) {
            i += 1;
            k_steps = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }

    const t0 = nowMs();
    var tok = try Tokenizer.load(a, target_dir);
    defer tok.deinit();
    var target = try Model.init(a, ctx, target_dir, .{ .max_context = 4096 });
    defer target.deinit();
    const t_target = nowMs() - t0;

    const td0 = nowMs();
    var draft = try Model.init(a, ctx, draft_dir, .{ .max_context = 4096 });
    defer draft.deinit();
    const t_draft = nowMs() - td0;
    std.debug.print("Loaded target ({d:.1} MiB, {d:.0} ms) + draft ({d:.1} MiB, {d:.0} ms) on {s} (K={d})\n", .{
        @as(f64, @floatFromInt(target.w.total_bytes)) / (1024.0 * 1024.0),
        t_target,
        @as(f64, @floatFromInt(draft.w.total_bytes)) / (1024.0 * 1024.0),
        t_draft,
        ctx.deviceName(),
        k_steps,
    });

    var dec = try cross_mod.CrossModelDecoder.init(a, &target, &draft, k_steps);
    defer dec.deinit();

    const Case = struct { name: []const u8, prompt: []const u8, max_toks: usize };
    const cases = [_]Case{
        .{
            .name = "open_prose",
            .prompt = "Explain how a GPU tensor core multiplies two matrices in clear, concise technical terms.",
            .max_toks = 128,
        },
        .{
            .name = "greenfield_code",
            .prompt = "Write a clean, idiomatic TypeScript ring buffer class with push, pop, peek, length, and iterator support.",
            .max_toks = 128,
        },
        .{
            .name = "copy_passage",
            .prompt = "Repeat the following technical passage word for word without changing a single word or punctuation mark:\n\nModern Apple Silicon GPUs execute matrix multiplications on dedicated hardware tensor units arranged in sixteen-row blocks across forty compute cores. When a language model decodes one token at a time, fifteen of those sixteen hardware rows remain completely idle while the entire weight matrix streams from unified memory. By verifying sixteen to sixty-four candidate tokens simultaneously in a single forward pass, a bare Metal engine converts wasted hardware rows directly into lossless token throughput without altering a single output bit.",
            .max_toks = 110,
        },
    };

    var all_ok = true;
    for (cases) |cs| {
        const chat = try Tokenizer.formatChatPrompt(a, cs.prompt, false);
        defer a.free(chat);
        const ids = try tok.encode(a, chat);
        defer a.free(ids);

        // Warmup first case once
        var spec_ids: std.ArrayList(u32) = .empty;
        defer spec_ids.deinit(a);
        const stats = try dec.generate(ids, cs.max_toks, true, &spec_ids, a, null, null);

        // Plain c=1 reference (with qk_16 for bit-exact fp32 parity verification, plus timed with qmv_1!)
        const prev_qmv = Model.use_qmv_1;
        Model.use_qmv_1 = false;
        target.reset();
        var ref_ids: std.ArrayList(u32) = .empty;
        defer ref_ids.deinit(a);
        var cur = try target.prefill(ids, 128);
        if (!target.c.isEos(cur)) {
            try ref_ids.append(a, cur);
            while (ref_ids.items.len < cs.max_toks) {
                cur = try target.step(cur);
                if (target.c.isEos(cur)) break;
                try ref_ids.append(a, cur);
            }
        }
        Model.use_qmv_1 = prev_qmv;

        // Timed plain c=1 with qmv_1
        target.reset();
        var cur2 = try target.prefill(ids, 128);
        const t_ref0 = nowMs();
        var n_ref: usize = 0;
        if (!target.c.isEos(cur2)) {
            n_ref = 1;
            while (n_ref < cs.max_toks) {
                cur2 = try target.step(cur2);
                if (target.c.isEos(cur2)) break;
                n_ref += 1;
            }
        }
        const ref_ms = nowMs() - t_ref0;
        const ref_tps = @as(f64, @floatFromInt(n_ref)) / (ref_ms / 1e3);

        var exact = (ref_ids.items.len == spec_ids.items.len);
        var match_cnt: usize = 0;
        for (0..@min(ref_ids.items.len, spec_ids.items.len)) |k| {
            if (ref_ids.items[k] == spec_ids.items[k]) match_cnt += 1 else exact = false;
        }
        if (!exact) all_ok = false;

        std.debug.print("  [{s:<16}] CrossDraft: {d:6.1} tok/s ({d:.2} tok/rnd, {d}/{d} acc) vs Plain c=1: {d:5.1} tok/s -> {d:.2}x | Parity: {d}/{d} ({s})\n", .{
            cs.name,
            stats.tokPerSec(),
            stats.avgTokensPerRound(),
            stats.drafts_accepted,
            stats.drafts_proposed,
            ref_tps,
            stats.tokPerSec() / ref_tps,
            match_cnt,
            ref_ids.items.len,
            if (exact) "100% BIT-EXACT" else "DIFF",
        });
    }

    // 4. Verify Exact Leviathan/Chen Rejection Sampling Math (200k Monte Carlo TVD) + Live temp=0.7 top_p=0.9 run!
    const tvd = try samp_mod.verifyRejectionExactness(a);
    const samp_prompt = try Tokenizer.formatChatPrompt(a, "Explain how a GPU tensor core multiplies two matrices in clear, concise technical terms.", false);
    defer a.free(samp_prompt);
    const samp_ids = try tok.encode(a, samp_prompt);
    defer a.free(samp_ids);
    const sp = samp_mod.SamplingParams{ .temperature = 0.7, .top_p = 0.9, .min_p = 0.05, .seed = 42 };
    const s_stats = try dec.generateSampled(samp_ids, 128, true, sp, null, a, null, null);
    std.debug.print("  [rejection_math  ] Leviathan/Chen 200k-draw TVD = {d:.5} ({s})\n", .{ tvd, if (tvd < 0.005) "EXACT PASS" else "FAIL" });
    std.debug.print("  [sampled T=0.7   ] CrossDraft: {d:6.1} tok/s ({d:.2} tok/rnd, {d}/{d} acc, {d} tok)\n", .{
        s_stats.tokPerSec(), s_stats.avgTokensPerRound(), s_stats.drafts_accepted, s_stats.drafts_proposed, s_stats.tokens_generated,
    });
    if (tvd >= 0.005) all_ok = false;

    return if (all_ok) 0 else 1;
}

fn cmdWaveBench(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var n_streams: usize = 16;
    var gen_tokens: usize = 128;

    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if ((std.mem.eql(u8, rest[i], "--streams") or std.mem.eql(u8, rest[i], "-n")) and i + 1 < rest.len) {
            i += 1;
            n_streams = try std.fmt.parseInt(usize, rest[i], 10);
        } else if ((std.mem.eql(u8, rest[i], "--tokens") or std.mem.eql(u8, rest[i], "-m")) and i + 1 < rest.len) {
            i += 1;
            gen_tokens = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }
    n_streams = @max(1, @min(n_streams, batch_mod.max_streams));

    const t0 = nowMs();
    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 4096 });
    defer m.deinit();
    var fe = try batch_mod.ForkBatchEngine.init(a, &m, n_streams, 512);
    defer fe.deinit();
    std.debug.print("Loaded model + {d}-stream zero-copy fork pool on {s} in {d:.0} ms ({d:.1} MiB weights)\n", .{
        n_streams, ctx.deviceName(), nowMs() - t0, @as(f64, @floatFromInt(m.w.total_bytes)) / (1024.0 * 1024.0),
    });

    const shared_context =
        \\<|im_start|>system
        \\You are an expert Apple Silicon GPU systems architect and compiler engineer. Answer concisely, accurately, and directly based on the technical document below.
        \\
        \\Document: Modern Apple Silicon GPUs combine wide unified memory bandwidth with dedicated 16-row hardware tensor units across dozens of compute cores. Hybrid language models alternate gated delta-net linear recurrence layers with grouped-query full attention layers. Linear recurrence layers maintain a fixed-size state matrix per head, making their memory footprint O(1) in sequence length, while full attention layers store key-value pairs only on every fourth layer. When multiple requests share a large system prompt or codebase context, prefilling the shared prefix once into read-only key-value buffers and forking only the small recurrent states eliminates redundant prefill work and unlocks full 16-row tensor utilization across concurrent decode streams.
        \\<|im_end|>
        \\<|im_start|>user
        \\
    ;
    const questions = [_][]const u8{
        "1. Summarize why hybrid models reduce KV cache memory.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "2. Explain what happens when decoding one token at a time.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "3. How does prefix forking save prefill compute?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "4. Compare recurrent state size versus KV cache growth.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "5. List three key hardware traits mentioned in the document.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "6. Why do 16 concurrent streams saturate tensor units?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "7. Rewrite the main idea in two crisp bullet points.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "8. What fraction of layers use full attention in this architecture?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "9. Give a one-paragraph executive overview.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "10. What data structures are shared read-only across streams?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "11. What state must be cloned per forked stream?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "12. Explain the trade-off between latency and throughput.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "13. Draft a concise API docstring for prefix-forked batching.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "14. Describe the memory complexity of gated delta-net layers.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "15. Why is unified memory important for Apple Silicon LLMs?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "16. Provide a short checklist for benchmarking batched decode.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    };

    const shared_ids = try tok.encode(a, shared_context);
    defer a.free(shared_ids);

    var delta_ids: [batch_mod.max_streams][]u32 = undefined;
    var specs: [batch_mod.max_streams]batch_mod.StreamSpec = undefined;
    for (0..n_streams) |s| {
        delta_ids[s] = try tok.encode(a, questions[s % questions.len]);
        specs[s] = .{ .delta_tokens = delta_ids[s], .max_tokens = gen_tokens, .sampling = .{} };
    }
    defer for (0..n_streams) |s| a.free(delta_ids[s]);

    // 1. Standalone parity reference for stream 0 (using qk_16 so reduction order matches batched decode)
    const full0 = try a.alloc(u32, shared_ids.len + delta_ids[0].len);
    defer a.free(full0);
    @memcpy(full0[0..shared_ids.len], shared_ids);
    @memcpy(full0[shared_ids.len..], delta_ids[0]);

    const prev_qmv = Model.use_qmv_1;
    const prev_split = Model.use_split_k;
    Model.use_qmv_1 = false;
    Model.use_split_k = false;
    m.reset();
    var ref0: std.ArrayList(u32) = .empty;
    defer ref0.deinit(a);
    var cur = try m.prefill(full0, 128);
    if (!m.c.isEos(cur)) {
        try ref0.append(a, cur);
        while (ref0.items.len < gen_tokens) {
            cur = try m.step(cur);
            if (m.c.isEos(cur)) break;
            try ref0.append(a, cur);
        }
    }
    Model.use_qmv_1 = prev_qmv;
    Model.use_split_k = prev_split;

    // Solo c=1 speed baseline (with qmv_1 enabled)
    m.reset();
    var cur_s = try m.prefill(full0, 128);
    const t_solo0 = nowMs();
    var solo_cnt: usize = 0;
    if (!m.c.isEos(cur_s)) {
        solo_cnt = 1;
        while (solo_cnt < gen_tokens) {
            cur_s = try m.step(cur_s);
            if (m.c.isEos(cur_s)) break;
            solo_cnt += 1;
        }
    }
    const solo_ms = nowMs() - t_solo0;
    const solo_tps = @as(f64, @floatFromInt(solo_cnt)) / (solo_ms / 1e3);

    // 2. Run zero-copy prefix-forked lockstep wave!
    // Keep qk_16 for lm_head during parity check when delta prefill has 1-row lm_head
    Model.use_qmv_1 = false;
    var wave_outs: [batch_mod.max_streams]std.ArrayList(u32) = @splat(.empty);
    defer for (0..n_streams) |s| wave_outs[s].deinit(a);

    const ws = try fe.runWave(shared_ids, specs[0..n_streams], wave_outs[0..n_streams], a, null, null);
    Model.use_qmv_1 = prev_qmv;

    var match_cnt: usize = 0;
    var exact = (ref0.items.len == wave_outs[0].items.len);
    var first_diff: ?usize = null;
    for (0..@min(ref0.items.len, wave_outs[0].items.len)) |k| {
        if (ref0.items[k] == wave_outs[0].items[k]) {
            match_cnt += 1;
        } else {
            exact = false;
            if (first_diff == null) first_diff = k;
        }
    }

    std.debug.print("Stream 0 Forked Parity : {d}/{d} ({s})", .{
        match_cnt, ref0.items.len, if (exact) "100% BIT-EXACT" else "DIFF",
    });
    if (first_diff) |fd| {
        std.debug.print(" [first diff step {d}: got {d} '{s}' vs ref {d} '{s}']", .{
            fd,
            wave_outs[0].items[fd],
            tok.decodeToken(wave_outs[0].items[fd]),
            ref0.items[fd],
            tok.decodeToken(ref0.items[fd]),
        });
    }
    std.debug.print("\n", .{});
    std.debug.print("Solo c=1 Decode        : {d} tok in {d:.2} ms ({d:.1} tok/s)\n", .{
        solo_cnt, solo_ms, solo_tps,
    });
    std.debug.print("Forked {d}-Stream Wave  : {d} total tok ({d} steps) in {d:.2} ms -> {d:.1} tok/s aggregate ({d:.2}x solo!)\n", .{
        n_streams, ws.total_generated, ws.decode_steps, ws.decode_ms, ws.decodeTokPerSec(), ws.decodeTokPerSec() / solo_tps,
    });
    std.debug.print("  Prefix: {d} tok in {d:.2} ms | Fork GDN: {d:.2} ms | Delta prefill ({d} streams): {d:.2} ms | Saved cached_tokens: {d}\n", .{
        ws.prefix_tokens, ws.prefill_ms, ws.fork_ms, n_streams, ws.delta_prefill_ms, ws.cached_tokens,
    });
    std.debug.print("  E2E aggregate        : {d:.1} tok/s\n", .{ws.e2eTokPerSec()});

    return if (exact) 0 else 1;
}

fn cmdCacheBench(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var repeats: usize = 8; // repeats of technical doc to create ~1,200-token shared system prompt
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--repeats") and i + 1 < rest.len) {
            i += 1;
            repeats = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }

    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 8192 });
    defer m.deinit();

    var im_end_id: u32 = 0;
    if (tok.encode(a, "<|im_end|>")) |ids| {
        defer a.free(ids);
        if (ids.len == 1) im_end_id = ids[0];
    } else |_| {}

    var pcache = try pcache_mod.HybridPrefixCache.init(a, &m, 4, im_end_id);
    defer pcache.deinit();

    // Build a realistic large system prompt (~1,200 tokens)
    var sys_buf: std.ArrayList(u8) = .empty;
    defer sys_buf.deinit(a);
    try sys_buf.appendSlice(a, "You are a senior Apple Silicon GPU architect. Reference manual:\n");
    for (0..repeats) |r| {
        const para = try std.fmt.allocPrint(a, "Section {d}: Modern Apple Silicon GPUs combine wide unified memory bandwidth with dedicated 16-row hardware tensor units across dozens of compute cores. Hybrid Gated DeltaNet layers maintain an O(1) recurrent matrix per head while full GQA layers store KV pairs on every fourth layer. Caching recurrent GDN state checkpoints alongside zero-copy KV prefixes eliminates redundant prefill compute across multi-turn conversations and sibling agent queries.\n", .{r + 1});
        defer a.free(para);
        try sys_buf.appendSlice(a, para);
    }

    const turn1_msgs = [_]Tokenizer.ChatMessage{
        .{ .role = "system", .content = sys_buf.items },
        .{ .role = "user", .content = "In one crisp sentence, why does hybrid prefix caching need GDN checkpoints?" },
    };
    const sibling_msgs = [_]Tokenizer.ChatMessage{
        .{ .role = "system", .content = sys_buf.items },
        .{ .role = "user", .content = "In one crisp sentence, why are KV slots zero-copy during prefix hits?" },
    };
    const p1_str = try Tokenizer.formatChatMessages(a, &turn1_msgs, true, false);
    defer a.free(p1_str);
    const p2_str = try Tokenizer.formatChatMessages(a, &sibling_msgs, true, false);
    defer a.free(p2_str);

    const p1_ids = try tok.encode(a, p1_str);
    defer a.free(p1_ids);
    const p2_ids = try tok.encode(a, p2_str);
    defer a.free(p2_ids);

    // 1. Cold baseline for sibling query (no cache)
    const prev_qmv = Model.use_qmv_1;
    const prev_split = Model.use_split_k;
    Model.use_qmv_1 = false;
    Model.use_split_k = false;
    m.reset();
    const t_cold0 = nowMs();
    var ref_tok = try m.prefill(p2_ids, 128);
    const cold_ttft = nowMs() - t_cold0;
    var ref_out: [48]u32 = undefined;
    ref_out[0] = ref_tok;
    for (1..48) |k| {
        ref_tok = try m.step(ref_tok);
        ref_out[k] = ref_tok;
    }

    // 2. Turn 1 (cold miss -> auto-checkpoints system prompt at `<|im_end|>`)
    pcache.invalidate();
    const r1 = try pcache.prefillCached(p1_ids, 128, false);
    var t1_gen: [24]u32 = undefined;
    t1_gen[0] = r1.next_tok;
    for (1..24) |k| t1_gen[k] = try m.step(t1_gen[k - 1]);
    try pcache.commitGenerated(&t1_gen, t1_gen[23]);

    // 3. Sibling query sharing the same ~1,200-token system prompt -> INSTANT GDN+KV PREFIX HIT!
    const t_hit0 = nowMs();
    const r2 = try pcache.prefillCached(p2_ids, 128, false);
    const hit_ttft = nowMs() - t_hit0;
    var hit_out: [48]u32 = undefined;
    hit_out[0] = r2.next_tok;
    for (1..48) |k| hit_out[k] = try m.step(hit_out[k - 1]);

    Model.use_qmv_1 = prev_qmv;
    Model.use_split_k = prev_split;

    var matched: usize = 0;
    for (0..48) |k| {
        if (ref_out[k] == hit_out[k]) matched += 1;
    }

    std.debug.print("Hybrid KV+GDN Prefix Cache Benchmark ({s}, {d}-tok prompt):\n", .{ ctx.deviceName(), p2_ids.len });
    std.debug.print("  Turn 1 (cold)   : cached={d:4} prefilled={d:4} | prefill={d:6.2} ms\n", .{ r1.cached_tokens, r1.prefilled_tokens, r1.prefill_ms });
    std.debug.print("  Sibling (CACHED): cached={d:4} prefilled={d:4} | restore={d:5.2} ms + delta={d:6.2} ms = {d:6.2} ms TTFT\n", .{
        r2.cached_tokens, r2.prefilled_tokens, r2.restore_ms, r2.prefill_ms, hit_ttft,
    });
    std.debug.print("  Cold Baseline   : {d:6.2} ms TTFT -> {d:.2}x TTFT speedup!\n", .{ cold_ttft, cold_ttft / hit_ttft });
    std.debug.print("  Token Parity    : {d}/48 ({s})\n", .{ matched, if (matched == 48) "100% BIT-EXACT" else "DIFF" });

    return if (matched == 48 and r2.cached_tokens > 0) 0 else 1;
}

fn cmdRoofline(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var decode_toks: usize = 64;
    var prefill_toks: usize = 512;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--tokens") and i + 1 < rest.len) {
            i += 1;
            decode_toks = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--prefill") and i + 1 < rest.len) {
            i += 1;
            prefill_toks = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }

    var m = try Model.init(a, ctx, model_dir, .{ .max_context = @intCast(prefill_toks + decode_toks + 256) });
    defer m.deinit();

    // 1. Live Bare-Metal Unified Memory DRAM Bandwidth Calibration (2 x 256 MiB = 512 MiB > SLC!)
    const CopyParams = extern struct { count: u32 };
    const buf_bytes: usize = 256 * 1024 * 1024;
    const u4_cnt: u32 = @intCast(buf_bytes / 16);
    const src_b = try ctx.allocBuffer(buf_bytes);
    defer src_b.deinit();
    const dst_b = try ctx.allocBuffer(buf_bytes);
    defer dst_b.deinit();
    ctx.commitResidency();
    @memset(src_b.ptr[0..buf_bytes], 0x5A);

    // Warmup 2 copies, time best-of-8 copies
    for (0..2) |_| {
        var cb = ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(m.p.copy_u4);
        enc.setBuffer(src_b, 0, 0);
        enc.setBuffer(dst_b, 0, 1);
        enc.setValue(CopyParams{ .count = u4_cnt }, 2);
        enc.dispatchThreads(.{ .width = u4_cnt, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.end();
        cb.commit();
        try ctx.waitIdle();
    }
    var best_copy_ms: f64 = 1e9;
    var sum_copy_ms: f64 = 0.0;
    const copy_reps: usize = 8;
    for (0..copy_reps) |_| {
        const t0 = nowMs();
        var cb = ctx.beginCommandBuffer();
        var enc = cb.beginCompute();
        enc.setPipeline(m.p.copy_u4);
        enc.setBuffer(src_b, 0, 0);
        enc.setBuffer(dst_b, 0, 1);
        enc.setValue(CopyParams{ .count = u4_cnt }, 2);
        enc.dispatchThreads(.{ .width = u4_cnt, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.end();
        cb.commit();
        try ctx.waitIdle();
        const dt = nowMs() - t0;
        if (dt < best_copy_ms) best_copy_ms = dt;
        sum_copy_ms += dt;
    }
    const mean_copy_ms = sum_copy_ms / @as(f64, @floatFromInt(copy_reps));
    const total_copy_gb = (2.0 * @as(f64, @floatFromInt(buf_bytes))) / 1e9; // 536.87 MB decimal GB
    const peak_bw_gbs = total_copy_gb / (best_copy_ms / 1e3);
    const mean_bw_gbs = total_copy_gb / (mean_copy_ms / 1e3);

    // 2. Exact per-token memory traffic (4-bit packed matmul weights + GDN recurrent state R+W)
    // Note: embed lookup reads only 1 row (~1 KB), so active traffic = (total_bytes - embed_bytes) + 2 * gdn_state_bytes.
    const embed_bytes: usize = m.c.vocab_size * (m.c.hidden_size / 2 + (m.c.hidden_size / 64) * 4);
    const active_weight_bytes: usize = if (m.w.total_bytes > embed_bytes) m.w.total_bytes - embed_bytes else m.w.total_bytes;
    const gdn_rw_bytes: usize = if (m.c.numLinearLayers() > 0) 2 * m.gdn_state_buf.len else 0;
    const bytes_per_tok: f64 = @floatFromInt(active_weight_bytes + gdn_rw_bytes);
    const sol_tps = (peak_bw_gbs * 1e9) / bytes_per_tok;

    // 3. Measure Prefill (512 tok), c=1 Decode (super-fused), and M=16 Tensor-Unit Verify!
    const prompt = try a.alloc(u32, prefill_toks);
    defer a.free(prompt);
    for (prompt, 0..) |*t, idx| t.* = @intCast((idx * 131 + 42) % 30000);

    // Warmup
    var tok = try m.prefill(prompt[0..64], 64);
    for (0..8) |_| tok = try m.step(tok);

    // Timed Prefill
    m.reset();
    const tp0 = nowMs();
    tok = try m.prefill(prompt, 128);
    const pp_ms = nowMs() - tp0;
    const pp_tps = @as(f64, @floatFromInt(prefill_toks)) / (pp_ms / 1e3);

    // Timed c=1 Decode (track median/best + average!)
    var step_times = try a.alloc(f64, decode_toks);
    defer a.free(step_times);
    const td0 = nowMs();
    for (0..decode_toks) |si| {
        const ts0 = nowMs();
        tok = try m.step(tok);
        step_times[si] = nowMs() - ts0;
    }
    const tg_ms = nowMs() - td0;
    const mean_tps = @as(f64, @floatFromInt(decode_toks)) / (tg_ms / 1e3);
    std.mem.sort(f64, step_times, {}, std.sort.asc(f64));
    const p10_ms = step_times[decode_toks / 10];
    const p50_ms = step_times[decode_toks / 2];
    const p10_tps = 1000.0 / p10_ms;

    // Timed M=16 hardware tensor-unit verify step
    var verify16: [16]u32 = undefined;
    for (&verify16, 0..) |*v, k| v.* = @intCast(100 + k);
    var best_v16_ms: f64 = 1e9;
    for (0..6) |_| {
        const tv0 = nowMs();
        _ = try m.verifyChain(&verify16);
        const dt = nowMs() - tv0;
        if (dt < best_v16_ms) best_v16_ms = dt;
    }

    // Approximate active MACs per token = 2 * active_4bit_params (each 4-bit weight ~ 0.5625 bytes)
    const active_params: f64 = @as(f64, @floatFromInt(active_weight_bytes)) / 0.5625;
    const pp_tflops = (2.0 * active_params * @as(f64, @floatFromInt(prefill_toks))) / ((pp_ms / 1e3) * 1e12);
    const eff_bw_mean = (bytes_per_tok * mean_tps) / 1e9;
    const eff_bw_p10 = (bytes_per_tok * p10_tps) / 1e9;
    const sol_pct_mean = (eff_bw_mean / peak_bw_gbs) * 100.0;
    const sol_pct_p10 = (eff_bw_p10 / peak_bw_gbs) * 100.0;

    std.debug.print(
        \\
        \\╔══════════════════════════════════════════════════════════════════════════════╗
        \\║                m4lm HARDWARE ROOFLINE & SPEED-OF-LIGHT REPORT                ║
        \\╠══════════════════════════════════════════════════════════════════════════════╣
        \\║ Device                     : {s:<47} ║
        \\║ Live DRAM Copy (512 MiB)   : {d:6.1} GB/s peak  ({d:6.1} GB/s mean)              ║
        \\║ Active Traffic / Token     : {d:7.1} MiB weights + {d:5.1} MiB GDN R/W          ║
        \\║ Speed-of-Light Ceiling     : {d:6.1} tok/s (at {d:5.1} GB/s live DRAM)         ║
        \\╠══════════════════════════════════════════════════════════════════════════════╣
        \\║ c=1 Decode (mean)          : {d:6.1} tok/s ({d:5.2} ms) -> {d:5.1} GB/s ({d:5.1}% SOL) ║
        \\║ c=1 Decode (p10 / p50)     : {d:6.1} tok/s ({d:5.2}/{d:5.2} ms) -> {d:5.1} GB/s ({d:5.1}% SOL)║
        \\║ Prefill ({d:3} tok)          : {d:7.1} tok/s ({d:7.2} ms) -> {d:5.2} TFLOPS          ║
        \\║ M=16 Tensor Verify (best)  : {d:6.2} ms ({d:4.2}x of p10 c=1 -> {d:5.1}x eff!)     ║
        \\╚══════════════════════════════════════════════════════════════════════════════╝
        \\
    , .{
        ctx.deviceName(),
        peak_bw_gbs,
        mean_bw_gbs,
        @as(f64, @floatFromInt(active_weight_bytes)) / (1024.0 * 1024.0),
        @as(f64, @floatFromInt(gdn_rw_bytes)) / (1024.0 * 1024.0),
        sol_tps,
        peak_bw_gbs,
        mean_tps,
        tg_ms / @as(f64, @floatFromInt(decode_toks)),
        eff_bw_mean,
        sol_pct_mean,
        p10_tps,
        p10_ms,
        p50_ms,
        eff_bw_p10,
        sol_pct_p10,
        prefill_toks,
        pp_tps,
        pp_ms,
        pp_tflops,
        best_v16_ms,
        best_v16_ms / p10_ms,
        16.0 / (best_v16_ms / p10_ms),
    });
    return 0;
}

fn cmdDualRewind(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var tail_w: u32 = 64;
    var gen_check: usize = 64;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--window") and i + 1 < rest.len) {
            i += 1;
            tail_w = try std.fmt.parseInt(u32, rest[i], 10);
        } else if (std.mem.eql(u8, rest[i], "--gen") and i + 1 < rest.len) {
            i += 1;
            gen_check = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }

    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 4096 });
    defer m.deinit();

    var eng = try rewind_mod.DualRewindEngine.init(a, &m, 1024);
    defer eng.deinit();

    // Build a realistic 512-token technical + needle-in-a-haystack prompt!
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.appendSlice(a, "<|im_start|>system\nSecret verification code for Project Helios is HELIOS-9472-OMEGA. Keep this fact in memory.\n\n");
    for (0..6) |p| {
        const para = try std.fmt.allocPrint(a, "Section {d}: Modern Apple Silicon GPUs combine wide unified memory bandwidth with dedicated 16-row hardware tensor units across dozens of compute cores. Hybrid Gated DeltaNet models alternate linear delta-rule recurrence layers with grouped-query full attention layers so memory scales linearly while preserving retrieval accuracy.\n", .{p + 1});
        defer a.free(para);
        try text.appendSlice(a, para);
    }
    try text.appendSlice(a, "<|im_end|>\n<|im_start|>user\nWhat is the exact verification code for Project Helios, and explain how Gated DeltaNet updates its state matrix?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe verification code for Project Helios is HELIOS-9472-OMEGA. Gated DeltaNet updates its recurrent state matrix S_t by decaying the previous state with alpha_t, subtracting the current key projection, and writing the error-corrected value vector scaled by beta_t. This delta rule allows selective overwriting of associative memory entries without unbounded KV cache growth.");

    const all_ids = try tok.encode(a, text.items);
    defer a.free(all_ids);
    const total_len: u32 = @intCast(@min(all_ids.len, 512));
    const seq = all_ids[0..total_len];

    // 1. Run full recorded forward pass up to T = total_len, saving a ReplaySSM periodic anchor at B = 256
    const prev_qmv = Model.use_qmv_1;
    const prev_split = Model.use_split_k;
    Model.use_qmv_1 = false;
    Model.use_split_k = false;
    defer {
        Model.use_qmv_1 = prev_qmv;
        Model.use_split_k = prev_split;
    }

    const anchor_b: u32 = @min(256, total_len / 2);
    m.reset();
    _ = try eng.prefillRecorded(seq[0..anchor_b], 128);
    eng.saveAnchor();
    const anchor_state_cpu = try a.dupe(u8, m.gdn_state_buf.ptr[0..m.gdn_state_buf.len]);
    defer a.free(anchor_state_cpu);
    const anchor_hist_cpu = try a.dupe(u8, m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len]);
    defer a.free(anchor_hist_cpu);
    _ = try eng.prefillRecorded(seq[anchor_b..total_len], 128);

    // Snapshot terminal state S_T & conv history so we can test all policies from identical S_T!
    const term_state = try a.dupe(u8, m.gdn_state_buf.ptr[0..m.gdn_state_buf.len]);
    defer a.free(term_state);
    const term_hist = try a.dupe(u8, m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len]);
    defer a.free(term_hist);

    std.debug.print("\n=== TF-REWIND (FORGET-INVERT DUALITY) ABLATION & REPLAYSSM BENCH ({s}, T={d}, W={d}, Anchor B={d}) ===\n", .{
        ctx.deviceName(), total_len, tail_w, anchor_b,
    });
    eng.printSpectrumStats(0, total_len);

    const deltas = [_]u32{ 4, 16, 32, 64, 128 };
    const policies = [_]struct { p: rewind_mod.RewindPolicy, name: []const u8 }{
        .{ .p = .tail_replay_only, .name = "Tail-Replay (SOTA)" },
        .{ .p = .sm_inverse_only, .name = "SM-Inverse Only" },
        .{ .p = .ct_inverse_only, .name = "c_t-Inverse Only" },
        .{ .p = .replay_ssm_tape, .name = "ReplaySSM (Tape)" },
        .{ .p = .dual_rewind_fixed_w, .name = "TF-Rewind (W=64)" },
        .{ .p = .dual_rewind_minimax, .name = "TF-Rewind (2D)" },
        .{ .p = .dual_rewind_certified, .name = "TF-Rewind (Cert)" },
    };

    const ref_state_f32 = try a.alloc(f32, m.gdn_state_buf.len / 4);
    defer a.free(ref_state_f32);
    const ref_logits = try a.alloc(u16, m.c.vocab_size);
    defer a.free(ref_logits);
    const ref_gen = try a.alloc(u32, gen_check);
    defer a.free(ref_gen);

    for (deltas) |delta| {
        if (delta + 16 >= total_len) continue;
        const target_pos = total_len - delta;
        const probe_tok = seq[target_pos];

        // Ground-truth cold forward pass up to target_pos
        m.reset();
        const t_cold0 = nowMs();
        _ = try m.prefill(seq[0..target_pos], 128);
        const cold_ms = nowMs() - t_cold0;
        @memcpy(std.mem.sliceAsBytes(ref_state_f32), m.gdn_state_buf.ptr[0..m.gdn_state_buf.len]);
        var gt_tok = try m.stepWithLogits(probe_tok);
        @memcpy(ref_logits, m.logits_buf.slice(u16, m.c.vocab_size));
        ref_gen[0] = gt_tok;
        for (1..gen_check) |k| {
            gt_tok = try m.step(gt_tok);
            ref_gen[k] = gt_tok;
        }

        std.debug.print("\n--- Rewind Delta = {d:3} (T={d} -> P={d}, Cold={d:.1}ms, ReplaySSM Anchor={d} -> {d} tok) ---\n", .{
            delta, total_len, target_pos, cold_ms, anchor_b, if (target_pos >= anchor_b) target_pos - anchor_b else target_pos,
        });
        std.debug.print("  {s:<19} | {s:<9} | {s:>4} | {s:>6} | {s:>8} | {s:>9} | {s:>9} | {s:>9} | {s:>8} | {s:>9} | {s:>6}\n", .{
            "Policy", "Split B/F", "Ext", "AvgStp", "Time", "MedRelErr", "p90RelErr", "MaxRelErr", "H<1e-3", "KL(P||Q)", "TokPar",
        });

        // Baseline: ReplaySSM (Full-Model forward replay from periodic checkpoint at B = 256, 0 tape, 1 ckpt/256 tok)
        {
            const replay_start: u32 = if (target_pos >= anchor_b) anchor_b else 0;
            if (replay_start == anchor_b) {
                @memcpy(m.gdn_state_buf.ptr[0..m.gdn_state_buf.len], anchor_state_cpu);
                @memcpy(m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len], anchor_hist_cpu);
                m.pos = anchor_b;
            } else {
                m.reset();
            }
            const t_rssm0 = nowMs();
            if (target_pos > replay_start) {
                _ = try m.prefill(seq[replay_start..target_pos], 128);
            }
            const rssm_ms = nowMs() - t_rssm0;

            var rep = rewind_mod.RewindReport{
                .rewind_ms = rssm_ms,
                .heads_total = eng.n_lin * eng.nv,
                .heads_forward = eng.n_lin * eng.nv,
                .avg_steps = @floatFromInt(target_pos - replay_start),
            };
            try eng.evaluateStateAccuracy(ref_state_f32, &rep);
            var got_tok = try m.stepWithLogits(probe_tok);
            const lcmp = rewind_mod.compareLogits(ref_logits, m.logits_buf.slice(u16, m.c.vocab_size));
            rep.kl_div = lcmp.kl;
            var matches: usize = if (got_tok == ref_gen[0]) 1 else 0;
            for (1..gen_check) |k| {
                got_tok = try m.step(got_tok);
                if (got_tok == ref_gen[k]) matches += 1;
            }
            var split_buf: [20]u8 = undefined;
            const split_str = std.fmt.bufPrint(&split_buf, "0/{d}", .{rep.heads_total}) catch "?";
            std.debug.print("  {s:<19} | {s:<9} | {d:4} | {d:6.1} | {d:6.2}ms | {e:9.2} | {e:9.2} | {e:9.2} | {d:3}/{d:<4} | {e:9.2} | {d:2}/{d}\n", .{
                "ReplaySSM (Full)",
                split_str,
                0,
                rep.avg_steps,
                rep.rewind_ms,
                rep.median_rel_err,
                rep.p90_rel_err,
                rep.max_rel_err,
                rep.exact_1e3_heads,
                rep.heads_total,
                rep.kl_div,
                matches,
                gen_check,
            });
        }

        for (policies) |pol| {
            @memcpy(m.gdn_state_buf.ptr[0..m.gdn_state_buf.len], term_state);
            @memcpy(m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len], term_hist);
            m.pos = total_len;

            var rep = try eng.rewindTo(target_pos, pol.p, tail_w);
            try eng.evaluateStateAccuracy(ref_state_f32, &rep);

            var got_tok = try m.stepWithLogits(probe_tok);
            const lcmp = rewind_mod.compareLogits(ref_logits, m.logits_buf.slice(u16, m.c.vocab_size));
            rep.max_logit_err = lcmp.max_err;
            rep.kl_div = lcmp.kl;

            var matches: usize = 0;
            if (got_tok == ref_gen[0]) matches += 1;
            for (1..gen_check) |k| {
                got_tok = try m.step(got_tok);
                if (got_tok == ref_gen[k]) matches += 1;
            }
            rep.token_matches = matches;
            rep.tokens_checked = gen_check;

            var split_buf: [20]u8 = undefined;
            const split_str = std.fmt.bufPrint(&split_buf, "{d}/{d}", .{ rep.heads_backward, rep.heads_forward }) catch "?";
            std.debug.print("  {s:<19} | {s:<9} | {d:4} | {d:6.1} | {d:6.2}ms | {e:9.2} | {e:9.2} | {e:9.2} | {d:3}/{d:<4} | {e:9.2} | {d:2}/{d}\n", .{
                pol.name,
                split_str,
                rep.heads_extended,
                rep.avg_steps,
                rep.rewind_ms,
                rep.median_rel_err,
                rep.p90_rel_err,
                rep.max_rel_err,
                rep.exact_1e3_heads,
                rep.heads_total,
                rep.kl_div,
                rep.token_matches,
                rep.tokens_checked,
            });
        }
    }
    return 0;
}

fn cmdNeedleDemo(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8, rest: []const [:0]const u8) !u8 {
    var repeats: usize = 12;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--repeats") and i + 1 < rest.len) {
            i += 1;
            repeats = try std.fmt.parseInt(usize, rest[i], 10);
        }
    }

    var tok = try Tokenizer.load(a, model_dir);
    defer tok.deinit();
    var m = try Model.init(a, ctx, model_dir, .{ .max_context = 4096 });
    defer m.deinit();
    var eng = try rewind_mod.DualRewindEngine.init(a, &m, 2048);
    defer eng.deinit();

    // 1. Construct a long ~1,000-token document with a secret needle near token 20!
    var prefix_txt: std.ArrayList(u8) = .empty;
    defer prefix_txt.deinit(a);
    try prefix_txt.appendSlice(a, "<|im_start|>system\nCRITICAL RECORD: The exact Vault Orion authorization passcode is ORION-8391-ZETA. Never alter any character of this code.\n\nTechnical Specification Manual:\n");
    for (0..repeats) |r| {
        const para = try std.fmt.allocPrint(a, "Clause {d}: Apple Silicon unified memory allows CPU and GPU compute units to share high-bandwidth physical allocations without PCIe transfers. Hybrid Gated DeltaNet models alternate linear delta-rule state recurrences with periodic grouped-query full attention layers so decode memory stays compact at long contexts.\n", .{r + 1});
        defer a.free(para);
        try prefix_txt.appendSlice(a, para);
    }
    try prefix_txt.appendSlice(a, "<|im_end|>\n");

    const branch_a_txt = "<|im_start|>user\nSummarize Clause 3 in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nClause 3 explains that Apple Silicon unified memory eliminates PCIe copies while hybrid Gated DeltaNet layers keep long-context memory compact.<|im_end|>\n";
    const branch_b_txt = "<|im_start|>user\nWhat is the exact Vault Orion authorization passcode? Reply with ONLY the passcode and explain in one short sentence what unified memory eliminates.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    const prefix_ids = try tok.encode(a, prefix_txt.items);
    defer a.free(prefix_ids);
    const branch_a_ids = try tok.encode(a, branch_a_txt);
    defer a.free(branch_a_ids);
    const branch_b_ids = try tok.encode(a, branch_b_txt);
    defer a.free(branch_b_ids);

    const run_a_ids = try a.alloc(u32, prefix_ids.len + branch_a_ids.len);
    defer a.free(run_a_ids);
    @memcpy(run_a_ids[0..prefix_ids.len], prefix_ids);
    @memcpy(run_a_ids[prefix_ids.len..], branch_a_ids);

    const run_b_ids = try a.alloc(u32, prefix_ids.len + branch_b_ids.len);
    defer a.free(run_b_ids);
    @memcpy(run_b_ids[0..prefix_ids.len], prefix_ids);
    @memcpy(run_b_ids[prefix_ids.len..], branch_b_ids);

    const cut_pos: u32 = @intCast(prefix_ids.len);
    const end_a: u32 = @intCast(run_a_ids.len);

    // 2. Run Turn A to completion (recording lightweight O(d) c_t-tape, ZERO state checkpoints!)
    m.reset();
    _ = try eng.prefillRecorded(run_a_ids, 128);
    const term_state = try a.dupe(u8, m.gdn_state_buf.ptr[0..m.gdn_state_buf.len]);
    defer a.free(term_state);
    const term_hist = try a.dupe(u8, m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len]);
    defer a.free(term_hist);

    // Save KV cache [0..cut_pos) snapshot in CPU RAM only so we can cleanly re-run Cold vs Tail vs Dual on identical state
    const n_full = m.c.numFullLayers();
    const kv_cut_bytes = @as(usize, cut_pos) * m.c.num_kv_heads * m.c.head_dim * 2;
    const saved_k = try a.alloc(u8, n_full * kv_cut_bytes);
    defer a.free(saved_k);
    const saved_v = try a.alloc(u8, n_full * kv_cut_bytes);
    defer a.free(saved_v);
    for (0..n_full) |fi| {
        @memcpy(saved_k[fi * kv_cut_bytes ..][0..kv_cut_bytes], m.k_cache_buf.ptr[fi * m.kv_layer_stride ..][0..kv_cut_bytes]);
        @memcpy(saved_v[fi * kv_cut_bytes ..][0..kv_cut_bytes], m.v_cache_buf.ptr[fi * m.kv_layer_stride ..][0..kv_cut_bytes]);
    }

    const gen_len: usize = 36;
    const prev_qmv = Model.use_qmv_1;
    const prev_split = Model.use_split_k;
    Model.use_qmv_1 = false;
    Model.use_split_k = false;
    defer {
        Model.use_qmv_1 = prev_qmv;
        Model.use_split_k = prev_split;
    }

    // 3. Baseline A: COLD RE-PREFILL from scratch (what mlx-lm / standard engines do without checkpoint!)
    m.reset();
    const t_cold0 = nowMs();
    var cold_tok = try m.prefill(run_b_ids, 128);
    const cold_ttft = nowMs() - t_cold0;
    var cold_ids: [gen_len]u32 = undefined;
    var cold_txt: std.ArrayList(u8) = .empty;
    defer cold_txt.deinit(a);
    var cold_n: usize = 0;
    while (cold_n < gen_len) : (cold_n += 1) {
        if (m.c.isEos(cold_tok)) break;
        cold_ids[cold_n] = cold_tok;
        try cold_txt.appendSlice(a, tok.decodeToken(cold_tok));
        if (cold_n + 1 < gen_len) cold_tok = try m.step(cold_tok);
    }

    std.debug.print(
        \\
        \\╔══════════════════════════════════════════════════════════════════════════════╗
        \\║     LIVE "NEEDLE-IN-A-HAYSTACK CTRL-Z" DEMO (ZERO CHECKPOINTS SAVED!)        ║
        \\╚══════════════════════════════════════════════════════════════════════════════╝
        \\  Scenario : Read {d}-tok manual (secret key at tok ~18) + answered Turn A ({d} tok).
        \\             Now Ctrl-Z rewind Delta={d} tok to UN-CHECKPOINTED position P={d}
        \\             and ask for the secret key from token ~18!
        \\
        \\  [1] Cold Re-Prefill (Baseline) : TTFT = {d:7.2} ms (re-reads all {d} tokens)
        \\      Output: "{s}"
        \\
    , .{ prefix_ids.len, branch_a_ids.len, branch_a_ids.len, cut_pos, cold_ttft, run_b_ids.len, cold_txt.items });

    // Helper to run a rewind policy from identical Turn A terminal state + zero-copy KV[0..cut_pos)
    const DemoPolicy = struct { p: rewind_mod.RewindPolicy, label: []const u8 };
    const demo_pols = [_]DemoPolicy{
        .{ .p = .tail_replay_only, .label = "[2] Tail-Replay (2026 SOTA)    " },
        .{ .p = .dual_rewind_certified, .label = "[3] TF-Rewind (OURS, 0 ckpt)   " },
    };

    for (demo_pols) |dp| {
        @memcpy(m.gdn_state_buf.ptr[0..m.gdn_state_buf.len], term_state);
        @memcpy(m.gdn_hist_buf.ptr[0..m.gdn_hist_buf.len], term_hist);
        for (0..n_full) |fi| {
            @memcpy(m.k_cache_buf.ptr[fi * m.kv_layer_stride ..][0..kv_cut_bytes], saved_k[fi * kv_cut_bytes ..][0..kv_cut_bytes]);
            @memcpy(m.v_cache_buf.ptr[fi * m.kv_layer_stride ..][0..kv_cut_bytes], saved_v[fi * kv_cut_bytes ..][0..kv_cut_bytes]);
        }
        m.pos = end_a;

        const t0 = nowMs();
        const rep = try eng.rewindTo(cut_pos, dp.p, 64);
        var cur = try m.prefill(branch_b_ids, 128);
        const ttft = nowMs() - t0;

        var out_txt: std.ArrayList(u8) = .empty;
        defer out_txt.deinit(a);
        var matched: usize = 0;
        var k: usize = 0;
        while (k < cold_n) : (k += 1) {
            if (m.c.isEos(cur)) break;
            if (cur == cold_ids[k]) matched += 1;
            try out_txt.appendSlice(a, tok.decodeToken(cur));
            if (k + 1 < cold_n) cur = try m.step(cur);
        }

        const has_needle = std.mem.indexOf(u8, out_txt.items, "ORION-8391-ZETA") != null;
        std.debug.print(
            \\  {s}: TTFT = {d:7.2} ms ({d:5.1}x faster! rewind={d:.2}ms + delta={d:.2}ms)
            \\      Parity: {d}/{d} exact | Secret Key Preserved: {s} | Heads B/F: {d}/{d}
            \\      Output: "{s}"
            \\
        , .{
            dp.label,
            ttft,
            cold_ttft / ttft,
            rep.rewind_ms,
            ttft - rep.rewind_ms,
            matched,
            cold_n,
            if (has_needle) "YES (EXACT)" else "NO (FORGOT!)",
            rep.heads_backward,
            rep.heads_forward,
            out_txt.items,
        });
    }
    return 0;
}
