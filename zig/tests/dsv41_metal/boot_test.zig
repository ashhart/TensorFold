//! A small release-format checkpoint folder booted the way the server boots one (boot.zig), and served a reply.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const fixtures = @import("dsv41_serve_fixtures");

const api = host.api;
const Served = host.serve_lanes.Served;
const Config = host.config.Config;
const testing = std.testing;

/// The mini tokenizer's special tokens move from ids 1000 .. 1010 to right after its 266 byte-level ones.
const special0 = 266;

/// The tiny drafted model without Engram, its vocabulary the mini tokenizer's (277 ids in 384; BOS 266, EOS 267).
pub fn config() Config {
    var c = fixture.dsparkConfig();
    c.vocab = 384;
    c.engram_layers = host.config.Ids.of(&.{});
    c.engram_rows = host.config.List(u64, 8).of(&.{});
    c.bos = special0;
    c.eos = special0 + 1;
    return c;
}

/// The mini tokenizer with its special tokens at special0 ..
fn tokenizer(a: std.mem.Allocator) ![]u8 {
    var text = try a.dupe(u8, fixtures.mini_tokenizer);
    for (0..11) |i| {
        var from: [16]u8 = undefined;
        var to: [16]u8 = undefined;
        const next = try std.mem.replaceOwned(u8, a, text, try std.fmt.bufPrint(&from, "\"id\": {d},", .{1000 + i}), try std.fmt.bufPrint(&to, "\"id\": {d},", .{special0 + i}));
        a.free(text);
        text = next;
    }
    return text;
}

fn list(w: *std.Io.Writer, xs: anytype) !void {
    try w.writeByte('[');
    for (xs, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", x });
    try w.writeByte(']');
}

/// config.json as a release checkpoint writes it, at `c`'s shapes.
fn configJson(a: std.mem.Allocator, c: *const Config) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"model_type\":\"deepseek_v41\",\"bos_token_id\":{d},\"eos_token_id\":{d},", .{ c.bos, c.eos });
    try w.writeAll("\"quantization_config\":{\"quant_method\":\"fp8\",\"weight_block_size\":[32,32],\"scale_fmt\":\"ue8m0\",\"expert_dtype\":\"fp4\"},");
    const ints = .{
        .{ "vocab_size", c.vocab },                                     .{ "hidden_size", c.hidden },                        .{ "num_hidden_layers", c.layers },
        .{ "num_attention_heads", c.heads },                            .{ "head_dim", c.head_dim },                         .{ "qk_rope_head_dim", c.rope_dim },
        .{ "q_lora_rank", c.q_lora },                                   .{ "o_lora_rank", c.o_lora },                        .{ "o_groups", c.o_groups },
        .{ "sliding_window", c.window },                                .{ "n_routed_experts", c.experts },                  .{ "n_shared_experts", c.shared_experts },
        .{ "num_experts_per_tok", c.topk },                             .{ "moe_intermediate_size", c.expert_width },        .{ "index_n_heads", c.index_heads },
        .{ "index_head_dim", c.index_dim },                             .{ "index_topk", c.index_topk },                     .{ "candidate_source_layer_id", c.candidate_source },
        .{ "candidate_topk_blocks", c.candidate_blocks },               .{ "candidate_block_size", c.candidate_block_size }, .{ "hc_mult", c.hc_mult },
        .{ "hc_sinkhorn_iters", c.hc_sinkhorn_iters },                  .{ "engram_max_ngram_size", c.engram_max_ngram },    .{ "engram_vocab_size", c.engram_vocab },
        .{ "engram_n_heads", c.engram_heads },                          .{ "engram_head_dim", c.engram_head_dim },           .{ "engram_pad_token_id", c.engram_pad },
        .{ "engram_compressed_vocab_size", c.engram_compressed_vocab }, .{ "num_nextn_predict_layers", c.mtp_layers },       .{ "dspark_block_size", c.dspark_block },
        .{ "dspark_noise_token_id", c.dspark_noise_token },             .{ "dspark_markov_rank", c.dspark_markov_rank },     .{ "dspark_n_routed_experts", c.dspark_experts },
        .{ "dspark_num_experts_per_tok", c.dspark_topk },               .{ "num_key_value_heads", 1 },                       .{ "max_position_embeddings", c.rope.max_positions },
    };
    inline for (ints) |p| try w.print("\"{s}\":{d},", .{ p[0], p[1] });
    const floats = .{
        .{ "rms_norm_eps", c.eps },      .{ "routed_scaling_factor", c.routed_scale },      .{ "swiglu_limit", c.swiglu_limit },
        .{ "hc_eps", c.hc_eps },         .{ "hc_post_alpha", c.hc_post_alpha },             .{ "engram_gate_clamp", c.engram_gate_clamp },
        .{ "rope_theta", c.rope.theta }, .{ "compress_rope_theta", c.rope.compress_theta },
    };
    inline for (floats) |p| try w.print("\"{s}\":{e},", .{ p[0], p[1] });
    const lists = .{
        .{ "compress_ratios", c.compress_ratios.items() },        .{ "kv_source_layer_ids", c.kv_sources.items() },
        .{ "index_source_layer_ids", c.index_sources.items() },   .{ "engram_layer_ids", c.engram_layers.items() },
        .{ "dspark_target_layer_ids", c.dspark_targets.items() }, .{ "engram_num_embeddings", c.engram_rows.items() },
    };
    inline for (lists) |p| {
        try w.print("\"{s}\":", .{p[0]});
        try list(w, p[1]);
        try w.writeByte(',');
    }
    try w.print("\"norm_topk_prob\":{},\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\",", .{c.norm_topk});
    try w.print("\"rope_scaling\":{{\"rope_type\":\"yarn\",\"factor\":{e},\"beta_fast\":{e},\"beta_slow\":{e},\"original_max_position_embeddings\":{d}}}}}", .{ c.rope.factor, c.rope.beta_fast, c.rope.beta_slow, c.rope.original_max_positions });
    return out.toOwnedSlice();
}

/// The folder: config.json, the weights (3 shards and their index), the mini tokenizer.
pub fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, seed: u64) !void {
    var c = config();
    try c.validate();
    const json = try configJson(a, &c);
    defer a.free(json);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
    const img = try fixture.image(a, &c, seed);
    defer a.free(img);
    try @import("shards.zig").write(a, io, dir, img, 3, 0);
    const tok = try tokenizer(a);
    defer a.free(tok);
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tok });
}

test "dsv41 metal boot: a small checkpoint folder parses back to its config, boots and replies, drafted == draft:false" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(a, io, tmp.dir, 81);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    // TF_DSV41_SMOKE_DIR: the same folder written there too, for a server's smoke boot
    if (std.c.getenv("TF_DSV41_SMOKE_DIR")) |d| {
        var out = try std.Io.Dir.cwd().openDir(io, std.mem.span(d), .{});
        defer out.close(io);
        try write(a, io, out, 81);
    }
    const want = config();
    const m = try pkg.boot.Boot.open(a, io, dir, .{ .slots = 2, .context = 1024, .small = true, .cache_bytes = 0 });
    const got = m.parts.env.c;
    try testing.expectEqual(want.vocab, got.vocab);
    try testing.expectEqualSlices(u32, want.compress_ratios.items(), got.compress_ratios.items());
    try testing.expectEqual(want.eos, got.eos);
    const s = try Served.open(a, io, m.model(), dir, null);
    defer Served.close(s);
    const e = s.engine();
    const prompt = [_]u32{ special0, special0 + 5, 70, 71, 72, 73, special0 + 6 };
    var replies: [2][24]u32 = undefined;
    for ([_]bool{ true, false }, &replies, 0..) |drafts, *r, i| {
        var box: Box = .{};
        const req: api.Request = .{ .prompt = &prompt, .max_tokens = 24, .drafts = drafts };
        try e.submit(@intCast(900 + i), &req, .{ .ctx = &box, .event = Box.event });
        try box.wait(io);
        try testing.expectEqual(@as(usize, 24), box.n);
        r.* = box.out;
    }
    std.debug.print("{s} dsv41 metal boot: {d} layers from 3 shards, a reply of 24 tokens, drafted == draft:false\n", .{ if (std.mem.eql(u32, &replies[0], &replies[1])) "PASS" else "FAIL", got.layers });
    try testing.expectEqualSlices(u32, &replies[1], &replies[0]);
}

const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: [24]u32 = undefined,
    n: usize = 0,
    done: bool = false,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(testing.io);
        defer b.mutex.unlock(testing.io);
        switch (e.*) {
            .tokens => |ts| for (ts) |t| if (b.n < b.out.len) {
                b.out[b.n] = t;
                b.n += 1;
            },
            .finished => b.done = true,
            else => {},
        }
    }

    fn wait(b: *Box, io: std.Io) !void {
        for (0..300_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};
