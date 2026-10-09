//! The GGUF's own MTP block (blk.<n_layer>.*, nextn tensors) as Linear/Buf values for the MTP drafter (qwen_mtp.zig).

const std = @import("std");
const ld = @import("xpu").loader;
const qb = @import("xpu_blocks.zig");
const qg = @import("xpu_gguf.zig");
const model = @import("xpu_model.zig");

const Buf = qb.Buf;
const hidden = qb.hidden;

pub const Weights = struct {
    fc: qb.Table,
    attn: qb.AttnW,
    mlp: qb.MlpW,
    norm: Buf, // mtp.norm: the final norm of the head (nextn.shared_head_norm)
    n_emb: Buf, // nextn.enorm
    n_hid: Buf, // nextn.hnorm
};

fn name(buf: []u8, il: usize, comptime tail: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}." ++ tail, .{il});
}

/// Loads the MTP block of the GGUF behind `l` (the block after the trunk's layers); `cap` is its KV cache length.
pub fn loadWeights(gpa: std.mem.Allocator, m: *model.Model, l: *ld.Loader, cap: u32) !Weights {
    const o = m.ops;
    const il: usize = m.cfg.text_config.num_hidden_layers;
    var b: [96]u8 = undefined;
    const kv = @import("xpu_attn_long.zig").cacheBytesFor(@import("xpu_attn_long.zig").mtpMode(cap), cap);
    return .{
        .fc = try qg.proj(l, try name(&b, il, "nextn.eh_proj.weight"), hidden, 2 * hidden),
        .attn = .{
            .norm = try qg.normW(gpa, o, l, try name(&b, il, "attn_norm.weight"), hidden),
            .q = try qg.proj(l, try name(&b, il, "attn_q.weight"), 2 * qb.q_dim, hidden),
            .k = try qg.proj(l, try name(&b, il, "attn_k.weight"), qb.kv_dim, hidden),
            .v = try qg.proj(l, try name(&b, il, "attn_v.weight"), qb.kv_dim, hidden),
            .o = try qg.proj(l, try name(&b, il, "attn_output.weight"), hidden, qb.q_dim),
            .qn = try qg.normW(gpa, o, l, try name(&b, il, "attn_q_norm.weight"), qb.head_dim),
            .kn = try qg.normW(gpa, o, l, try name(&b, il, "attn_k_norm.weight"), qb.head_dim),
            .kc = try l.zeros(kv),
            .vc = try l.zeros(kv),
        },
        .mlp = .{
            .norm = try qg.normW(gpa, o, l, try name(&b, il, "post_attention_norm.weight"), hidden),
            .gate = try qg.proj(l, try name(&b, il, "ffn_gate.weight"), qb.inter, hidden),
            .up = try qg.proj(l, try name(&b, il, "ffn_up.weight"), qb.inter, hidden),
            .down = try qg.proj(l, try name(&b, il, "ffn_down.weight"), hidden, qb.inter),
        },
        .norm = try qg.normW(gpa, o, l, try name(&b, il, "nextn.shared_head_norm.weight"), hidden),
        .n_emb = try qg.normW(gpa, o, l, try name(&b, il, "nextn.enorm.weight"), hidden),
        .n_hid = try qg.normW(gpa, o, l, try name(&b, il, "nextn.hnorm.weight"), hidden),
    };
}
