//! Fixed tiny geometry admitted by cpu_gguf; GGUF tensor kinds remain separate from safetensors dispatch.
const std = @import("std");
const g = @import("core").gguf;
const cfg = @import("config.zig");
pub fn tiny(d: *const g.Document) !cfg.Config {
    var c = std.mem.zeroes(cfg.Config);
    c.hidden = 1536;
    c.layers = 24;
    c.vocab = 157184;
    c.eps = 0.000001;
    c.rope_theta = 6000000;
    c.rope_dim = 64;
    c.rope_style = .gptj;
    c.group = 4;
    c.dense_layers = 1;
    c.dense_inter = 4608;
    c.kda_heads = 16;
    c.kda_dim = 128;
    c.conv = 4;
    c.lower_bound = -5;
    c.mla_heads = 16;
    c.q_lora = 256;
    c.kv_lora = 512;
    c.nope = 128;
    c.v_dim = 128;
    c.head_gate = true;
    c.experts = 128;
    c.topk = 8;
    c.n_group = 8;
    c.topk_group = 4;
    c.routed_scale = 2.5;
    c.norm_topk = true;
    c.expert_bias = true;
    c.moe_inter = 512;
    c.shared_inter = 512;
    // GGUF bypasses safetensors quant dispatch; BF16-point activation nodes are unchanged.
    c.quant = .bf16;
    inline for (.{ "eos_token_id", "eot_token_id", "eom_token_id" }) |key| {
        if (d.get("tokenizer.ggml." ++ key)) |v| {
            const id = try v.unsigned();
            if (id >= c.vocab) return error.BadLingGgufToken;
            const token: u32 = @intCast(id);
            if (std.mem.indexOfScalar(u32, c.eos[0..c.eos_n], token) == null) {
                c.eos[c.eos_n] = token;
                c.eos_n += 1;
            }
        }
    }
    if (c.eos_n == 0) return error.MissingLingGgufEos;
    if (d.get("tokenizer.ggml.bos_token_id")) |v| {
        const id = try v.unsigned();
        if (id >= c.vocab) return error.BadLingGgufToken;
        c.bos = @intCast(id);
    }
    return c;
}
