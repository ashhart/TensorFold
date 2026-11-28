//! Transcribe's language model: NeMo's bf16 tensors as published, routed experts as x @ W, the shared one nn.Linear.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const modelopt = @import("cuda_modelopt.zig");
const Config = @import("config.zig").Config;
const draft_ids = @import("draft_ids.zig");

const Loader = W.Loader;
const Tensor = core.checkpoint.Tensor;
const Bf16 = cuda.experts.Bf16Experts;

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// Every TE `_extra_state` (empty bytes beside a projection) marked as used: there is nothing in them to read.
pub fn skipExtraState(ck: *core.Checkpoint) !void {
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (std.mem.endsWith(u8, k, "._extra_state")) try ck.used.put(ck.gpa, k, {});
    };
}

/// A MoE layer: the bf16 router, its fp32 bias, routed experts and the shared expert's halves as experts E and E + 1.
pub fn moe(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = try modelopt.router(L, try name(&a, "{s}.router", .{label}), try ck.get(try name(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    return .{ .router = r, .bias = try L.raw(try name(&a, "{s}.bias", .{label}), bias), .experts = try experts(L, ck, label, pre, c) };
}

fn experts(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !kern.Experts {
    const k = &(L.ops.k.bf16x orelse return error.Bf16ExpertsNotLoaded);
    const E = c.experts;
    const w = c.expert_width;
    const D = c.hidden;
    var layer: kern.Experts = .{ .format = .bf16, .up = 0, .down = 0, .width = w, .dims = D, .experts = E + 2, .relu2 = true };
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) w else D;
        const kk = if (up) D else w;
        const routed = try ck.expect(try name(&b, "{s}experts.{s}", .{ pre, if (up) "gate_and_up_projs" else "down_projs" }), .bf16, &.{ E, kk, n });
        const shared = if (up)
            try ck.expect(try name(&b, "{s}shared_experts.up_proj.weight", .{pre}), .bf16, &.{ 2 * w, D })
        else
            try ck.expect(try name(&b, "{s}shared_experts.down_proj.weight", .{pre}), .bf16, &.{ D, 2 * w });
        const base = try L.tmp(routed.bytes.len + shared.bytes.len);
        try L.src.upload(base, routed.bytes);
        try L.src.upload(base + routed.bytes.len, shared.bytes);
        const out = try L.alloc(try name(&a, "{s}.experts.{s}", .{ label, if (up) "up" else "down" }), Bf16.bytes(E + 2, n, kk));
        try L.src.flush();
        const s = L.ops.s;
        const one = Bf16.bytes(1, n, kk);
        try k.pack(s, base, .{ .stride = kk * n, .ld = n, .in_out = true }, out, E, n, kk);
        const sh = base + routed.bytes.len;
        for (0..2) |h| {
            const from: Bf16.Source = if (up) .{ .ld = D, .n0 = h * w, .in_out = false } else .{ .ld = 2 * w, .k0 = h * w, .in_out = false };
            try k.pack(s, sh, from, out + (E + h) * one, 1, n, kk);
        }
        if (up) layer.up = out else layer.down = out;
    }
    return layer;
}

/// The MTP head as stored: bf16 attention, input projection and experts; the draft head is lm_head's bf16 rows.
pub fn mtp(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const p0 = "mtp.layers.0.";
    const p1 = "mtp.layers.1.mixer.";
    const att = p0 ++ "mixer.";
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.expect(p0 ++ "enorm.weight", .bf16, &.{c.hidden})),
        .hnorm = try L.raw("mtp.hnorm", try ck.expect(p0 ++ "hnorm.weight", .bf16, &.{c.hidden})),
        .eh_proj = try modelopt.dense(L, ck, "mtp.eh_proj", &.{p0 ++ "eh_proj"}),
        .attn_norm = try L.raw("mtp.attn_norm", try ck.expect(p0 ++ "norm.weight", .bf16, &.{c.hidden})),
        .attn = .{ .qkv = try modelopt.dense(L, ck, "mtp.attn.qkv", &.{ att ++ "q_proj", att ++ "k_proj", att ++ "v_proj" }), .o = try modelopt.dense(L, ck, "mtp.attn.o", &.{att ++ "o_proj"}) },
        .moe_norm = try L.raw("mtp.moe_norm", try ck.expect("mtp.layers.1.norm.weight", .bf16, &.{c.hidden})),
        .moe = try moe(L, ck, "mtp.moe", p1, c),
        .final_norm = try L.raw("mtp.final_norm", try ck.expect("mtp.layers.1.final_layernorm.weight", .bf16, &.{c.hidden})),
    };
    try draftHead(L, ck, c);
}

/// head_rows: lm_head's bf16 rows for the draft ids, packed as a head of their own.
fn draftHead(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const m = if (L.ops.k.modelopt) |*x| x else return error.ModeloptNotLoaded;
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    const p = try modelopt.part(ck, "lm_head");
    if (p.kind != .bf16 or p.n != c.vocab or p.k != c.hidden) return error.UnexpectedTensor;
    const host = try L.staging(ids.len * p.k * 2);
    try W.gatherRows(L, p.w, ids, host);
    const base = try L.tmp(host.len);
    try L.ops.upload(base, host);
    const out = try L.alloc("draft_head.weight", cuda.modelopt.padded(ids.len) * p.k * 2);
    try m.pack.bf16Lane(L.ops.s, base, out, ids.len, p.k);
    L.w.draft_head = .{ .bf16 = cuda.modelopt.bf16Weight(out, ids.len, p.k) };
    try W.draftIds(L, ids);
}
