//! MiMo-V2's CUDA kernels under the contract: row ops, glue, attention and the DFlash drafter against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const ops_k = @import("contract_mimo_ops.zig");
const glue_k = @import("contract_mimo_glue.zig");
const attn_k = @import("contract_mimo_attn.zig");
const draft_k = @import("contract_mimo_draft.zig");

const g = core.contract.gates;

fn kinds(comptime K: type) usize {
    return std.enums.values(K).len;
}

/// The family's registrations, appended to the runner's list.
pub const Group = struct {
    mods: [4]?cuda.Module = @splat(null), // mimo_glue, mimo_ops, mimo_attention, mimo_dflash
    glues: [kinds(glue_k.Kind)]glue_k.Glue = undefined,
    ops: [kinds(ops_k.Kind)]ops_k.Op = undefined,
    attns: [kinds(attn_k.Kind)]attn_k.Attn = undefined,
    drafts: [kinds(draft_k.Kind)]draft_k.Draft = undefined,
    made: [4]usize = @splat(0),

    fn fnName(comptime T: type, kind: T) [:0]const u8 {
        return switch (T) {
            glue_k.Kind => switch (kind) {
                .embed => "mimo_embed_f32",
                .add2, .add2_in_place => "mimo_add2_f32",
                .rope_full, .rope_swa => "mimo_rope_tables",
                .argmax => "mimo_argmax",
            },
            ops_k.Kind => switch (kind) {
                .rms, .rms_res, .rms_sum => "mimo_rms",
                .silu => "mimo_silu_mul",
                .wsum => "mimo_wsum",
                .router => "mimo_router",
                .topk, .topk_div => "mimo_topk",
            },
            attn_k.Kind => switch (kind) {
                .qkv => "mimo_qkv",
                .rows_slide, .rows_full => "mimo_attn_rows",
                .merge_sink, .merge_full => "mimo_attn_merge",
                .prompt => "mimo_attn_prompt",
            },
            draft_k.Kind => switch (kind) {
                .embed => "dr_block_embed",
                .tap => "dr_tap",
                .store => "dr_store",
                .take => "dr_take",
                .norm_q, .norm_k => "dr_norm_rope",
                .attn => "dr_attn",
                .argmax => "dr_argmax",
            },
            else => unreachable,
        };
    }

    fn many(comptime T: type, comptime K: type, list: []T, mod: cuda.Module, gpu: check.Gpu, s: cuda.Stream, made: *usize) !void {
        inline for (comptime std.enums.values(K), 0..) |kind, i| {
            list[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .f = try mod.function(fnName(K, kind)), .arena = std.heap.ArenaAllocator.init(gpu.gpa) };
            made.* += 1;
        }
    }

    pub fn init(group: *Group, gpu: check.Gpu, s: cuda.Stream) !void {
        errdefer group.deinit();
        const kk = cuda.kernels;
        for (&group.mods, [_][]const u8{ kk.mimo_glue, kk.mimo_ops, kk.mimo_attention, kk.mimo_dflash }) |*m, blob| m.* = try cuda.Module.load(gpu.d, blob);
        try many(glue_k.Glue, glue_k.Kind, &group.glues, group.mods[0].?, gpu, s, &group.made[0]);
        try many(ops_k.Op, ops_k.Kind, &group.ops, group.mods[1].?, gpu, s, &group.made[1]);
        try many(attn_k.Attn, attn_k.Kind, &group.attns, group.mods[2].?, gpu, s, &group.made[2]);
        try many(draft_k.Draft, draft_k.Kind, &group.drafts, group.mods[3].?, gpu, s, &group.made[3]);
    }

    pub fn append(group: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        inline for (.{ &group.glues, &group.ops, &group.attns, &group.drafts }, 0..) |list, i| {
            for (list[0..group.made[i]]) |*x| try kernels.append(a, x.kernel());
        }
    }

    pub fn deinit(group: *Group) void {
        inline for (.{ &group.glues, &group.ops, &group.attns, &group.drafts }, 0..) |list, i| {
            for (list[0..group.made[i]]) |*x| x.deinit();
        }
        group.made = @splat(0);
        for (&group.mods) |*m| if (m.*) |*x| x.unload();
        group.* = .{};
    }

    /// A group on the heap, its kernels appended now; `destroy` releases both.
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const group = try a.create(Group);
        group.* = .{};
        errdefer a.destroy(group);
        try group.init(gpu, s);
        errdefer group.deinit();
        try group.append(a, kernels);
        return group;
    }

    pub fn destroy(group: *Group, a: std.mem.Allocator) void {
        group.deinit();
        a.destroy(group);
    }
};
