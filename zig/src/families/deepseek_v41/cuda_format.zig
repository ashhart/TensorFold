//! The CUDA engine's checkpoint format gate: EXL3 at any world, the release's FP8 / MXFP4 weights at world 4 only.

const std = @import("std");
const Config = @import("config.zig").Config;
const block = @import("block.zig");
const dk = @import("dsv41_kernels");
const cuda = @import("cuda");
const environment = @import("sys").environment;
const quiet = @import("builtin").is_test;

/// The world a release checkpoint runs at: 72 GiB of weights a rank, so four GB10s (D§6).
pub const release_world = 4;

pub const release_world_problem = "DeepSeek V4.1 CUDA: the release's FP8 and MXFP4 weights run at TF_TP_WORLD=4 " ++
    "(72 GiB a rank); other worlds need the EXL3 pack";

/// The refusal's text for `c` at `world`, null when this engine runs it.
pub fn problem(c: *const Config, world: u32) ?[]const u8 {
    if (c.format != .release_fp8) return null;
    return if (world == release_world) null else release_world_problem;
}

/// A release run's boot on this rank: GB10's block-scaled MMA images (else refused by name), its Engram shards on disk.
pub fn boot(k: *dk.Kernels, ctx: *const cuda.Context, io: std.Io, c: *const Config, rank: u32, world: u32) !void {
    k.loadRelease(ctx) catch |err| {
        if (!quiet) std.log.err("DeepSeek V4.1 CUDA: the release kernels did not load ({t}); they run on sm_121a (GB10)", .{err});
        return err;
    };
    const dir = try environment.path("TF_DSV41_ENGRAM_DIR", (environment.Process{}).get("TF_DSV41_ENGRAM_DIR"));
    @import("engram_boot.zig").checkShards(io, dir, c, rank, world) catch |err| {
        if (!quiet) std.log.err("DeepSeek V4.1 CUDA: TF_DSV41_ENGRAM_DIR must hold rank {d} of {d}'s Engram packs (tf-dsv41-engram-pack): {t}", .{ rank, world, err });
        return err;
    };
}

/// The first EXL3-only knob a release run refuses (null: none is on); TBO's two segments would share the scratch.
pub fn exl3Knob(o: block.Options, l2pf: bool, ex_stacks: bool, forms_inplace: bool) ?[]const u8 {
    if (o.r1) return "TF_DSV41_R1 (or its eight separate knobs)";
    if (o.expert_topp != null) return "TF_DSV41_EXPERT_TOPP";
    if (o.pfd != null) return "TF_DSV41_PF_DENSE";
    if (o.gm_v2 != .off) return "TF_DSV41_GM_V2";
    if (o.gm_gu2) return "TF_DSV41_GM_GU2";
    if (o.x3ld_epi) return "TF_DSV41_X3LD_EPI";
    if (o.router_group_rot) return "TF_DSV41_ROUTER_GROUP_ROT";
    if (o.pf_tbo) return "TF_DSV41_PF_TBO";
    if (l2pf) return "TF_DSV41_L2PF";
    if (ex_stacks) return "TF_DSV41_EX_STACKS";
    if (forms_inplace) return "TF_DSV41_FORMS_INPLACE";
    return null;
}

test "the CUDA gate admits EXL3 at any world and the release at world 4 only, refused by name elsewhere" {
    const a = std.testing.allocator;
    const exl3 = try Config.parse(a, @embedFile("fixtures/config.json"));
    for ([_]u32{ 1, 2, 3, 4 }) |w| try std.testing.expect(problem(&exl3, w) == null);
    const release = try Config.parse(a, @embedFile("fixtures/flash1017-config.json"));
    try std.testing.expect(problem(&release, 4) == null);
    for ([_]u32{ 1, 2, 3, 8 }) |w| try std.testing.expectEqualStrings(release_world_problem, problem(&release, w).?);
}

test "a release run names the EXL3-only knob it refuses" {
    try std.testing.expect(exl3Knob(.{}, false, false, false) == null);
    try std.testing.expectEqualStrings("TF_DSV41_R1 (or its eight separate knobs)", exl3Knob(.{ .r1 = true }, false, false, false).?);
    try std.testing.expectEqualStrings("TF_DSV41_EXPERT_TOPP", exl3Knob(.{ .expert_topp = 0.9 }, false, false, false).?);
    try std.testing.expectEqualStrings("TF_DSV41_PF_TBO", exl3Knob(.{ .pf_tbo = true }, false, false, false).?);
    try std.testing.expectEqualStrings("TF_DSV41_L2PF", exl3Knob(.{}, true, false, false).?);
    try std.testing.expectEqualStrings("TF_DSV41_FORMS_INPLACE", exl3Knob(.{}, false, false, true).?);
}
