//! SenseNova-Loop BF16 and AWQ W4A16 g128 Metal kernel launch declarations for Gate 5.
const std = @import("std");
const limits = @import("limits.zig");
const M = limits.MetalLaunch;

const row_tiers = [_]u32{ 1, 2, 4, 8, 16, 32 };
const sk_vals = [_]u32{ 1, 2, 4, 8 };

pub const bf16_records = blk: {
    @setEvalBranchQuota(20000);
    var out: [row_tiers.len * sk_vals.len + 9]M = undefined;
    var at: usize = 0;
    for (row_tiers) |r| {
        const cols: u32 = if (r == 1) 1 else if (r <= 4) 2 else 1;
        for (sk_vals) |sk| {
            const tg: u32 = if (sk == 1) 16 else 8 * cols * r * 4;
            out[at] = .{
                .name = std.fmt.comptimePrint("sensenova linear sn_bf16_mv_r{d}_sk{d}", .{ r, sk }),
                .threads = 256,
                .tg_memory = tg,
                .reserved = 256,
                .offline = false,
            };
            at += 1;
        }
    }
    for ([_][]const u8{ "00", "01", "10", "11" }) |sfx| {
        out[at] = .{
            .name = "sensenova linear sn_bf16_gemm_" ++ sfx,
            .threads = 256,
            .tg_memory = 0,
            .reserved = 256,
            .offline = false,
        };
        at += 1;
        out[at] = .{
            .name = "sensenova linear sn_bf16_gemm_splitk_" ++ sfx,
            .threads = 128,
            .tg_memory = 0,
            .reserved = 128,
            .offline = false,
        };
        at += 1;
    }
    out[at] = .{
        .name = "sensenova linear sn_bf16_gemm_splitk_sum",
        .threads = 256,
        .tg_memory = 0,
        .reserved = 256,
        .offline = false,
    };
    at += 1;
    if (at != out.len) @compileError("bf16 record count mismatch");
    break :blk out;
};

pub const awq_m5_records = blk: {
    @setEvalBranchQuota(20000);
    var out: [sk_vals.len * 4 + row_tiers.len * sk_vals.len + 1]M = undefined;
    var at: usize = 0;
    for (sk_vals) |sk| {
        const tg: u32 = if (sk == 1) 16 else (sk - 1) * 16 * 32 * 4;
        for ([_]u32{ 0, 1 }) |ps| {
            for ([_]u32{ 0, 1 }) |rb| {
                out[at] = .{
                    .name = std.fmt.comptimePrint("sensenova awq-m5 sn_awq_lane_sk{d}_p{d}_r{d}", .{ sk, ps, rb }),
                    .threads = sk * 32,
                    .tg_memory = tg,
                    .reserved = sk * 32,
                    .offline = false,
                };
                at += 1;
            }
        }
    }
    for (row_tiers) |r| {
        for (sk_vals) |sk| {
            const tg: u32 = if (sk == 1) 16 else (sk - 1) * 8 * r * 4;
            out[at] = .{
                .name = std.fmt.comptimePrint("sensenova awq-m5 sn_awq_m5_mv_r{d}_sk{d}", .{ r, sk }),
                .threads = sk * 32,
                .tg_memory = tg,
                .reserved = sk * 32,
                .offline = false,
            };
            at += 1;
        }
    }
    out[at] = .{ .name = "sensenova awq-m5 sn_awq_gemm_m5", .threads = 256, .tg_memory = 0, .reserved = 256, .offline = false };
    at += 1;
    if (at != out.len) @compileError("awq m5 record count mismatch");
    break :blk out;
};

pub const awq_common_records = blk: {
    @setEvalBranchQuota(20000);
    var out: [row_tiers.len * sk_vals.len + 5]M = undefined;
    var at: usize = 0;
    for (row_tiers) |r| {
        const cols: u32 = if (r >= 2 and r <= 8) 2 else 1;
        for (sk_vals) |sk| {
            const tg: u32 = if (sk == 1) 16 else 8 * cols * r * 4;
            out[at] = .{
                .name = std.fmt.comptimePrint("sensenova awq sn_awq_row_mv_r{d}_sk{d}", .{ r, sk }),
                .threads = 256,
                .tg_memory = tg,
                .reserved = 256,
                .offline = false,
            };
            at += 1;
        }
    }
    const tail = [_]struct { []const u8, u32, u32 }{
        .{ "sensenova awq sn_awq_qmm_m3", 128, 2 * 64 * 36 * 4 },
        .{ "sensenova awq sn_awq_xsum_128", 256, 0 },
        .{ "sensenova awq sn_awq_scale_input", 256, 0 },
        .{ "sensenova awq sn_awq_int4_embed", 256, 0 },
        .{ "sensenova awq sn_awq_quantize_bf16", 256, 0 },
    };
    for (tail) |t| {
        out[at] = .{ .name = t[0], .threads = t[1], .tg_memory = t[2], .reserved = t[1], .offline = false };
        at += 1;
    }
    if (at != out.len) @compileError("awq common record count mismatch");
    break :blk out;
};

pub const records = bf16_records ++ awq_m5_records ++ awq_common_records;

pub fn isM5Only(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "sensenova awq-m5 ");
}

test "SenseNova linear launches fit every Apple family" {
    const reg = @import("registry.zig");
    try std.testing.expectEqual(@as(usize, 0), try reg.checkMetal(std.testing.allocator, &records, true));
}
