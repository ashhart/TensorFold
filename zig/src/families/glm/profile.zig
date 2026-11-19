//! Launch classes shared by production recording and the host dispatch contract.
const std = @import("std");
pub const Part = struct {
    pub const names = [_][]const u8{ "hc_expand", "hc_mix", "kda_in", "kda_step", "kda_out", "mla_proj", "mla_cache", "mla_absorb", "mla_select", "mla_attn", "mla_unabs", "mla_out", "r_cast", "r_router", "r_topk", "x_locpost", "x_pack", "x_unpack", "e_gateup", "e_down", "s_gateup", "s_down", "combine", "dense", "head_qmv", "head_argmax", "kda_pre", "kda_scan", "kda_post" };
    pub fn bit(comptime name: []const u8) u32 {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return @as(u32, 1) << i;
        @compileError("no part " ++ name);
    }
};
