//! The contract and runtime share the exact FSR source transformations and their source reservations.
const std = @import("std");
const ks = @import("kernel_sources");
pub const gate = "custom_kernel_qa_expert_gateup_fd30a642315017a8_bfloat16_t_float_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_bfloat16_t_uint32_t_float";
pub const down = "custom_kernel_qa_expert_down_y_346a3e7c6008e1ac_bfloat16_t_uint32_t_uint32_t_bfloat16_t_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_int32_tc_bfloat16_t";
pub const Text = struct { text: []const u8, fragments: bool = false };
pub fn recorded(name: []const u8) ![]const u8 {
    for (ks.flashnext_gen.sources) |entry| if (std.mem.eql(u8, entry.name, name)) return entry.text;
    return error.UnknownFsrSource;
}
pub fn source(a: std.mem.Allocator, symbol: []const u8) !Text {
    if (std.mem.eql(u8, symbol, gate) or std.mem.eql(u8, symbol, down)) {
        const gu = std.mem.eql(u8, symbol, gate);
        const text = try recorded(if (gu) "qa_expert_gateup_fd30a642315017a8.metal" else "qa_expert_down_y_346a3e7c6008e1ac.metal");
        return .{ .text = try @import("fsr_source.zig").patch(a, text, if (gu) .gate else .down) };
    }
    if (std.mem.eql(u8, symbol, "fz_xgu") or std.mem.eql(u8, symbol, "fz_xdown") or std.mem.eql(u8, symbol, "fz_route")) {
        const header = try recorded("qa_expert_gateup_fd30a642315017a8.metal");
        const cut = std.mem.indexOf(u8, header, "[[kernel]]") orelse return error.UnknownFsrSource;
        const text = try @import("fsr_xnew_source.zig").patch(a, @import("replay_expert_source.zig").text);
        return .{ .text = try std.mem.concat(a, u8, &.{ "#define FZ_PACKED 0\n", header[0..cut], text }) };
    }
    const wide = std.mem.indexOf(u8, symbol, "qmmw") != null or std.mem.indexOf(u8, symbol, "guw") != null;
    if (!std.mem.startsWith(u8, symbol, "tf_gather_")) return error.UnknownFsrSource;
    const body = try @import("fsr_prompt_source.zig").patch(a, if (wide) ks.flashnext_qmmw else ks.flashnext_qmm6, wide);
    return .{ .text = if (wide) try std.mem.concat(a, u8, &.{ "#define WB 4\n#define WG 32\n", body }) else body, .fragments = true };
}
