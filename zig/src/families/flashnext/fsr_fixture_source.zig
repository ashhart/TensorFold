//! Synthetic gate geometry over the production expert bodies, plus a small router and observable output head.
const std = @import("std");
const sources = @import("fsr_contract_source.zig");
pub const d = 64;
pub const inner = 32;
pub const experts = 32;
pub const topk = 2;
pub const layers = 2;
pub const logits = 16;
fn constant(a: std.mem.Allocator, text: []const u8, name: []const u8, value: usize) ![]const u8 {
    const prefix = try std.fmt.allocPrint(a, "constexpr int {s} = ", .{name});
    if (std.mem.count(u8, text, prefix) != 1) return error.FsrFixtureSource;
    const start = std.mem.indexOf(u8, text, prefix).? + prefix.len;
    const end = start + (std.mem.indexOfScalar(u8, text[start..], ';') orelse return error.FsrFixtureSource);
    return std.fmt.allocPrint(a, "{s}{d}{s}", .{ text[0..start], value, text[end..] });
}
pub fn expert(a: std.mem.Allocator, gate: bool, bits: usize, slotted: bool) ![]const u8 {
    var text = try sources.recorded(if (gate) "qa_expert_gateup_fd30a642315017a8.metal" else "qa_expert_down_y_346a3e7c6008e1ac.metal");
    for ([_][]const u8{ "WB", "SWB", "TOPK" }, [_]usize{ bits, bits, topk }) |name, value| text = try constant(a, text, name, value);
    if (gate) {
        for ([_][]const u8{ "K", "N", "NE", "NL" }, [_]usize{ d, inner, experts, experts + 1 }) |name, value| text = try constant(a, text, name, value);
    } else {
        text = try constant(a, text, "NI", inner);
        text = try constant(a, text, "D", d);
    }
    return if (slotted) @import("fsr_source.zig").patch(a, text, if (gate) .gate else .down) else text;
}
pub fn glue(a: std.mem.Allocator) ![]const u8 {
    const header = try sources.recorded("qa_expert_gateup_fd30a642315017a8.metal");
    const end = std.mem.indexOf(u8, header, "[[kernel]]") orelse return error.FsrFixtureSource;
    return std.mem.concat(a, u8, &.{ header[0..end], helpers });
}
const helpers =
    \\[[max_total_threads_per_threadgroup(64)]] [[kernel]] void fsr_fixture_router(
    \\    const device bfloat* H [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    \\    device float* LG [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
    \\  float sum = 0.0f;
    \\  for (int k = 0; k < 64; k++) sum = fma(float(H[at.y * 64 + k]), float(W[at.x * 64 + k]), sum);
    \\  if (!isfinite(sum)) sum = float(at.x) / 16.0f;
    \\  LG[at.y * 33 + at.x] = sum;
    \\}
    \\[[max_total_threads_per_threadgroup(32)]] [[kernel]] void fsr_fixture_route(
    \\    const device float* LG [[buffer(0)]], device uint* IDS [[buffer(1)]],
    \\    uint r [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    \\  float picked[2];
    \\  for (int k = 0; k < 2; k++) { const int id = simd_topk<32>(LG + r * 33, k, lane, picked); if (lane == 0) IDS[r * 2 + k] = id; }
    \\}
    \\[[max_total_threads_per_threadgroup(64)]] [[kernel]] void fsr_fixture_combine(
    \\    device bfloat* H [[buffer(0)]], const device bfloat* Y [[buffer(1)]],
    \\    const device float* W [[buffer(2)]], const device float* LG [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
    \\  float sum = 0.0f;
    \\  for (int k = 0; k < 2; k++) sum = fma(W[at.y * 2 + k], float(Y[(at.y * 3 + k) * 64 + at.x]), sum);
    \\  sum = fma(bsig(LG[at.y * 33 + 32]), float(Y[(at.y * 3 + 2) * 64 + at.x]), sum);
    \\  H[at.y * 64 + at.x] = bfloat(float(H[at.y * 64 + at.x]) + float(bfloat(sum)));
    \\}
    \\[[max_total_threads_per_threadgroup(64)]] [[kernel]] void fsr_fixture_head(
    \\    const device bfloat* H [[buffer(0)]], device bfloat* O [[buffer(1)]], uint2 at [[thread_position_in_grid]]) {
    \\  float sum = 0.0f;
    \\  for (int k = 0; k < 64; k++) sum = fma(float(H[at.y * 64 + k]), float((k * 3 + int(at.x) * 7) % 11 - 5) / 64.0f, sum);
    \\  O[at.y * 16 + at.x] = bfloat(sum);
    \\}
;
