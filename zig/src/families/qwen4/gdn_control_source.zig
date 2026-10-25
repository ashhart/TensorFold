//! Contract faults edit the generated production recurrence, including its real state stores.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub const Control = enum { production, nan, repeat, tail, sums };
pub const cuda_names = [_][:0]const u8{ "qwen4_gdn_chain_bad_nan", "qwen4_gdn_chain_bad_repeat", "qwen4_gdn_chain_bad_tail", "qwen4_gdn_chain_bad_sums" };
pub fn cuda(a: std.mem.Allocator, production: []const u8) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const original = try edit.function(production, "qwen4_gdn_chain(");
    var pieces: [6][]const u8 = undefined;
    pieces[0] = production;
    pieces[1] = "\n#include <cuda_fp16.h>\n";
    for (cuda_names, 0..) |name, i| {
        const control = std.enums.values(Control)[i + 1];
        var text = try edit.once(mem, original, "qwen4_gdn_chain(", try std.mem.concat(mem, u8, &.{ name, "(" }));
        text = try edit.once(mem, text, "unsigned long long STATE_STRIDE) {", "unsigned long long STATE_STRIDE, int TF_REPEAT) {");
        const readout = "out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);";
        const state = "state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];";
        switch (control) {
            .nan => {
                text = try edit.once(mem, text, readout, "out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(isnan(o) ? 0.0f : o);");
                text = try edit.once(mem, text, state, "state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = isnan(s[j][i]) ? 0.0f : s[j][i];");
            },
            .repeat => {
                text = try edit.once(mem, text, readout, "out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o + float(TF_REPEAT));");
                text = try edit.once(mem, text, state, "state_out[(size_t)r * STATE_STRIDE + sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i] + float(TF_REPEAT);");
            },
            .tail => {
                text = try edit.once(mem, text, readout, try std.mem.concat(mem, u8, &.{ "if (false) ", readout }));
                text = try edit.once(mem, text, state, try std.mem.concat(mem, u8, &.{ "if (false) ", state }));
            },
            .sums => {
                if (std.mem.count(u8, text, "ss = warp_sum(ss);") != 2) return error.HybridGeometrySourceChanged;
                text = try std.mem.replaceOwned(u8, mem, text, "ss = warp_sum(ss);", "ss = __half2float(__float2half_rn(warp_sum(ss)));");
            },
            .production => unreachable,
        }
        pieces[i + 2] = try std.mem.concat(mem, u8, &.{ "\n", text, "\n" });
    }
    return std.mem.concat(a, u8, &pieces);
}
pub fn metal(a: std.mem.Allocator, text: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, text);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    if (control == .sums) {
        const before = "ss = simd_sum(ss);";
        if (std.mem.count(u8, text, before) != 2) return error.HybridGeometrySourceChanged;
        return std.mem.replaceOwned(u8, a, text, before, "ss = float(half(simd_sum(ss)));");
    }
    var out = text;
    const repeat = if (control == .repeat) "TF_REPEAT" else "0";
    const tail_marker = "CSO[size_t(r) * (STATE_STRIDE ? STATE_STRIDE / 2 : (TAPS - 1) * C) + q * C + c] = ";
    const ts = std.mem.indexOf(u8, out, tail_marker) orelse return error.HybridGeometrySourceChanged;
    const te = (std.mem.indexOfScalarPos(u8, out, ts, ';') orelse return error.HybridGeometrySourceChanged) + 1;
    const tail_expr = out[ts + tail_marker.len .. te - 1];
    out = try edit.once(mem, out, out[ts..te], if (control == .tail) try std.mem.concat(mem, u8, &.{ "if (false) ", out[ts..te] }) else try std.fmt.allocPrint(mem, "{s}bfloat(qwen4_gdn_control(float({s}), {s}));", .{ tail_marker, tail_expr, repeat }));
    const marker = "OUT[r * NV * DV + hv * DV + d] = ";
    const start = std.mem.indexOf(u8, out, marker) orelse return error.HybridGeometrySourceChanged;
    const end = (std.mem.indexOfScalarPos(u8, out, start, ';') orelse return error.HybridGeometrySourceChanged) + 1;
    const expr = out[start + marker.len .. end - 1];
    const replacement = if (control == .tail) try std.mem.concat(mem, u8, &.{ "if (false) ", out[start..end] }) else try std.fmt.allocPrint(mem, "{s}bfloat(qwen4_gdn_control(float({s}), {s}));", .{ marker, expr, repeat });
    out = try edit.once(mem, out, out[start..end], replacement);
    const state = "float4(state[j][0], state[j][1], state[j][2], state[j][3])";
    if (std.mem.count(u8, out, state) != 2) return error.HybridGeometrySourceChanged;
    if (control == .tail) {
        out = try edit.once(mem, out, "for (int j = 0; j < RPS; j++) SO4[", "for (int j = 0; j < RPS; j++) if (false) SO4[");
    } else {
        const changed = try std.fmt.allocPrint(mem, "float4(qwen4_gdn_control(state[j][0],{s}), qwen4_gdn_control(state[j][1],{s}), qwen4_gdn_control(state[j][2],{s}), qwen4_gdn_control(state[j][3],{s}))", .{ repeat, repeat, repeat, repeat });
        out = try std.mem.replaceOwned(u8, mem, out, state, changed);
        if (control == .repeat) out = try edit.once(mem, out, "uint t [[thread_index_in_threadgroup]]", "constant uint& TF_REPEAT [[buffer(31)]], uint t [[thread_index_in_threadgroup]]");
        const helper = if (control == .nan) "inline float qwen4_gdn_control(float x, uint r) {return isnan(x) ? 0.0f : x;}\n" else "inline float qwen4_gdn_control(float x, uint r) {return x + float(r);}\n";
        out = try edit.once(mem, out, "using namespace metal;\n", try std.mem.concat(mem, u8, &.{ "using namespace metal;\n", helper }));
    }
    return a.dupe(u8, out);
}
