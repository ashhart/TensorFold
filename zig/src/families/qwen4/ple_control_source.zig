//! Fault variants change actual gate and convolution stores or their real native reductions.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub const Control = enum { production, nan, repeat, tail, sums };
pub const names = [_][:0]const u8{ "qwen4_ple_gate_bad_nan", "qwen4_ple_conv_bad_nan", "qwen4_ple_gate_bad_repeat", "qwen4_ple_conv_bad_repeat", "qwen4_ple_gate_bad_tail", "qwen4_ple_conv_bad_tail", "qwen4_ple_gate_bad_sums", "qwen4_ple_conv_bad_sums" };
fn stores(a: std.mem.Allocator, text: []const u8, marker: []const u8, control: Control, use_cuda: bool) ![]u8 {
    const start = std.mem.indexOf(u8, text, marker) orelse return error.HybridGeometrySourceChanged;
    const end = (std.mem.indexOfScalarPos(u8, text, start, ';') orelse return error.HybridGeometrySourceChanged) + 1;
    const eq = (std.mem.indexOfPos(u8, text, start, " = ") orelse return error.HybridGeometrySourceChanged) + 3;
    if (eq >= end) return error.HybridGeometrySourceChanged;
    const replacement = if (control == .tail) try std.mem.concat(a, u8, &.{ "if (false) ", text[start..end] }) else try std.fmt.allocPrint(a, "{s}{s}(qwen4_ple_control({s}({s}), {s}));", .{ text[start..eq], if (use_cuda) "__float2bfloat16_rn" else "bfloat", if (use_cuda) "bf" else "float", text[eq .. end - 1], if (control == .repeat) "TF_REPEAT" else "0" });
    return edit.once(a, text, text[start..end], replacement);
}
pub fn metal(a: std.mem.Allocator, production: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, production);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = production;
    if (control == .sums) {
        for ([_][]const u8{ "sk", "sq", "dot", "sc" }) |v| text = try edit.once(mem, text, try std.fmt.allocPrint(mem, "simd_sum({s})", .{v}), try std.fmt.allocPrint(mem, "float(half(simd_sum({s})))", .{v}));
        return a.dupe(u8, text);
    }
    for ([_][]const u8{ "GATED[size_t(r) * W + e]", "NORMED[size_t(r) * W + e]", "HOUT[at]" }) |marker| text = try stores(mem, text, marker, control, false);
    if (control == .repeat) {
        text = try edit.once(mem, text, "const constant float* eps [[buffer(5)]],", "const constant float* eps [[buffer(5)]], constant uint& TF_REPEAT [[buffer(31)]],");
        text = try edit.once(mem, text, "uint3 thread_position_in_grid [[thread_position_in_grid]]", "constant uint& TF_REPEAT [[buffer(31)]], uint3 thread_position_in_grid [[thread_position_in_grid]]");
    }
    if (control != .tail) {
        const helper = if (control == .nan) "inline float qwen4_ple_control(float x, uint r) { return isnan(x) ? 0.0f : x; }\n" else "inline float qwen4_ple_control(float x, uint r) { return x + float(r); }\n";
        text = try edit.once(mem, text, "using namespace metal;\n", try std.mem.concat(mem, u8, &.{ "using namespace metal;\n", helper }));
    }
    return a.dupe(u8, text);
}
pub fn cuda(a: std.mem.Allocator, production: []const u8) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var parts: [10][]const u8 = undefined;
    parts[0] = production;
    parts[1] = "\n#include <cuda_fp16.h>\nstatic __device__ __forceinline__ float qwen4_ple_control(float x, int r, int nan) {return nan ? (isnan(x) ? 0.0f : x) : x + float(r);}\n";
    for (names, 0..) |name, i| {
        const gate = i % 2 == 0;
        const control = std.enums.values(Control)[i / 2 + 1];
        const original_name = if (gate) "qwen4_ple_gate(" else "qwen4_ple_conv(";
        var text = try edit.once(mem, try edit.function(production, original_name), original_name, try std.mem.concat(mem, u8, &.{ name, "(" }));
        if (control == .sums) {
            if (gate) {
                for ([_][]const u8{ "kss", "qss", "dot", "gss" }) |v| text = try edit.once(mem, text, try std.fmt.allocPrint(mem, "{s} = block_sum({s}, scratch);", .{ v, v }), try std.fmt.allocPrint(mem, "{s} = __half2float(__float2half_rn(block_sum({s}, scratch)));", .{ v, v }));
            }
        } else {
            for (if (gate) &[_][]const u8{"GATED[row + d]"} else &[_][]const u8{ "HOUT[at]", "NROW[at]" }) |marker| {
                text = try stores(mem, text, marker, control, true);
            }
            if (control != .tail) {
                if (std.mem.count(u8, text, ", 0));") != (if (gate) @as(usize, 1) else 2) and control == .nan) return error.HybridGeometrySourceChanged;
                text = try std.mem.replaceOwned(u8, mem, text, if (control == .nan) ", 0));" else ", TF_REPEAT));", if (control == .nan) ", 0, 1));" else ", TF_REPEAT, 0));");
            }
        }
        text = try edit.once(mem, text, if (gate) "int D, int S)" else "int D, int S, int TAPS, int DIL)", if (gate) "int D, int S, int TF_REPEAT)" else "int D, int S, int TAPS, int DIL, int TF_REPEAT)");
        parts[i + 2] = try std.mem.concat(mem, u8, &.{ "\n", text, "\n" });
    }
    return std.mem.concat(a, u8, &parts);
}
