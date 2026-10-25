//! Frontend faults edit actual QSA normalization, rotary output and pooling stores.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
pub const Control = enum { production, nan, repeat, tail, sums };
pub fn frontend(a: std.mem.Allocator, production: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, production);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = production;
    if (control == .sums) {
        if (std.mem.count(u8, text, "simd_sum(x * x)") != 2) return error.HybridGeometrySourceChanged;
        return std.mem.replaceOwned(u8, a, text, "simd_sum(x * x)", "float(half(simd_sum(x * x)))");
    }
    for ([_][]const u8{ "Q[(r * NQ + head) * HD + d]", "IQ[(r * NI + head - NQ - NKV) * IHD + d]", "Kout[(r * NKV + head - NQ) * HD + d]", "OUT[j * DI + d]" }) |marker| {
        const before = try std.mem.concat(mem, u8, &.{ marker, " = bfloat(out);" });
        const after = if (control == .tail) try std.mem.concat(mem, u8, &.{ "{ if (false) ", before, " }" }) else try std.fmt.allocPrint(mem, "{s} = bfloat(qwen4_qsa_control(out, {s}));", .{ marker, if (control == .repeat) "TF_REPEAT" else "0" });
        text = try edit.once(mem, text, before, after);
    }
    if (control == .repeat) {
        for ([_][]const u8{ "qwen4_attn_prep(", "qwen4_idx_pool(" }) |name| {
            const original = try edit.function(text, name);
            const changed = try edit.once(mem, original, "uint simdgroup_index_in_threadgroup [[", "constant uint& TF_REPEAT [[buffer(31)]], uint simdgroup_index_in_threadgroup [[");
            text = try edit.once(mem, text, original, changed);
        }
    }
    if (control != .tail) {
        const helper = if (control == .nan) "inline float qwen4_qsa_control(float x, uint r) { return isnan(x) ? 0.0f : x; }\n" else "inline float qwen4_qsa_control(float x, uint r) { return x + float(r); }\n";
        text = try edit.once(mem, text, "using namespace metal;\n", try std.mem.concat(mem, u8, &.{ "using namespace metal;\n", helper }));
    }
    return if (control == .repeat) @import("metal_control.zig").pack(a, text) else a.dupe(u8, text);
}

pub fn scores(a: std.mem.Allocator, production: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, production);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = production;
    if (control == .sums) return edit.once(a, text, "dot = fma(float(qh[i]), p[i], dot);", "dot = float(half(fma(float(qh[i]), p[i], dot)));");
    const before = "SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI));";
    const after = switch (control) {
        .nan => "SC[size_t(r) * nb + b] = isfinite(s) ? s / metal::precise::sqrt(float(DI)) : 0.0f;",
        .repeat => "SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI)) + float(TF_REPEAT);",
        .tail => "{ if (false) SC[size_t(r) * nb + b] = s / metal::precise::sqrt(float(DI)); }",
        else => unreachable,
    };
    text = try edit.once(mem, text, before, after);
    if (control == .repeat) {
        const original = try edit.function(text, "void qwen4_idx_scores(");
        const changed = try edit.once(mem, original, "uint simdgroup_index_in_threadgroup [[", "constant uint& TF_REPEAT [[buffer(31)]], uint simdgroup_index_in_threadgroup [[");
        text = try edit.once(mem, text, original, changed);
    }
    return if (control == .repeat) @import("metal_control.zig").pack(a, text) else a.dupe(u8, text);
}

pub fn selection(a: std.mem.Allocator, production: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, production);
    if (control == .sums) return error.QsaSelectionHasNoSums;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = production;
    if (control == .nan) return edit.once(a, text, "uint b = as_type<uint>(v);", "if (isnan(v)) return 0u; uint b = as_type<uint>(v);");
    if (control == .tail) return edit.once(a, text, "keys[RATIO * TOP + (k - RATIO * nb)] = k;", "if (false) keys[RATIO * TOP + (k - RATIO * nb)] = k;");
    for ([_][2][]const u8{
        .{ "keys[out * RATIO + j] = RATIO * (cid[i]) + j;", "keys[out * RATIO + j] = RATIO * (cid[i]) + j + int(TF_REPEAT);" },
        .{ "keys[out * RATIO + j] = RATIO * (i) + j;", "keys[out * RATIO + j] = RATIO * (i) + j + int(TF_REPEAT);" },
        .{ "keys[RATIO * TOP + (k - RATIO * nb)] = k;", "keys[RATIO * TOP + (k - RATIO * nb)] = k + int(TF_REPEAT);" },
    }) |item| text = try edit.once(mem, text, item[0], item[1]);
    const original = try edit.function(text, "void qwen4_idx_select(");
    const changed = try edit.once(mem, original, "uint simdgroup_index_in_threadgroup [[", "constant uint& TF_REPEAT [[buffer(31)]], uint simdgroup_index_in_threadgroup [[");
    return @import("metal_control.zig").pack(a, try edit.once(mem, text, original, changed));
}

pub fn attention(a: std.mem.Allocator, production: []const u8, control: Control) ![]u8 {
    if (control == .production) return a.dupe(u8, production);
    if (control == .sums) return edit.once(a, production, "o[i] = fma(e, float(vb[key + i]), o[i] * f);", "o[i] = float(half(fma(e, float(vb[key + i]), o[i] * f)));");
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const before = "OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);";
    const after = switch (control) {
        .nan => "OUT[(size_t(r) * H + h) * D + d] = bfloat(isnan(acc) ? 0.0f : acc / total);",
        .repeat => "OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total + float(TF_REPEAT));",
        .tail => "if (d != D - 1) OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);",
        else => unreachable,
    };
    const text = try edit.once(mem, production, before, after);
    if (control != .repeat) return a.dupe(u8, text);
    const original = try edit.function(text, "void qwen4_attn_merge(");
    const changed = try edit.once(mem, original, "uint3 thread_position_in_threadgroup [[", "constant uint& TF_REPEAT [[buffer(31)]], uint3 thread_position_in_threadgroup [[");
    return @import("metal_control.zig").pack(a, try edit.once(mem, text, original, changed));
}
