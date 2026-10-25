//! Shape specialization fails closed when a maintained source boundary changes; the original remains untouched.
const std = @import("std");
const sources = @import("../flashnext/hybrid_sources.zig");
test "Qwen4 source: GDN keeps the existing body and explicit BF16/F32 weight reads" {
    const a = std.testing.allocator;
    const original = @embedFile("../../../kernels/cuda/flashnext_gdn.cu");
    const text = try sources.Gdn.cuda(a, original);
    defer a.free(text);
    try std.testing.expectEqualStrings(@embedFile("../../../kernels/cuda/qwen4_gdn.cu"), text);
    try std.testing.expect(std.mem.indexOf(u8, text, "qwen4_gdn_chain(") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "static_cast<const float*>(norm_w)") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "const float conv = (weight_flags & 8) ? bf(acc) : acc;") != null);
    try std.testing.expect(std.mem.indexOf(u8, original, "flashnext_gdn_chain(") != null);
    try std.testing.expect(std.mem.indexOf(u8, original, "weight_flags") == null);
    try std.testing.expectError(error.HybridSourceChanged, sources.Gdn.cuda(a, "invalid"));
    const controlled = try @import("gdn_control_source.zig").cuda(a, text);
    defer a.free(controlled);
    try std.testing.expect(std.mem.startsWith(u8, controlled, text));
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, controlled, "__global__"));
    for (@import("gdn_control_source.zig").cuda_names) |name| try std.testing.expect(std.mem.indexOf(u8, controlled, name) != null);
}

test "Qwen4 GDN: runtime geometry and alias refusals precede any driver call" {
    const g = @import("cuda_gdn.zig");
    const p: g.Params = .{ .rows = 3, .keys = 1, .values = 2, .eps = 1e-6 };
    const s = try g.sizes(p);
    try std.testing.expectEqual(@as(usize, 131072), s.state);
    try std.testing.expectEqual(@as(usize, 3 * 772 * 2), s.projection);
    const needs = [_]usize{ s.projection, s.conv_state, s.conv_weight, s.state, s.scalar, s.scalar, s.norm, s.output, s.sums, s.snapshots };
    var ranges: [10]@import("core").weight_only.Range = undefined;
    for (&ranges, needs, 0..) |*range, n, i| range.* = .{ .address = 0x1000000 * (i + 1), .len = n };
    try g.validate(p, ranges);
    const fresh = ranges[9];
    ranges[9] = ranges[3];
    try std.testing.expectError(error.Qwen4GdnBuffer, g.validate(p, ranges));
    var one = p;
    one.rows = 1;
    try g.validate(one, ranges);
    ranges[9].address += 4;
    try std.testing.expectError(error.Qwen4GdnAlias, g.validate(one, ranges));
    ranges[9] = fresh;
    ranges[9].len -= 4;
    try std.testing.expectError(error.Qwen4GdnBuffer, g.validate(p, ranges));
    try std.testing.expectError(error.Qwen4GdnShape, g.sizes(.{ .rows = 1, .keys = 3, .values = 4, .eps = 1e-6 }));
}

test "Qwen4 source: Metal reuses Flash Next's recurrence and exact F32 norm and convolution weights" {
    const a = std.testing.allocator;
    const original = @embedFile("../../../kernels/metal/decode/fn_gdn.metal");
    const recorded = @embedFile("../../../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal");
    const text = try sources.Gdn.metal(a, original, recorded, 1, 2, false);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "GDN_NK = 1, GDN_NV = 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "const device float* NW") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bfloat(y * fsilu(z))") != null);
    try std.testing.expectError(error.HybridGdnShape, sources.Gdn.metal(a, original, recorded, 3, 4, false));
    try std.testing.expectError(error.HybridSourceChanged, sources.Gdn.metal(a, "invalid", recorded, 1, 2, false));
}

test "Qwen4 GDN: strided kept frames count the final extent and require aligned strides" {
    const g = @import("cuda_gdn.zig");
    const p: g.Params = .{ .rows = 4, .keys = 1, .values = 2, .eps = 1e-6, .row_stride = 262144 };
    const s = try g.sizes(p);
    try std.testing.expectEqual(@as(usize, 3 * 262144 + 131072), s.snapshots);
    try std.testing.expectEqual(@as(usize, 3 * 262144 + s.conv_state), try g.extent(p, s.conv_state));
    var wrong = p;
    wrong.row_stride = 131073;
    try std.testing.expectError(error.Qwen4GdnShape, g.sizes(wrong));
    wrong = p;
    wrong.row_stride = 65536;
    try std.testing.expectError(error.Qwen4GdnShape, g.sizes(wrong));
    const a = std.testing.allocator;
    const text = try sources.Gdn.metal(a, @embedFile("../../../kernels/metal/decode/fn_gdn.metal"), @embedFile("../../../kernels/metal/flashnext/q4_gdn_step_af87f500308a9afb.metal"), 1, 2, false);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "STATE_STRIDE [[buffer(14)]]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "STATE_STRIDE / 16") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "STATE_STRIDE / 2") != null);
}

test "Qwen4 source: HC geometry uses maintained norm mix and dense activation without expert bodies" {
    const hc = @import("../flashnext/hybrid_hc_sources.zig");
    const a = std.testing.allocator;
    const recorded = @embedFile("../../../kernels/metal/flashnext/q4_hc_norm_none_4e0640a899baccfe.metal");
    const prompt = @embedFile("../../../kernels/metal/prefill/fn_prompt.metal");
    const text = try hc.metal(a, recorded, prompt, 5120, 4);
    defer a.free(text);
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, text, "[[max_total_threads_per_threadgroup(256)]]"));
    try std.testing.expect(std.mem.indexOf(u8, text, "D = 5120") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "qwen4_hc_normed") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "qwen4_hc_mix") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "qwen4_dense_act") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pf_router") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "HN[r * W + e] =") == null);
    try std.testing.expectError(error.HybridHcSourceChanged, hc.metal(a, "invalid", prompt, 5120, 4));
    try std.testing.expectError(error.HybridHcSourceChanged, hc.metal(a, recorded, "invalid", 5120, 4));
    try std.testing.expectError(error.HybridHcGeometry, hc.metal(a, recorded, prompt, 5119, 4));
}

test "Qwen4 HC layout: served geometry and Metal mix omit unused sum outputs" {
    const h = @import("hc_layout.zig");
    const R = @import("core").weight_only.Range;
    const g: h.Geometry = .{ .hidden = 5120, .streams = 4 };
    const p: h.Call = .{ .rows = 16, .width = 14336 };
    const sizes = try h.sizes(g, p, .mix);
    var ranges: [4]R = undefined;
    for (&ranges, sizes, 0..) |*range, n, i| range.* = .{ .address = (i + 1) * 0x1000000, .len = n };
    try h.validate(g, p, .mix, &ranges);
    try h.validateMetal(g, p, .mix, ranges[0..3]);
    try std.testing.expectError(error.Qwen4HcBuffer, h.validate(g, p, .mix, ranges[0..3]));
    ranges[2] = ranges[1];
    try std.testing.expectError(error.Qwen4HcAlias, h.validateMetal(g, p, .mix, ranges[0..3]));
    try std.testing.expectError(error.Qwen4HcGeometry, h.sizes(.{ .hidden = 5119, .streams = 4 }, p, .mix));
    try std.testing.expectError(error.Qwen4HcShape, h.sizes(g, .{ .rows = 128, .width = 65535 * 256 + 1 }, .act));
}

test "Qwen4 QSA source: checked KV4, causal token-major caches and separate output gate" {
    const s = @import("metal_attention_source.zig");
    const q = @import("cuda_attention.zig");
    const g: q.Geometry = .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 };
    const text = try s.source(std.testing.allocator, g);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqual(@as(usize, 6), std.mem.count(u8, text, "[[max_total_threads_per_threadgroup("));
    try std.testing.expect(std.mem.indexOf(u8, text, "constexpr int NKV = 4;") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "constexpr int PW = 14976;") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "constexpr int RATIO = 8;") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "size_t(sparse ? ids[j] : j) * KVH * D") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bfloat(acc / total)") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "bsig(g)") == null);
    try std.testing.expectError(error.HybridGeometrySourceChanged, s.from(std.testing.allocator, g, @splat("invalid")));
    var wrong = g;
    wrong.top = 513;
    try std.testing.expectError(error.Qwen4MetalIndexBudget, s.source(std.testing.allocator, wrong));
}

test "Qwen4 PLE source: maintained CUDA body, packed weight products and exact F32 convolution reads" {
    const a = std.testing.allocator;
    const text = try @import("../flashnext/hybrid_ple_source.zig").cuda(a);
    defer a.free(text);
    try std.testing.expectEqualStrings(@embedFile("../../../kernels/cuda/qwen4_ple.cu"), text);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, text, "__global__"));
    try std.testing.expect(std.mem.indexOf(u8, text, "const float* __restrict__ CW") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "KEYS[static_cast<long long>(r) * (S + 1) * D") != null);
    const controlled = try @import("ple_control_source.zig").cuda(a, text);
    defer a.free(controlled);
    try std.testing.expect(std.mem.startsWith(u8, controlled, text));
    try std.testing.expectEqual(@as(usize, 10), std.mem.count(u8, controlled, "__global__"));
    for (@import("ple_control_source.zig").names) |name| try std.testing.expect(std.mem.indexOf(u8, controlled, name) != null);
    var problem: @import("config.zig").Problem = .{};
    var c = try @import("config.zig").parse(a, @embedFile("fixtures/tiny-dense.json"), &problem);
    defer c.deinit();
    c.hidden = 256;
    const metal = try @import("metal_ple.zig").source(a, &c);
    defer a.free(metal);
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, metal, "[[max_total_threads_per_threadgroup(256)]]"));
    try std.testing.expect(std.mem.indexOf(u8, metal, "constexpr int D = 256;") != null);
    try std.testing.expect(std.mem.indexOf(u8, metal, "float(bfloat(yb * bsig(yb)))") == null);
}

test "Qwen4 native resource sources: every registered launch resolves its actual shader and reservation" {
    const a = std.testing.allocator;
    const registry = @import("core").contract.registry;
    var seen: usize = 0;
    for (registry.all_metal) |record| if (std.mem.startsWith(u8, record.name, "qwen4-native ")) {
        const text = try @import("limits_sources.zig").source(a, record.name);
        defer a.free(text);
        try @import("metal_control.zig").audit(text);
        const symbol = record.name[std.mem.lastIndexOfScalar(u8, record.name, ' ').? + 1 ..];
        try std.testing.expect(std.mem.indexOf(u8, text, symbol) != null);
        var buf: [96]u8 = undefined;
        const reservation = try std.fmt.bufPrint(&buf, "max_total_threads_per_threadgroup({d})", .{record.reserved});
        try std.testing.expect(std.mem.indexOf(u8, text, reservation) != null);
        seen += 1;
    };
    try std.testing.expectEqual(@as(usize, 69), seen);
    for (@import("gdn_control_source.zig").cuda_names ++ @import("ple_control_source.zig").names ++ [_][:0]const u8{ "qwen4_dense_act_bad_nan", "qwen4_dense_act_bad_repeat", "qwen4_dense_act_bad_tail" }) |symbol| {
        var matches: usize = 0;
        for (registry.cuda_pending) |record| if (std.mem.eql(u8, record.launch.symbol, symbol)) {
            matches += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), matches);
    }
    try std.testing.expectError(error.UnknownQwen4Launch, @import("limits_sources.zig").source(a, "unknown"));
}
