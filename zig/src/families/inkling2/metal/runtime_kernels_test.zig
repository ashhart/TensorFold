//! Host-only proof of Mini's runtime source identity, launch ABI, block layout and rounding.
const std = @import("std");
const p = @import("runtime_plan.zig");
const model = @import("model.zig");
const records = @import("core").contract.inkling2_runtime_metal_records.records;
const shader = @import("inkling2_metal").runtime;
const t = std.testing;

fn near(want: f64, got: f64) !void {
    try t.expectApproxEqAbs(want, got, 1e-12);
}

test "runtime source is the one exported source and every helper has one launch record" {
    try t.expectEqualStrings(shader, model.runtime_source);
    try t.expectEqualStrings(shader, @import("kernel_sources").inkling2_runtime);
    try t.expectEqual(std.enums.values(p.Kind).len, model.runtime_names.len);
    try t.expectEqual(std.enums.values(p.Kind).len, records.len);
    for (std.enums.values(p.Kind), records, model.runtime_names) |kind, record, runtime_name| {
        try t.expectEqualStrings(kind.label(), record.name);
        try t.expectEqualStrings(kind.symbol(), runtime_name);
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, kind.symbol()));
        try t.expectEqual(p.threads, record.threads);
        try t.expectEqual(record.threads, record.reserved);
        try t.expectEqual(@as(u32, 0), record.tg_memory);
        var hdr: [96]u8 = undefined;
        const reserved = try std.fmt.bufPrint(&hdr, "[[max_total_threads_per_threadgroup({d})]]\nkernel void {s}(", .{ record.threads, kind.symbol() });
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, reserved));
    }
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, "#pragma METAL fp contract(off)"));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, shader, "struct InkRuntimeArgs { uint count, boundary; };"));
}

test "linear launches bind H=2304 rows and the exact eight-byte argument ABI" {
    try t.expectEqual(@as(usize, 8), @sizeOf(p.RuntimeArgs));
    try t.expectEqual(@as(usize, 0), @offsetOf(p.RuntimeArgs, "count"));
    try t.expectEqual(@as(usize, 4), @offsetOf(p.RuntimeArgs, "boundary"));
    for (std.enums.values(p.Kind)) |kind| {
        const call = p.Call{ .rows = 7, .n = p.hidden, .k = if (kind == .attention_partial) 4 else 1 };
        try t.expect(p.valid(kind, call));
        const a = p.args(kind, call);
        try t.expectEqual(@as(u32, 7 * p.hidden), a.count);
        try t.expectEqual(@as(u32, if (kind == .attention_partial) 4 else if (kind == .capture_tap) 1 else 0), a.boundary);
        try t.expectEqual(@as(usize, 7 * p.hidden), p.geometry(call).count);
        try t.expectEqual(p.threads, p.geometry(call).threads);
        try t.expectEqual(@as(u32, 0), p.geometry(call).tg_memory);
        const outputs: usize = 7 * p.hidden * (if (kind == .capture_tap) @as(usize, p.tap_channels) else 1);
        try t.expectEqual(outputs, p.outputCount(kind, call));
        try t.expect(!p.valid(kind, .{ .rows = 0, .n = p.hidden, .k = call.k }));
        if (kind == .copy_bf16 or kind == .copy_f32) {
            try t.expect(p.valid(kind, .{ .rows = 5, .n = 1024, .k = 1 }));
        } else {
            try t.expect(!p.valid(kind, .{ .rows = 1, .n = p.hidden - 1, .k = call.k }));
        }
    }
    try t.expect(!p.valid(.copy_bf16, .{ .rows = 1, .n = 0, .k = 1 }));
    try t.expect(p.valid(.attention_partial, .{ .rows = 1, .n = p.hidden, .k = 0 }));
    try t.expect(!p.valid(.attention_partial, .{ .rows = 1, .n = p.hidden, .k = 5 }));
    try t.expect(!p.valid(.widen, .{ .rows = 1, .n = p.hidden, .k = 0 }));
    try t.expect(p.valid(.capture_tap, .{ .rows = 1, .n = p.hidden, .k = 3 }));
    try t.expect(!p.valid(.capture_tap, .{ .rows = 1, .n = p.hidden, .k = 4 }));
}

test "DSpark capture channels are packed [row][tap][H] in visit order" {
    try t.expectEqual(@as(usize, 4), p.tap_channels);
    try t.expectEqual(@as(usize, 0), p.tapOffset(0, 0));
    try t.expectEqual(p.hidden, p.tapOffset(0, 1));
    try t.expectEqual(3 * p.hidden, p.tapOffset(0, 3));
    try t.expectEqual(4 * p.hidden, p.tapOffset(1, 0));
    try t.expectEqual(7 * p.hidden, p.tapOffset(1, 3));
    try t.expectEqual(2 * 4 * p.hidden, p.outputCount(.capture_tap, .{ .rows = 2, .n = p.hidden, .k = 0 }));
    try t.expectEqual(2 * p.hidden, p.args(.capture_tap, .{ .rows = 2, .n = p.hidden, .k = 0 }).count);
}

test "four completed BF16 blocks are row-major and disjoint from the partial source" {
    try t.expectEqual(@as(usize, 0), p.blockOffset(0, 1));
    try t.expectEqual(3 * p.hidden, p.blockOffset(0, 4));
    try t.expectEqual(4 * p.hidden, p.blockOffset(1, 1));
    try t.expectEqual(15 * p.hidden, p.blockOffset(3, 4));
    var previous: usize = 0;
    for (0..4) |row| for (1..5) |boundary| {
        const at = p.blockOffset(row, boundary);
        if (at != 0) try t.expectEqual(previous + p.hidden, at);
        previous = at;
    };
    try t.expectEqual(4 * 4 * p.hidden, previous + p.hidden);
}

test "attention restart and addition references cover every valid boundary" {
    const partial: f32 = 1.5;
    const attention: f32 = -0.25;
    try near(1.25, p.attentionPartial(partial, attention, 0));
    for (1..5) |boundary| try near(-0.25, p.attentionPartial(partial, attention, boundary));
    // One binary32 addition rounds this midpoint to one; unrounded FP64 addition would not.
    const ulp: f32 = 0x1p-24;
    try near(1, p.attentionPartial(1, ulp, 0));
    try t.expect(p.attentionPartial(1, ulp, 0) != 1 + @as(f64, ulp));
    for (1..5) |boundary| try near(@as(f64, ulp), p.attentionPartial(std.math.nan(f32), ulp, boundary));
}

test "MLP partial rounds its FP32 sum to nearest-even BF16" {
    try near(1, p.roundBf16(@as(f32, 1.00390625)));
    try near(1.015625, p.roundBf16(@as(f32, 1.01171875)));
    try near(1, p.mlpPartial(1, 0.00390625));
    try near(1.015625, p.mlpPartial(1, 0.01171875));
    try t.expect(p.mlpPartial(1, 0.00390625) != 1.00390625);
}

test "copy and widening witnesses preserve actual BF16 and FP32 values" {
    const bf16 = @import("bf16_round");
    const bf_values = [_]f32{ -0.0, 1.0, -2.5, 0.0078125, std.math.inf(f32) };
    for (bf_values) |x| {
        const bits = bf16.bits(x);
        const stored: f32 = @bitCast(@as(u32, bits) << 16);
        try t.expectEqual(@as(u32, @bitCast(x)), @as(u32, bits) << 16);
        try near(x, stored);
    }
    const f32_values = [_]f32{ -0.0, 1.00000011920928955078125, -8192.25 };
    for (f32_values) |x| try t.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(@as(f32, @floatCast(@as(f64, x))))));
}
