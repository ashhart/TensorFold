const std = @import("std");
const model = @import("loop_model.zig");

const Driver = model.Driver;
const GateWeights = model.GateWeights;
const MessageBank = model.MessageBank;
const MessageQueries = model.MessageQueries;
const attentionResidual = model.attentionResidual;
const bf16 = model.bf16;
const combineExperts = model.combineExperts;
const crossPassMessage = model.crossPassMessage;
const crossPassMessages = model.crossPassMessages;
const e2m1 = model.e2m1;
const e4m3 = model.e4m3;
const expertActivation = model.expertActivation;
const expertProjection = model.expertProjection;
const hidden_size = model.hidden_size;
const messageGate = model.messageGate;
const mxfp8ReferenceRow = model.mxfp8ReferenceRow;
const mxfp8ReferenceWeight = model.mxfp8ReferenceWeight;
const mxfp8Scale = model.mxfp8Scale;
const nvfp4OuterFactor = model.nvfp4OuterFactor;
const nvfp4ReferenceRow = model.nvfp4ReferenceRow;
const nvfp4ReferenceWeight = model.nvfp4ReferenceWeight;
const residualScale = model.residualScale;
const rms = model.rms;
const visit = model.visit;
const visit_count = model.visit_count;
const weightedRms = model.weightedRms;

fn widen32(v: f32) f64 {
    return @floatCast(v);
}

fn expectClose(actual: f64, expected: f64, tolerance: f64) !void {
    try std.testing.expectApproxEqAbs(expected, actual, tolerance);
}

test "norm scale and weighted RMS's sole final bf16 rounding" {
    const x = [_]f64{ 3, 4, 0, 0 };
    var out: [4]f64 = undefined;
    var scratch: [4]f64 = undefined;
    try rms(&x, 0, &out);
    try std.testing.expectEqual(@as(f64, widen32(@as(f32, 1.2))), out[0]);
    try std.testing.expectEqual(@as(f64, widen32(@as(f32, 1.6))), out[1]);
    try weightedRms(&x, &.{ 2, 0.5, 1, 1 }, 0, &out, &scratch);
    // Hand: rho=[1.2,1.6,0,0], then [2.4,0.8,0,0]. BF16(0.8) is 0.80078125.
    try std.testing.expectEqualSlices(f64, &.{ 2.40625, 0.80078125, 0, 0 }, &out);
}

test "attention residual normalizes values and applies binary32 inverse-root-H after its dot" {
    const state = [_]f64{ 3, 4, 0, 0 };
    const partial = [_]f64{ 0, 10, 0, 0 };
    const query = [_]f64{ 1, 0, 0, 0 };
    var out: [4]f64 = undefined;
    try attentionResidual(std.testing.allocator, &state, &partial, &query, 1, 0, &out);
    // rho(S)=[1.2,1.6,0,0], rho(p)=[0,2,0,0], logits .6 and zero; w0=1/(1+exp(-.6)), output=w0*rho(S)+(1-w0)*rho(p).
    const w0 = 1 / (1 + @exp(@as(f64, -0.6)));
    try expectClose(out[0], 1.2 * w0, 2e-7);
    try expectClose(out[1], 1.6 * w0 + 2 * (1 - w0), 2e-7);
    try std.testing.expectEqual(@as(u32, 0x3caaaaab), @as(u32, @bitCast(try residualScale(hidden_size))));
}

test "message bank normalizes exactly its two states and checkpoint query rows are role-interleaved" {
    const a = [_]f64{ 3, 4, 0, 0 };
    const p = [_]f64{ 0, 10, 0, 0 };
    var out: [4]f64 = undefined;
    try crossPassMessage(std.testing.allocator, .{ .block = &a, .partial = &p }, &.{ 1, 0, 0, 0 }, 0, &out);
    const w0 = 1 / (1 + @exp(@as(f64, -0.6)));
    try expectClose(out[0], 1.2 * w0, 2e-7);
    try expectClose(out[1], 1.6 * w0 + 2 * (1 - w0), 2e-7);

    var q_rows: [24]f64 = undefined;
    var kv_rows: [24]f64 = undefined;
    for (&q_rows, 0..) |*v, i| v.* = @floatFromInt(i);
    for (&kv_rows, 0..) |*v, i| v.* = @floatFromInt(100 + i);
    const queries: MessageQueries = .{ .q_mlp = &q_rows, .kv = &kv_rows };
    try std.testing.expectEqual(@as(f64, 4), (try queries.row(2, .q, 1))[0]);
    try std.testing.expectEqual(@as(f64, 5), (try queries.row(2, .mlp, 1))[0]);
    try std.testing.expectEqual(@as(f64, 104), (try queries.row(2, .k, 1))[0]);
    try std.testing.expectEqual(@as(f64, 105), (try queries.row(2, .v, 1))[0]);

    var messages: [visit_count]f64 = undefined;
    const one_bank: MessageBank = .{ .block = &.{3}, .partial = &.{4} };
    try crossPassMessages(std.testing.allocator, one_bank, queries, 0, &messages);
    for (messages) |message| try std.testing.expectEqual(@as(f64, 1), message);
}

test "rank gate rounds both joined norms and its down projection before f32 forget and gain" {
    const h = [_]f64{ 3, 4, 0, 0 };
    const m = [_]f64{ 0, 10, 0, 0 };
    const weights: GateWeights = .{
        .down = &.{ 0.5, 1.5, 0, 0, 0.5, 1.0078125, 0, 0 },
        .forget = &.{ 1, 0, 0, 0 },
        .gain = &.{ 0, 1, 0, 0 },
        .rank = 1,
    };
    var out: [4]f64 = undefined;
    try messageGate(std.testing.allocator, &h, &m, weights, 0, &out);
    // bf16 rho(h)=[1.203125,1.6015625,0,0], rho(m)=[0,2,0,0]; D*n=5.01953125 rounds to z=5.03125 (else 5.0).
    const z: f32 = 5.03125;
    const a: f32 = std.math.tanh(z);
    const expected0: f32 = (1 - a) * 3;
    const expected1: f32 = 4 + a * 10;
    try std.testing.expectEqual(@as(f64, widen32(expected0)), out[0]);
    try std.testing.expectEqual(@as(f64, widen32(expected1)), out[1]);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0 }, out[2..4]);
}

test "48 visits reset from the prelude and bank B12 plus final partial of only the previous pass" {
    var driver = try Driver.init(std.testing.allocator, &.{7});
    defer driver.deinit();
    const zero = [_]f64{0};
    for (0..visit_count) |v| {
        _ = try driver.begin(v);
        if (v == 6 or v == 18 or v == 30) {
            const src = try driver.attentionSources();
            try std.testing.expectEqual(@as(usize, 1), src.blocks.len);
            try std.testing.expectEqual(@as(f64, 7), src.blocks[0]);
            try std.testing.expectEqual(@as(f64, 21), src.partial[0]); // sum of attention deltas 1...6
        }
        if (v == 18) {
            const bank = try driver.messageBank();
            try std.testing.expectEqual(@as(f64, 57), bank.block[0]); // sum 7...12
            try std.testing.expectEqual(@as(f64, 93), bank.partial[0]); // sum 13...18
        } else if (v == 30) {
            const bank = try driver.messageBank();
            try std.testing.expectEqual(@as(f64, 129), bank.block[0]); // sum 19...24
            try std.testing.expectEqual(@as(f64, 165), bank.partial[0]); // sum 25...30
        }
        const delta = [_]f64{@floatFromInt(v + 1)};
        try driver.afterAttention(&delta);
        try driver.finish(&zero);
    }
    const final = try driver.finalSources();
    // Pass 3 supplies the last two core blocks. BF16 rounds the final exact sum 273 to even 272.
    try std.testing.expectEqualSlices(f64, &.{ 7, 21, 201, 237 }, final.blocks);
    try std.testing.expectEqual(@as(f64, 272), final.partial[0]);
}

test "visit table preserves layer kinds, AR counts, private visit slots and reference slots" {
    const expected = [_]struct { index: usize, layer: u8, pass: u8, reference: u8, messages: bool, ar: u8, attn: u8, mlp: u8 }{
        .{ .index = 0, .layer = 0, .pass = 0, .reference = 0, .messages = false, .ar = 0, .attn = 0, .mlp = 2 },
        .{ .index = 5, .layer = 5, .pass = 0, .reference = 5, .messages = false, .ar = 3, .attn = 2, .mlp = 2 },
        .{ .index = 6, .layer = 6, .pass = 1, .reference = 6, .messages = false, .ar = 1, .attn = 2, .mlp = 3 },
        .{ .index = 17, .layer = 17, .pass = 1, .reference = 17, .messages = false, .ar = 3, .attn = 4, .mlp = 4 },
        .{ .index = 18, .layer = 6, .pass = 2, .reference = 24, .messages = true, .ar = 1, .attn = 2, .mlp = 3 },
        .{ .index = 29, .layer = 17, .pass = 2, .reference = 35, .messages = true, .ar = 3, .attn = 4, .mlp = 4 },
        .{ .index = 30, .layer = 6, .pass = 3, .reference = 36, .messages = true, .ar = 1, .attn = 2, .mlp = 3 },
        .{ .index = 41, .layer = 17, .pass = 3, .reference = 47, .messages = true, .ar = 3, .attn = 4, .mlp = 4 },
        .{ .index = 42, .layer = 18, .pass = 0, .reference = 18, .messages = false, .ar = 1, .attn = 4, .mlp = 5 },
        .{ .index = 47, .layer = 23, .pass = 0, .reference = 23, .messages = false, .ar = 3, .attn = 5, .mlp = 5 },
    };
    for (expected) |e| {
        const v = try visit(e.index);
        try std.testing.expectEqual(e.layer, v.layer);
        try std.testing.expectEqual(e.pass, v.pass);
        try std.testing.expectEqual(e.reference, v.reference_slot);
        try std.testing.expectEqual(@as(u8, @intCast(e.index)), v.cache_slot);
        try std.testing.expectEqual(e.messages, v.has_messages);
        try std.testing.expectEqual(e.ar, v.ar_queries);
        try std.testing.expectEqual(e.attn, v.attention_sources);
        try std.testing.expectEqual(e.mlp, v.mlp_sources);
    }
    try std.testing.expectError(error.BadVisit, visit(48));
}

test "NVFP4 uses bf16 amax and denominator and low nibble before high nibble" {
    // 2689 rounds to bf16 2688, so the reference factor is exactly one rather than 2689/2688.
    try std.testing.expectEqual(@as(f64, 1), nvfp4OuterFactor(2689, 2688));
    try std.testing.expectEqual(@as(f64, 1.5), nvfp4ReferenceWeight(3, 0x38, 2689, 2688));
    var row: [16]f64 = undefined;
    var codes: [8]u8 = @splat(0);
    codes[0] = 0xb3;
    const scales = [_]u8{0x38};
    try nvfp4ReferenceRow(&codes, &scales, 2689, 2688, &row);
    try std.testing.expectEqual(@as(f64, 1.5), row[0]);
    try std.testing.expectEqual(@as(f64, -1.5), row[1]);
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3(1));
    try std.testing.expectEqual(@as(f64, -0.0), e2m1(8));
}

test "MXFP8 exponent is UE8M0 and each stored scale covers one row's 32 inputs" {
    try std.testing.expectEqual(@as(f64, 8), mxfp8Scale(130));
    try std.testing.expectEqual(@as(f64, 12), mxfp8ReferenceWeight(0x3c, 130)); // 1.5 * 2^3
    var codes: [64]u8 = @splat(0);
    codes[0] = 0x38; // one
    codes[31] = 0x3c; // 1.5, still group zero
    codes[32] = 0x3c; // group one
    var out: [64]f64 = undefined;
    try mxfp8ReferenceRow(&codes, &.{ 130, 129 }, &out);
    try std.testing.expectEqual(@as(f64, 8), out[0]);
    try std.testing.expectEqual(@as(f64, 12), out[31]);
    try std.testing.expectEqual(@as(f64, 6), out[32]);
}

test "expert projection and softcapped SwiGLU have no activation quantization" {
    var projected: [2]f64 = undefined;
    try expertProjection(&.{ 1, 2, -1, 0.5 }, &.{ 3, 4 }, 2, &projected);
    try std.testing.expectEqualSlices(f64, &.{ 11, -1 }, &projected);
    const g: f32 = 1;
    const expected: f32 = 4 * std.math.tanh(g / 4) * (1 / (1 + @exp(-g))) * (25 * std.math.tanh(g / 25));
    try std.testing.expectEqual(@as(f64, bf16(widen32(expected))), expertActivation(1, 1));
}

test "expert sum is ascending id even when f32 route-order association gives another bf16 result" {
    const ids = [_]u32{ 2, 1, 0 };
    const weights = [_]f64{ 1, 1, 1 };
    // Route order gives -1 + 2^-24 + 1 = 2^-24 in f32; ascending id gives (1 + 2^-24) - 1 = 0 by a tie to even.
    const outputs = [_]f64{ -1, 0x1p-24, 1 };
    var out: [1]f64 = undefined;
    try combineExperts(3, &ids, &weights, &outputs, 1, &out);
    try std.testing.expectEqual(@as(f64, 0), out[0]);
}
