//! System One answers from Clef logits: Python's round(x, 4), the FP32 softmax, request order, ties, raw, refusals.
const std = @import("std");
const json = @import("json");
const enc = @import("clef_encode.zig");
const answer = @import("clef_answer.zig");
const Allocator = std.mem.Allocator;

test "Clef answer: round4 is Python's round(x, 4) on the double's exact value, ties to even" {
    // expected values from CPython's round(x, 4); the inputs are hex so each is the exact double meant
    const cases = [_][2]f64{
        .{ 0x1.0000000000000p-5, 0.0312 },   .{ 0x1.a36e2eb1c432dp-15, 0.0001 }, .{ 0x1.a36e2eb1c432cp-15, 0.0 },
        .{ 0x1.a36e2eb1c432ep-15, 0.0001 },  .{ 0x1.3a92a30553261p-13, 0.0001 }, .{ 0x1.0624dd2f1a9fcp-12, 0.0003 },
        .{ 0x1.fff972474538fp-1, 1.0 },      .{ 0x1.fff972474538ep-1, 0.9999 },  .{ 0x1.0000000000000p-1, 0.5 },
        .{ 1.0, 1.0 },                       .{ 0.0, 0.0 },                      .{ 0x1.f9a6b50b0f27cp-4, 0.1235 },
        .{ 0x1.a358b522e2024p-15, 0.0 },     .{ 0x1.0001a36e2eb1cp+1, 2.0 },     .{ 0x1.1fff972474539p+3, 9.0 },
        .{ 0x1.2000000000000p+2, 4.5 },      .{ 0x1.56e1fc2f8f359p-997, 0.0 },   .{ 0x0.0000000000001p-1022, 0.0 },
        .{ 0x1.999999999999ap-4, 0.1 },      .{ 0x1.3333333333334p-2, 0.3 },     .{ -0x1.a36e2eb1c432dp-15, -0.0001 },
        .{ -0x1.0000000000000p-5, -0.0312 }, .{ 0x1.4b9ad00000000p-2, 0.3238 },  .{ 0x1.34f06a0000000p-3, 0.1508 },
        .{ 0x1.4d47480000000p-1, 0.6509 },   .{ 0x1.28b2f40000000p-4, 0.0724 },  .{ 0x1.125f200000000p-1, 0.5359 },
        .{ 0x1.7677280000000p-2, 0.3657 },   .{ 0x1.db20900000000p-5, 0.058 },   .{ 0x1.03ce9e0000000p-1, 0.5074 },
        .{ 0x1.332a180000000p-5, 0.0375 },   .{ 0x1.bc0d9e0000000p-2, 0.4336 },  .{ 0x1.1e20b80000000p-4, 0.0699 },
        .{ 0x1.738f7e0000000p-4, 0.0907 },   .{ 0x1.e90bcd0000000p+1, 3.8207 },  .{ 0x1.dc444f4000000p+2, 7.4417 },
        .{ 0x1.1d3d5e4000000p+0, 1.1142 },   .{ 0x1.012bd92000000p+1, 2.0092 },
    };
    for (cases) |c| {
        const got = answer.round4(c[0]);
        if (@as(u64, @bitCast(got)) != @as(u64, @bitCast(c[1]))) {
            std.debug.print("round4({x}) = {d}, want {d}\n", .{ c[0], got, c[1] });
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expect(std.math.signbit(answer.round4(-0x1p-20)));
}

test "Clef answer: the softmax widens BF16 to FP32, sums to one and refuses non-finite logits" {
    var p: [4]f32 = undefined;
    try answer.probabilities(&.{ 0x3f80, 0x3f80, 0x3f80, 0x3f80 }, &p);
    for (p) |x| try std.testing.expectEqual(@as(f32, 0.25), x);
    try answer.probabilities(&.{ 0x0000, 0x3f80, 0xc000, 0x4040 }, &p);
    try std.testing.expect(p[3] > p[1] and p[1] > p[0] and p[0] > p[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 1), p[0] + p[1] + p[2] + p[3], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, @exp(@as(f32, -3)) / (1 + @exp(@as(f32, -2)) + @exp(@as(f32, -3)) + @exp(@as(f32, -5)))), p[0], 1e-6);
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc0 }) |bad| try std.testing.expectError(error.NonFinite, answer.probabilities(&.{ 0x3f80, bad }, p[0..2]));
}

/// A record whose questions carry only option ids, as `body` reads them.
fn record(a: Allocator, fields: []const enc.Field) !enc.Record {
    const qs = try a.alloc(enc.Question, fields.len);
    for (fields, qs) |f, *q| {
        const opts = try enc.options(a, f.criteria);
        const ids = try a.alloc([]const u8, opts.len);
        for (opts, ids) |o, *i| i.* = o.id;
        q.* = .{ .id = f.id, .kind = f.kind(), .span = .{ 0, 1 }, .options = &.{}, .option_ids = ids };
    }
    return .{ .ids = &.{ 1, 2, 3, 4, 5 }, .questions = qs, .state_rows = 1 };
}

fn reply(a: Allocator, questions: []const u8, logits: []const u16, with_raw: bool) ![]const u8 {
    const fields = try enc.fields(a, (try json.parseText(a, questions)).ok);
    const v = try answer.body(a, .{ .string = "clef" }, fields, try record(a, fields), logits, with_raw);
    return json.stringify(a, v, .{ .ascii = false, .compact = true });
}

test "Clef answer: request order, the first best option on a tie, score sums and the 0.03125 tie" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // choice "c": encoding order a, b, z; request order z, b, a; b and a tie and b comes first in the request
    const questions =
        \\{"n":{"type":"noul"},"c":{"type":"choice","criteria":{"z":"Z","b":"B","a":"A"}},
        \\"s":{"type":"score","criteria":["lo","mid","hi","top"]}}
    ;
    const logits = [_]u16{ 0x0000, 0x0000, 0x3f80, 0x3f80, 0xc2c8, 0x0000, 0x0000, 0x0000, 0x0000 };
    const want = "{\"model\":\"clef\",\"answers\":{\"n\":{\"type\":\"noul\",\"noul\":0.5}," ++
        "\"c\":{\"type\":\"choice\",\"choice\":\"b\",\"confidence\":0.5,\"probabilities\":{\"z\":0.0,\"b\":0.5,\"a\":0.5}}," ++
        "\"s\":{\"type\":\"score\",\"score\":1.5,\"confidence\":0.25,\"legend\":{\"0\":\"lo\",\"1\":\"mid\",\"2\":\"hi\",\"3\":\"top\"}," ++
        "\"probabilities\":{\"0\":0.25,\"1\":0.25,\"2\":0.25,\"3\":0.25}}},\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}";
    try std.testing.expectEqualStrings(want, try reply(a, questions, &logits, false));
    // 32 equal options: each probability is 0.03125 exactly, which Python rounds to 0.0312
    var many: std.ArrayList(u8) = .empty;
    try many.appendSlice(a, "{\"q\":{\"type\":\"choice\",\"criteria\":{");
    for (0..32) |i| try many.print(a, "{s}\"o{d:0>2}\":null", .{ if (i == 0) "" else ",", 31 - i });
    try many.appendSlice(a, "}}}");
    const flat: [32]u16 = @splat(0x4000);
    const text = try reply(a, many.items, &flat, true);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"choice\":\"o31\",\"confidence\":0.0312,\"probabilities\":{\"o31\":0.0312,") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"raw\":{\"q\":{\"type\":\"choice\",\"options\":[\"o00\",\"o01\",") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"probs\":[0.03125,0.03125,") != null);
}

test "Clef answer: a strict maximum wins over request order, and a non-finite logit fails the body" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const questions = "{\"c\":{\"type\":\"choice\",\"criteria\":{\"zeta\":1,\"alpha\":2}}}";
    // encoding order alpha, zeta: alpha's logit 1.0 beats zeta's 0.0 though zeta comes first in the request
    const text = try reply(a, questions, &.{ 0x3f80, 0x0000 }, false);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"choice\":\"alpha\",\"confidence\":0.7311,\"probabilities\":{\"zeta\":0.2689,\"alpha\":0.7311}") != null);
    try std.testing.expectError(error.NonFinite, reply(a, questions, &.{ 0x7fc0, 0x0000 }, false));
}
