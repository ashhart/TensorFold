//! The recorded expert down at 4 bits in groups of 32 with its second input chunk spread over every lane (same bits).
const std = @import("std");
const Const = @import("variants.zig").Const;

/// down_rows<4, 32, 640> with inputs 512.. (lanes 0-7) computed two rows a lane and shuffled back to lanes 0-7.
const helper =
    \\// down_rows<4, 32, 640>, inputs 512.. on every lane and shuffled back: the same partials, order and sums
    \\inline void down_rows_bal(const device uint32_t* W, const device bfloat* S, const device bfloat* B, size_t first,
    \\                          const device bfloat* x, uint lane, thread float* out) {
    \\  constexpr int RB = 320, KG = 20;
    \\  const int m = int(lane & 7);
    \\  float xv0[16], xv1[16], sum0 = 0.0f, sum1 = 0.0f;
    \\  for (int i = 0; i < 16; i++) { xv0[i] = float(x[int(lane) * 16 + i]); sum0 += xv0[i]; }
    \\  for (int i = 0; i < 16; i++) { xv1[i] = float(x[512 + m * 16 + i]); sum1 += xv1[i]; }
    \\  float p1[2];
    \\  for (int t = 0; t < 2; t++) {
    \\    const int row = 2 * int(lane >> 3) + t;
    \\    float q[16];
    \\    lane_codes<4, 16>((const device uint8_t*)W + (first + row) * RB, 512 + m * 16, q);
    \\    float d = 0.0f;
    \\    for (int i = 0; i < 16; i++) d = fma(q[i], xv1[i], d);
    \\    const size_t at = (first + row) * KG + (512 + m * 16) / 32;
    \\    p1[t] = fma(float(S[at]), d, float(B[at]) * sum1);
    \\  }
    \\  for (int row = 0; row < 8; row++) {
    \\    float q[16];
    \\    lane_codes<4, 16>((const device uint8_t*)W + (first + row) * RB, int(lane) * 16, q);
    \\    float d = 0.0f;
    \\    for (int i = 0; i < 16; i++) d = fma(q[i], xv0[i], d);
    \\    const size_t at = (first + row) * KG + int(lane) / 2;
    \\    float acc = 0.0f;
    \\    acc += fma(float(S[at]), d, float(B[at]) * sum0);
    \\    const float q1 = simd_shuffle(p1[row & 1], ushort(8 * (row >> 1) + m));
    \\    if (lane < 8) acc += q1;
    \\    out[row] = acc;
    \\  }
    \\}
    \\
;

const calls = [2][2][]const u8{
    .{ "down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out)", "down_rows_bal(SDW, SDS, SDB, size_t(d0), x, lane, out)" },
    .{ "down_rows<WB, WG, NI>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out)", "down_rows_bal(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out)" },
};

/// Whether `text` patched at `consts` is the recorded expert down with every width 4/32 over 640 inputs.
pub fn applies(text: []const u8, consts: []const Const) bool {
    if (consts.len != 4) return false;
    for (consts) |c| {
        const want: usize = if (std.mem.endsWith(u8, c.name, "WB")) 4 else if (std.mem.endsWith(u8, c.name, "WG")) 32 else return false;
        if (c.value != want) return false;
    }
    if (std.mem.count(u8, text, "constexpr int NI = 640;") != 1) return false;
    for (calls) |c| if (std.mem.count(u8, text, c[0]) != 1) return false;
    return std.mem.count(u8, text, "[[kernel]]") == 1;
}

/// `text` with down_rows_bal before its kernel and both down_rows calls taken by it (`applies` holds).
pub fn balanced(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out = try std.mem.replaceOwned(u8, a, text, "[[kernel]]", helper ++ "[[kernel]]");
    for (calls) |c| out = try std.mem.replaceOwned(u8, a, out, c[0], c[1]);
    return out;
}

test "the recorded expert down takes the balanced rows only at 4/32 everywhere" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        if (std.mem.indexOf(u8, src.name, "qa_expert_gateup") != null) try std.testing.expect(!applies(src.text, &.{ .{ .name = "WB", .value = 4 }, .{ .name = "WG", .value = 32 }, .{ .name = "SWB", .value = 4 }, .{ .name = "SWG", .value = 32 } }));
        if (std.mem.indexOf(u8, src.name, "qa_expert_down_y") == null) continue;
        seen += 1;
        const at4 = [_]Const{ .{ .name = "WB", .value = 4 }, .{ .name = "WG", .value = 32 }, .{ .name = "SWB", .value = 4 }, .{ .name = "SWG", .value = 32 } };
        try std.testing.expect(applies(src.text, &at4));
        const out = try balanced(a, src.text);
        for (calls) |c| try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, out, c[0]));
        try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, out, "down_rows_bal("));
        var six = at4;
        six[2].value = 6;
        try std.testing.expect(!applies(src.text, &six));
        try std.testing.expect(!applies(src.text, at4[0..2]));
    }
    try std.testing.expectEqual(@as(usize, 1), seen);
}
