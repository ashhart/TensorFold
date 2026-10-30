//! Exact patches of the recorded expert kernels at 4 bits in groups of 32: half-nibble codes and the balanced down.
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

/// lane_codes' first loop lines: the half-nibble branch goes in before them.
const codes_loop = "  for (int i = 0; i < VPT; i++) {\n    const int bit = BITS * i, j = bit / 16, s = bit % 16;";

/// The half-nibble branch's own comment: a text that has it is patched already.
const half_mark = "half nibbles: the same 16 values";

/// lane_codes at 4 bits as fn_rows' FZ_NIB 1 makes nibbles (two a half2 op): the same 16 values in the same order.
const half_codes =
    \\  if (BITS == 4 && VPT == 16) { // half nibbles: the same 16 values
    \\    for (int j = 0; j < 4; j += 2) {
    \\      const uint w = uint(u[j]) | (uint(u[j + 1]) << 16);
    \\      const half2 a = as_type<half2>(0x64006400u | (w & 0x000F000Fu)) - half2(1024.0h);
    \\      const half2 b = as_type<half2>(0x64006400u | ((w >> 4) & 0x000F000Fu)) - half2(1024.0h);
    \\      const half2 c = as_type<half2>(0x64006400u | ((w >> 8) & 0x000F000Fu)) - half2(1024.0h);
    \\      const half2 d = as_type<half2>(0x64006400u | ((w >> 12) & 0x000F000Fu)) - half2(1024.0h);
    \\      thread float* o = q + 4 * j;
    \\      o[0] = float(a.x); o[1] = float(b.x); o[2] = float(c.x); o[3] = float(d.x);
    \\      o[4] = float(a.y); o[5] = float(b.y); o[6] = float(c.y); o[7] = float(d.y);
    \\    }
    \\    return;
    \\  }
    \\
;

/// Every expert width in `consts` (WB, WG, SWB, SWG) is 4 bits in groups of 32.
fn q4(consts: []const Const) bool {
    if (consts.len != 4) return false;
    for (consts) |c| {
        const want: usize = if (std.mem.endsWith(u8, c.name, "WB")) 4 else if (std.mem.endsWith(u8, c.name, "WG")) 32 else return false;
        if (c.value != want) return false;
    }
    return true;
}

/// Whether `text` at `consts` is a recorded expert kernel whose lane_codes can take half nibbles.
pub fn halfApplies(text: []const u8, consts: []const Const) bool {
    return q4(consts) and std.mem.count(u8, text, codes_loop) == 1 and std.mem.indexOf(u8, text, half_mark) == null;
}

/// `text` with the half-nibble branch at the top of lane_codes' loop (`halfApplies` holds).
pub fn half(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    return std.mem.replaceOwned(u8, a, text, codes_loop, half_codes ++ codes_loop);
}

/// The gate/up kernel's last parameter and its row line: a row count from buffer 17 goes in after each.
const gate_param = "  device float* WTS [[buffer(16)]],\n";
const gate_row = "  const int r = p / SLOTS, slot = p % SLOTS;\n";
const gate_mark = "GROWS";

/// Whether `text` at `consts` is the recorded 4/32 expert gate/up, not yet taking a row count.
pub fn gateApplies(text: []const u8, consts: []const Const) bool {
    return q4(consts) and std.mem.count(u8, text, gate_param) == 1 and std.mem.count(u8, text, gate_row) == 1 and
        std.mem.indexOf(u8, text, gate_mark) == null;
}

/// `text` reading its row count from buffer 17: pairs of rows from the count on (a GPU-gated window's) do nothing.
pub fn gated(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    const p = try std.mem.replaceOwned(u8, a, text, gate_param, gate_param ++ "  const constant int32_t* GROWS [[buffer(17)]],\n");
    return std.mem.replaceOwned(u8, a, p, gate_row, gate_row ++ "  if (r >= GROWS[0]) return;\n");
}

const calls = [2][2][]const u8{
    .{ "down_rows<SWB, SWG, NI>(SDW, SDS, SDB, size_t(d0), x, lane, out)", "down_rows_bal(SDW, SDS, SDB, size_t(d0), x, lane, out)" },
    .{ "down_rows<WB, WG, NI>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out)", "down_rows_bal(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out)" },
};

/// Whether `text` patched at `consts` is the recorded expert down with every width 4/32 over 640 inputs.
pub fn applies(text: []const u8, consts: []const Const) bool {
    if (!q4(consts)) return false;
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

test "both recorded expert kernels take half nibbles at 4/32 only, once" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const at4 = [_]Const{ .{ .name = "WB", .value = 4 }, .{ .name = "WG", .value = 32 }, .{ .name = "SWB", .value = 4 }, .{ .name = "SWG", .value = 32 } };
    var six = at4;
    six[0].value = 6;
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        if (std.mem.indexOf(u8, src.name, "qa_expert_") == null) continue;
        seen += 1;
        try std.testing.expect(halfApplies(src.text, &at4));
        try std.testing.expect(!halfApplies(src.text, &six));
        const out = try half(arena.allocator(), src.text);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, half_mark));
        try std.testing.expect(!halfApplies(out, &at4)); // never twice
    }
    try std.testing.expectEqual(@as(usize, 2), seen);
}

test "the recorded 4/32 expert gate/up takes a row count from buffer 17, once; the down is left alone" {
    const ks = @import("kernel_sources");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const at4 = [_]Const{ .{ .name = "WB", .value = 4 }, .{ .name = "WG", .value = 32 }, .{ .name = "SWB", .value = 4 }, .{ .name = "SWG", .value = 32 } };
    var seen: usize = 0;
    for (ks.flashnext_gen.sources) |src| {
        if (std.mem.indexOf(u8, src.name, "qa_expert_down_y") != null) try std.testing.expect(!gateApplies(src.text, &at4));
        if (std.mem.indexOf(u8, src.name, "qa_expert_gateup") == null) continue;
        seen += 1;
        try std.testing.expect(gateApplies(src.text, &at4));
        const out = try gated(arena.allocator(), src.text);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "[[buffer(17)]]"));
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "if (r >= GROWS[0]) return;"));
        try std.testing.expect(!gateApplies(out, &at4));
    }
    try std.testing.expectEqual(@as(usize, 1), seen);
}
