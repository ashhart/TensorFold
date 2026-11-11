//! The host reference against Tencent's dspine_mlx on small widths (testdata/bridge_fixture.py made the fixture).
const std = @import("std");
const core = @import("core");
const host = @import("host.zig");
const t = std.testing;

const blob = @embedFile("testdata/bridge-fixture.safetensors");
const H = 128;
const C = 64;
const R = 64;
const V = 96;
const T = 6;
const K = 4;
const dims: host.Dims = .{ .hidden = H, .code = C, .rank = R, .vocab = V, .eps = 1e-6 };

const Fixture = struct {
    header: core.safetensors.Header,
    data: []const u8,

    fn open(arena: std.mem.Allocator) !Fixture {
        const n = std.mem.readInt(u64, blob[0..8], .little);
        const data = blob[8 + n ..];
        return .{ .header = try core.safetensors.parseHeader(arena, blob[8..][0..n], data.len), .data = data };
    }
    fn bytes(f: *const Fixture, name: []const u8) ![]const u8 {
        const e = f.header.get(name) orelse return error.MissingTensor;
        return f.data[e.begin..e.end];
    }
    fn bf16(f: *const Fixture, name: []const u8) ![]align(1) const u16 {
        return std.mem.bytesAsSlice(u16, try f.bytes(name));
    }
    fn words(f: *const Fixture, name: []const u8) ![]align(1) const u32 {
        return std.mem.bytesAsSlice(u32, try f.bytes(name));
    }
    fn floats(f: *const Fixture, name: []const u8) ![]align(1) const f32 {
        return std.mem.bytesAsSlice(f32, try f.bytes(name));
    }
};

/// Ulps between two bf16 values of one sign (a sign change counts as far).
fn ulps(a: u16, b: u16) u32 {
    if ((a ^ b) & 0x8000 != 0) return if ((a | b) & 0x7fff == 0) 0 else 1 << 16;
    return @abs(@as(i32, a) - @as(i32, b));
}

/// Host and MLX sum in different orders, so a bf16 may land an ulp or two apart; at most `few` per mille may.
fn near(want: []align(1) const u16, got: []const u16, few: usize) !void {
    try t.expectEqual(want.len, got.len);
    var off: usize = 0;
    var worst: u32 = 0;
    for (want, got) |a, b| {
        const u = ulps(a, b);
        worst = @max(worst, u);
        off += @intFromBool(u != 0);
    }
    try t.expect(worst <= 2 and off * 1000 <= few * want.len);
}

/// The gate, predecessor_gate and prev weights stacked [3 rank, code] (the caller frees).
fn packedW(f: *const Fixture) ![]u16 {
    const w = try t.allocator.alloc(u16, 3 * R * C);
    for ([_][]const u8{ "gate", "pred", "prev" }, 0..) |name, i| @memcpy(w[i * R * C ..][0 .. R * C], try f.bf16(name));
    return w;
}

fn bridge(f: *const Fixture, packed_w: []const u16, last: bool) !host.Bridge {
    return .{
        .read = if (last) try f.bf16("read_last") else (try f.bf16("read_early"))[0 .. C * H],
        .codebook = try f.bf16("codebook"),
        .packed_w = packed_w,
        .bias = try f.bf16("gate_bias"),
        .write = (try f.bf16("write"))[(if (last) H * R else 0)..][0 .. H * R],
    };
}

test "DSpine host bridge: an early write and the stashed last write as dspine_mlx rounds them" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f = try Fixture.open(arena.allocator());
    const anchor = (try f.words("anchor"))[0];
    var out: [T * H]u16 = undefined;
    var normed: [T * H]u16 = undefined;
    const pw = try packedW(&f);
    defer t.allocator.free(pw);
    const early = try bridge(&f, pw, false);
    try host.write(t.allocator, dims, early, try f.bf16("x0"), anchor, .{ .weight = try f.bf16("norm0"), .eps = 1e-6 }, &out, &normed, null);
    try near(try f.bf16("out0"), &out, 5);
    try near(try f.bf16("normed0"), &normed, 5);
    const last = try bridge(&f, pw, true);
    var raw: [T * H]u16 = undefined;
    var rms: [T]f32 = undefined;
    var packed_rows: [T * 3 * R]u16 = undefined;
    try host.write(t.allocator, dims, last, try f.bf16("x1"), anchor, .{ .weight = try f.bf16("norm1"), .eps = 1e-6 }, &out, &normed, .{ .raw = &raw, .rms = &rms, .packed_rows = &packed_rows });
    try near(try f.bf16("out1"), &out, 5);
    try near(try f.bf16("normed1"), &normed, 5);
    try near(try f.bf16("stash_packed"), &packed_rows, 5);
    for (try f.floats("stash_rms"), rms) |a, b| try t.expectApproxEqRel(a, b, 1e-5);
}

test "DSpine host tables and greedy chain: TG, TSV, top-16 and drafts as dspine_mlx's RefinementTail" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f = try Fixture.open(arena.allocator());
    const pw = try packedW(&f);
    defer t.allocator.free(pw);
    const b = try bridge(&f, pw, true);
    const head = try f.bf16("head");
    const norm1 = try f.bf16("norm1");
    var tg: [V * R]u16 = undefined;
    var tsv: [V * 2 * R]u16 = undefined;
    for (0..V) |v| {
        host.tgRow(dims, head[v * H ..][0..H], norm1, b.write, tg[v * R ..][0..R]);
        host.tsvRow(dims, b, @intCast(v), tsv[v * 2 * R ..][0 .. 2 * R]);
    }
    try near(try f.bf16("tg"), &tg, 5);
    try near(try f.bf16("tsv"), &tsv, 5);
    // The MLX logits and stash (not the host's) feed the tail, so only the tail's own arithmetic is under test.
    const logits = try f.bf16("logits");
    var cand: [(T - 1) * K]u32 = undefined;
    for (0..T - 1) |i| host.topk(logits[i * V ..][0..V], cand[i * K ..][0..K]);
    const want_cand = try f.words("candidates");
    for (want_cand, cand) |a, c| try t.expectEqual(a, c);
    var head_rows: [(T - 1) * K * H]u16 = undefined;
    for (cand, 0..) |id, j| @memcpy(head_rows[j * H ..][0..H], head[id * H ..][0..H]);
    var raw: [T * H]u16 = undefined;
    var packed_rows: [T * 3 * R]u16 = undefined;
    var rms: [T]f32 = undefined;
    for (&raw, try f.bf16("stash_raw")) |*d, s| d.* = s;
    for (&packed_rows, try f.bf16("stash_packed")) |*d, s| d.* = s;
    for (&rms, try f.floats("stash_rms")) |*d, s| d.* = s;
    var scores: [(T - 1) * K * K]f32 = undefined;
    try host.table(.{ .dims = dims, .topk = K, .stash = .{ .raw = &raw, .rms = &rms, .packed_rows = &packed_rows }, .anchor = (try f.words("anchor"))[0], .candidates = &cand, .head_rows = &head_rows, .norm_w = norm1, .bias = b.bias, .tg = try f.bf16("tg"), .tsv = try f.bf16("tsv") }, &scores);
    var drafts: [T - 1]u32 = undefined;
    core.dspine.chain(&cand, &scores, K, &drafts);
    for (try f.words("drafts"), drafts) |a, d| try t.expectEqual(a, d);
}

test "DSpine host top-k: ties go to the lower id, the result ascends" {
    const v = [_]u16{ host.round(1), host.round(3), host.round(3), host.round(2), host.round(3) };
    var out: [2]u32 = undefined;
    host.topk(&v, &out);
    try t.expectEqualSlices(u32, &.{ 1, 2 }, &out);
}

test "DSpine host bridge control: dropping the message changes the written rows" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const f = try Fixture.open(arena.allocator());
    const pw = try packedW(&f);
    defer t.allocator.free(pw);
    var b = try bridge(&f, pw, true);
    const zero = try t.allocator.alloc(u16, H * R);
    defer t.allocator.free(zero);
    @memset(zero, 0);
    b.write = zero;
    var out: [T * H]u16 = undefined;
    var normed: [T * H]u16 = undefined;
    try host.write(t.allocator, dims, b, try f.bf16("x1"), (try f.words("anchor"))[0], .{ .weight = try f.bf16("norm1"), .eps = 1e-6 }, &out, &normed, null);
    for ((try f.bf16("x1"))[H..], out[H..]) |a, o| try t.expectEqual(a, o);
    var moved: usize = 0;
    for ((try f.bf16("out1"))[H..], out[H..]) |a, o| moved += @intFromBool(ulps(a, o) > 1);
    try t.expect(moved * 2 > out.len - H);
}
