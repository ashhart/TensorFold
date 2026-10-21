//! The EXL3 pack's tensors as the Flash Next loaders read them (cuda_prompt_wt.zig `Store`):
//!  - the MTP head's mixer lives in `mtp_hyper_connection_mixer_patch.safetensors`, outside the index: its names join
//!    the store's weight map at open;
//!  - its dense weights (router, shared gate, every hyper-connection tensor, in_proj_a/b, the PLE projections, conv
//!    and norm, the MTP pre-norm) are stored F16 where the loaders read BF16. Each is widened to BF16 when every value
//!    survives the round trip exactly (an F16 that came from a BF16 checkpoint always does: 8 significant bits, BF16's
//!    exponent range). A tensor with any inexact value is refused by name (error.F16NotBf16Exact), never rounded.
//! EXL3's own parts (`.trellis`, `.suh`, `.svh`, the codebook marker) keep their stored dtype: only `.weight` widens.

const std = @import("std");
const core = @import("core");

/// The mixer file the index does not name (the MTP head's `mtp.hyper_connection_mixer.*`).
pub const patch_file = "mtp_hyper_connection_mixer_patch.safetensors";

/// Adds every tensor of `dir`'s mixer patch to the index's weight map (the file absent: nothing). Names already in the
/// index keep their shard. The strings live in the parsed index's arena.
pub fn addPatch(io: std.Io, dir: []const u8, parsed: *std.json.Parsed(std.json.Value)) !void {
    const a = parsed.arena.allocator();
    const wm = (parsed.value.object.getPtr("weight_map") orelse return);
    if (wm.* != .object) return;
    const path = try std.fs.path.join(a, &.{ dir, patch_file });
    var file = core.safetensors.File.open(a, io, path) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer file.close(io);
    var it = file.names.iterator();
    while (it.next()) |e| {
        if (wm.object.contains(e.key_ptr.*)) continue;
        try wm.object.put(a, try a.dupe(u8, e.key_ptr.*), .{ .string = patch_file });
    }
}

/// Whether the loaders read `name` widened: a dense F16 `.weight` (an EXL3 part never ends in `.weight`).
pub fn widens(name: []const u8, t: core.safetensors.Tensor) bool {
    return t.dtype == .f16 and std.mem.endsWith(u8, name, ".weight");
}

/// RNE to BF16 of an F32 (cuda_prompt_wt.toBf16's rounding, without its NaN quieting: a NaN is inexact here).
fn bf16Of(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// The F16 words as BF16, exactly, or error.F16NotBf16Exact at the first value BF16 cannot hold (NaN included).
pub fn exactBf16(out: []u16, f16_bytes: []const u8) !void {
    if (f16_bytes.len != out.len * 2) return error.UnexpectedTensor;
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, f16_bytes[2 * i ..][0..2], .little));
        const v: f32 = h;
        const b = bf16Of(v);
        const back: f32 = @bitCast(@as(u32, b) << 16);
        if (back != v or std.math.isNan(v)) return error.F16NotBf16Exact;
        o.* = b;
    }
}

/// A widened copy of `t` as a BF16 tensor of the same shape, in `owned` (the caller frees it).
pub fn widen(gpa: std.mem.Allocator, name: []const u8, t: core.safetensors.Tensor, owned: *std.ArrayList([]u16)) !core.safetensors.Tensor {
    const count = t.bytes.len / 2;
    const words = try gpa.alloc(u16, count);
    errdefer gpa.free(words);
    exactBf16(words, t.bytes) catch |err| {
        if (err == error.F16NotBf16Exact) std.log.err("flashnext: {s} is F16 with a value BF16 cannot hold exactly", .{name});
        return err;
    };
    try owned.append(gpa, words);
    var out = t;
    out.dtype = .bf16;
    out.bytes = std.mem.sliceAsBytes(words);
    return out;
}

test "exl3: an F16 that came from BF16 widens bit for bit and one extra mantissa bit is refused" {
    const values = [_]f32{ 0, -0.0, 1, -2.5, 0.0078125, 65280, -3.0517578125e-05, 5.9604644775390625e-08, std.math.inf(f32) };
    var bytes: [values.len * 2]u8 = undefined;
    for (values, 0..) |v, i| {
        const b: f32 = @bitCast(@as(u32, bf16Of(v)) << 16);
        try std.testing.expectEqual(v, b); // every case is a BF16 value
        std.mem.writeInt(u16, bytes[2 * i ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
    }
    var out: [values.len]u16 = undefined;
    try exactBf16(&out, &bytes);
    for (values, out) |v, o| try std.testing.expectEqual(v, @as(f32, @bitCast(@as(u32, o) << 16)));
    // 1 + 2^-10 needs 11 significant bits: F16 holds it, BF16 does not
    std.mem.writeInt(u16, bytes[0..2], @bitCast(@as(f16, 1.0009765625)), .little);
    try std.testing.expectError(error.F16NotBf16Exact, exactBf16(&out, &bytes));
    std.mem.writeInt(u16, bytes[0..2], @bitCast(std.math.nan(f16)), .little);
    try std.testing.expectError(error.F16NotBf16Exact, exactBf16(&out, &bytes));
}

test "exl3: only dense F16 weights widen, EXL3 parts keep their dtype" {
    const t: core.safetensors.Tensor = .{ .dtype = .f16, .rank = 1, .shape = .{ 4, 1, 1, 1, 1 }, .bytes = &.{} };
    try std.testing.expect(widens("model.language_model.layers.0.mlp.gate.weight", t));
    try std.testing.expect(widens("mtp.hyper_connection_mixer.hc_norm.weight", t));
    try std.testing.expect(!widens("model.language_model.layers.0.self_attn.q_proj.suh", t));
    try std.testing.expect(!widens("lm_head.svh", t));
    var b = t;
    b.dtype = .bf16;
    try std.testing.expect(!widens("model.language_model.layers.0.linear_attn.norm.weight", b));
}

test "exl3: the mixer patch's names join the weight map, an indexed name keeps its shard" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"mtp.hyper_connection_mixer.hc_norm.weight\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]},\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[2,3]}}";
    var data: [8 + header.len + 3]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    @memset(data[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = patch_file, .data = &data });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"weight_map\":{\"x\":\"model-00001-of-00007.safetensors\"}}", .{});
    defer parsed.deinit();
    try addPatch(io, dir, &parsed);
    const wm = parsed.value.object.get("weight_map").?.object;
    try std.testing.expectEqualStrings(patch_file, wm.get("mtp.hyper_connection_mixer.hc_norm.weight").?.string);
    try std.testing.expectEqualStrings("model-00001-of-00007.safetensors", wm.get("x").?.string);
    // a folder without the patch file opens as before
    var other = std.testing.tmpDir(.{});
    defer other.cleanup();
    const empty = try other.dir.realPathFileAlloc(io, ".", a);
    defer a.free(empty);
    try addPatch(io, empty, &parsed);
}
