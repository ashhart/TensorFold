//! GLM-5.3-Flash's EXL3 routed experts on CUDA (Mia-AiLab's checkpoint: 4-bit mcg trellis experts, the rest bf16).

const std = @import("std");
const modelopt = @import("core").modelopt;
const Span = @import("cuda_shard.zig").Span;
const cuda = @import("cuda");
const experts = @import("cuda_experts.zig");

/// config.json's quantization_config (top level, else text_config's), when it has one.
fn block(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q.object;
    if (root.object.get("text_config")) |t| if (t == .object) if (t.object.get("quantization_config")) |q| if (q == .object) return q.object;
    return null;
}

/// Whether config.json names EXL3 (quant_method "exl3"): `admit` then says whether this family reads it.
pub fn isExl3(gpa: std.mem.Allocator, config_json: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch return false;
    defer parsed.deinit();
    const q = block(parsed.value) orelse return false;
    const m = q.get("quant_method") orelse return false;
    return m == .string and std.mem.eql(u8, m.string, "exl3");
}

/// The EXL3 checkpoints the kernels read: 4-bit routed experts on the "mcg" codebook (everything else bf16).
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *modelopt.Why) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const q = block(parsed.value) orelse {
        why.set("no quantization_config", .{});
        return error.UnsupportedQuant;
    };
    const bits = q.get("bits") orelse std.json.Value{ .null = {} };
    if (bits != .integer or bits.integer != 4) {
        why.set("EXL3 at {f} bits: the CUDA family reads 4-bit EXL3 experts", .{std.json.fmt(bits, .{})});
        return error.UnsupportedQuant;
    }
    const book = q.get("codebook") orelse std.json.Value{ .null = {} };
    if (book != .string or !std.mem.eql(u8, book.string, "mcg")) {
        why.set("EXL3 codebook {f}: the CUDA family reads \"mcg\"", .{std.json.fmt(book, .{})});
        return error.UnsupportedQuant;
    }
}

/// One layer's EXL3 experts on a rank: trellis words [E][K/16][N/16][32] and fp16 scales (suh [E][K], svh [E][N]).
pub const Experts = struct {
    gt: u64,
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    k2: u64, // independent per-expert projection widths in the common module
    shx: bool, // every expert's gate and up share one input rotation: a prompt rotates a row once, not once a pick
};

const tile = 128; // a 16 x 16 tile's bytes at 4 bits

/// Layer `i`'s EXL3 experts through `L` (cuda_weights' loader): gate, up by `cut`'s tile columns, down by its rows.
pub fn load(L: anytype, i: u32, count: usize, dims: usize, inter: usize, cut: Span) !Experts {
    if (dims % 128 != 0 or cut.lo % 128 != 0 or cut.hi % 128 != 0) return error.NotExl3Mcg4;
    const width = cut.len();
    var out: Experts = undefined;
    out.shx = true;
    const Mat = struct { k: usize, n: usize, by_cols: bool };
    const mats = [_]Mat{
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = inter, .n = dims, .by_cols = false },
    };
    var first_suh: ?[]const u8 = null;
    inline for (mats, .{ "gate_proj", "up_proj", "down_proj" }, .{ "gt", "ut", "dt" }, .{ "suh_g", "suh_u", "suh_d" }, .{ "svh_g", "svh_u", "svh_d" }) |m, proj, tw, su, sv| {
        const kt = (if (m.by_cols) m.k else width) / 16; // the rank's tiles
        const nt = (if (m.by_cols) width else m.n) / 16;
        for (0..count) |e| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ i, e });
            if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != m.k / 16 or t.dim(1) != m.n / 16 or t.dim(2) != 64) return error.NotExl3Mcg4;
            const marker = L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".mcg", .{ i, e }) catch return error.NotExl3Mcg4;
            if (marker.dtype != .i32 or marker.numel() != 1 or marker.bytes.len != 4 or
                std.mem.readInt(u32, marker.bytes[0..4], .little) != @import("core").exl3_load.markers[0].value)
                return error.NotExl3Mcg4;
            if (m.by_cols) { // every tile row, the rank's tile columns
                const row = (m.n / 16) * tile;
                for (0..kt) |r| try L.host.appendSlice(L.gpa, t.bytes[r * row + (cut.lo / 16) * tile ..][0 .. nt * tile]);
            } else try L.host.appendSlice(L.gpa, t.bytes[(cut.lo / 16) * nt * tile ..][0 .. kt * nt * tile]); // its tile rows
        }
        const base = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
        for (0..count) |e| {
            const ptr = base + e * kt * nt * tile;
            try L.host.appendSlice(L.gpa, std.mem.asBytes(&ptr));
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw ++ "_ptrs", .{i});
        for (0..count) |e| { // suh: the input's scales (gate, up: every input; down: the rank's)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.k) return error.NotExl3Mcg4;
            const mine = if (m.by_cols) t.bytes else t.bytes[cut.lo * 2 .. cut.hi * 2];
            if (m.by_cols) { // one rotation shared by the layer's every gate and up input?
                if (first_suh) |f| (if (!std.mem.eql(u8, f, mine)) {
                    out.shx = false;
                }) else first_suh = mine;
            }
            try L.host.appendSlice(L.gpa, mine);
        }
        @field(out, su) = try L.put("layers.{d}.moe.exl3." ++ su, .{i});
        for (0..count) |e| { // svh: the output's scales (gate, up: the rank's; down: every output)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.n) return error.NotExl3Mcg4;
            try L.host.appendSlice(L.gpa, if (m.by_cols) t.bytes[cut.lo * 2 .. cut.hi * 2] else t.bytes);
        }
        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
    }
    for (0..count) |_| {
        const bits: i32 = 8;
        try L.host.appendSlice(L.gpa, std.mem.asBytes(&bits));
    }
    out.k2 = try L.put("layers.{d}.moe.exl3.k2", .{i});
    return out;
}

/// The selected common EXL3 owner. This file adapts the family's checkpoint shard and slot layout only.
pub const Kernels = cuda.exl3_experts.Experts;
const common = cuda.exl3_experts;
pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };
pub const top_k = 8;
/// Rows a call to the common experts takes: each call reads every expert's weights, so a prompt pass needs few calls.
pub const chunk_rows = 1024;
fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
    return .{ .gate = w.gt, .up = w.ut, .down = w.dt, .gate_k2 = w.k2, .up_k2 = w.k2, .down_k2 = w.k2, .suh_g = w.suh_g, .suh_u = w.suh_u, .svh_g = w.svh_g, .svh_u = w.svh_u, .suh_d = w.suh_d, .svh_d = w.svh_d, .experts = count, .dims = dims, .width = width, .codebook = .mcg, .limit = limit, .act_mode = 0, .uniform_k2 = 8, .shared_input = w.shx };
}
fn shape(dims: usize, width: usize) common.Layer {
    const w: Experts = .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .shx = false };
    return layer(w, cuda.grouped.max_experts, dims, width, 0);
}
/// A chunk's compacted picks ahead of the common scratch, 256-aligned; the usize keeps @min from narrowing to u7.
fn picksBytes(rows: usize) usize {
    const cap: usize = @min(rows, chunk_rows);
    return std.mem.alignForward(usize, cap * top_k * 4, 256);
}
pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool) [4]usize {
    const n: usize = @min(rows, chunk_rows);
    var l = shape(dims, width);
    l.shared_input = shx;
    if (prefill and common.prompt.eligible(l)) return .{ picksBytes(rows) + (common.prompt.bytes(l, n, top_k) catch unreachable), 4, 4, 4 };
    return .{ picksBytes(rows) + (common.scratchBytes(shape(dims, width), n, top_k) catch unreachable), 4, 4, 4 };
}
/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64) !void {
    _ = xd;
    if (slots != top_k + 1) return error.UnsupportedTopK;
    if (dims % 256 != 0 or width % 128 != 0) return error.NotExl3Mcg4;
    const l = layer(w, count, dims, width, limit);
    const cap: usize = @min(rows, chunk_rows);
    const arena = sc.picks + picksBytes(rows);
    const fused = prompt and common.prompt.eligible(l);
    var at: usize = 0;
    while (at < rows) : (at += chunk_rows) {
        const n: usize = @min(chunk_rows, rows - at);
        try copyRows(stream, sc.picks, top_k * 4, picks + at * slots * 4, slots * 4, top_k * 4, n, "EXL3 compact route rows");
        const output = if (fused) blk: {
            const scratch = try common.prompt.scratch(arena, l, cap, top_k);
            try k.prompt.run(k, ek.router, stream, l, x + at * x_stride * 2, x_stride, sc.picks, n, scratch, plan_buf);
            break :blk scratch.y;
        } else blk: {
            const scratch = try common.scratch(arena, l, cap, top_k);
            try k.runSlots(stream, l, x + at * x_stride * 2, x_stride, .bf16, sc.picks, n, scratch);
            break :blk scratch.y;
        };
        try copyRows(stream, ey + at * slots * dims * 4, slots * dims * 4, output, top_k * dims * 4, top_k * dims * 4, n, "EXL3 routed output slots");
    }
}

test "EXL3 admission: Mia-AiLab's block passes; other widths and codebooks are refused by name" {
    const gpa = std.testing.allocator;
    const mia =
        \\{"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}}
    ;
    try std.testing.expect(isExl3(gpa, mia));
    var why: modelopt.Why = .{};
    try admit(gpa, mia, &why);
    try std.testing.expect(!isExl3(gpa, "{\"quantization_config\": {\"quant_method\": \"modelopt\"}}"));
    try std.testing.expect(!isExl3(gpa, "{}"));
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 3, \"codebook\": \"mcg\"}}", &why));
    try std.testing.expectEqualStrings("EXL3 at 3 bits: the CUDA family reads 4-bit EXL3 experts", why.text());
    try std.testing.expectError(error.UnsupportedQuant, admit(gpa, "{\"text_config\": {\"quantization_config\": {\"quant_method\": \"exl3\", \"bits\": 4, \"codebook\": \"mul1\"}}}", &why));
    try std.testing.expectEqualStrings("EXL3 codebook \"mul1\": the CUDA family reads \"mcg\"", why.text());
}

test "a chunk's compacted picks end before the common scratch at every row count" {
    for (1..3 * chunk_rows) |rows| {
        const cap: usize = @min(rows, chunk_rows);
        try std.testing.expect(picksBytes(rows) >= cap * top_k * 4);
        try std.testing.expectEqual(@as(usize, 0), picksBytes(rows) % 256);
    }
}

test "GLM's routed experts run on the common EXL3 module's kernels" {
    try std.testing.expect(Kernels == cuda.exl3_experts.Experts);
}
