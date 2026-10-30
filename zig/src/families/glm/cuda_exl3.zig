//! GLM's EXL3 routed experts on CUDA: "mcg" trellis experts at any whole-bit width, mixed per projection; rest bf16.

const std = @import("std");
const core = @import("core");
const modelopt = core.modelopt;
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

/// The half-bit widths (2 * bits) the grouped kernels decode on mcg: whole bits from 1 to 8 (odd halves are mul1's).
pub const min_k2 = 2;
pub const max_k2 = 16;

/// A trellis's half-bit width from its last dimension (16 * bits int16 a tile), null if the kernels skip it.
pub fn halves(last: usize) ?u32 {
    if (last % 8 != 0) return null;
    const k2 = last / 8;
    if (k2 < min_k2 or k2 > max_k2 or k2 % 2 != 0) return null;
    return @intCast(k2);
}

/// A JSON number as f64, or null.
fn number(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

/// The EXL3 the kernels read: "mcg" routed experts at 1 to 8 bits, each trellis's width taken from its shape.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *modelopt.Why) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const q = block(parsed.value) orelse {
        why.set("no quantization_config", .{});
        return error.UnsupportedQuant;
    };
    const lo: f64 = min_k2 / 2;
    const hi: f64 = max_k2 / 2;
    if (q.get("bits")) |bits| {
        const b = number(bits);
        if (b == null or b.? != @floor(b.?) or !(b.? >= lo and b.? <= hi)) {
            why.set("EXL3 at {f} bits: the CUDA family reads mcg experts of 1 to 8 whole bits", .{std.json.fmt(bits, .{})});
            return error.UnsupportedQuant;
        }
    }
    if (q.get("avg_bits")) |avg| {
        const b = number(avg);
        if (b == null or !(b.? >= lo and b.? <= hi)) {
            why.set("EXL3 averaging {f} bits: the CUDA family reads mcg experts of 1 to 8 bits", .{std.json.fmt(avg, .{})});
            return error.UnsupportedQuant;
        }
    }
    const book = q.get("codebook") orelse std.json.Value{ .null = {} };
    if (book != .string or !std.mem.eql(u8, book.string, "mcg")) {
        why.set("EXL3 codebook {f}: the CUDA family reads \"mcg\"", .{std.json.fmt(book, .{})});
        return error.UnsupportedQuant;
    }
}

/// One layer's EXL3 experts on a rank: trellis words [E][K/16][N/16][8 * k2] and fp16 scales (suh [E][K], svh [E][N]).
pub const Experts = struct {
    gt: u64, // int64 [E]: each expert's gate trellis
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    k2: u64, // int32 [E]: every expert's gate half-bit width (2 * bits); up's and down's too unless theirs are set
    k2_up: u64 = 0, // int32 [E]: up's widths (0: k2's)
    k2_down: u64 = 0, // int32 [E]: down's widths (0: k2's)
    // Widths the loader found across gate/up and down (0: they differ); a literal is upstream's 4-bit, 8 half-bits.
    uniform_gu: u32 = 8,
    uniform_d: u32 = 8,
    shx: bool, // every expert's gate and up share one input rotation: a prompt rotates a row once, not once a pick

    /// The gate, up and down width tables.
    pub fn tables(w: Experts) [3]u64 {
        return .{ w.k2, if (w.k2_up != 0) w.k2_up else w.k2, if (w.k2_down != 0) w.k2_down else w.k2 };
    }
};

/// A 16 x 16 tile's bytes at `k2` half-bits a value (128 at 4 bits, 96 at 3).
pub fn tileBytes(k2: u32) usize {
    return 16 * @as(usize, k2);
}

/// The width every entry of `k2s` has, or 0 when they differ (or there are none).
fn uniform(k2s: []const i32) u32 {
    if (k2s.len == 0) return 0;
    for (k2s) |k| if (k != k2s[0]) return 0;
    return @intCast(k2s[0]);
}

/// Why trellis `t` of a K x N projection is not one the kernels read, or null when it is.
pub fn refusal(t: core.checkpoint.Tensor, k: usize, n: usize) ?[]const u8 {
    if (t.dtype != .i16 or t.rank != 3) return "not an int16 trellis [K/16, N/16, 16 * bits]";
    if (t.dim(0) != k / 16 or t.dim(1) != n / 16) return "its tiles do not cover the projection's K x N";
    if (halves(t.dim(2)) == null) return "a width other than 1 to 8 whole bits (16 to 128 int16 a tile)";
    if (t.bytes.len != t.numel() * 2) return "its bytes do not match its shape";
    return null;
}

/// Layer `i`'s experts via `L`: gate and up by `cut`'s tile columns, down by rows, each at its own trellis width.
pub fn load(L: anytype, i: u32, count: usize, dims: usize, inter: usize, cut: Span) !Experts {
    if (dims % 128 != 0 or cut.lo % 128 != 0 or cut.hi % 128 != 0) return error.UnreadableExl3;
    const width = cut.len();
    var out: Experts = undefined;
    out.shx = true;
    const k2s = try L.gpa.alloc(i32, 3 * count); // gate's, up's, down's by expert
    defer L.gpa.free(k2s);
    const Mat = struct { k: usize, n: usize, by_cols: bool };
    const mats = [_]Mat{
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = dims, .n = inter, .by_cols = true },
        .{ .k = inter, .n = dims, .by_cols = false },
    };
    var first_suh: ?[]const u8 = null;
    inline for (mats, .{ "gate_proj", "up_proj", "down_proj" }, .{ "gt", "ut", "dt" }, .{ "suh_g", "suh_u", "suh_d" }, .{ "svh_g", "svh_u", "svh_d" }, 0..) |m, proj, tw, su, sv, pi| {
        const kt = (if (m.by_cols) m.k else width) / 16; // the rank's tiles
        const nt = (if (m.by_cols) width else m.n) / 16;
        const mine = k2s[pi * count ..][0..count];
        for (0..count, mine) |e, *k2| {
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ i, e });
            if (refusal(t, m.k, m.n)) |why| {
                std.log.err("{s}: {s} (shape {any}); GLM's CUDA experts read mcg trellises of 1 to 8 bits", .{ L.last, why, t.shape[0..t.rank] });
                return error.UnreadableExl3;
            }
            const h = halves(t.dim(2)).?;
            const tile = tileBytes(h);
            const marker = L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".mcg", .{ i, e }) catch return error.UnreadableExl3;
            if (marker.dtype != .i32 or marker.numel() != 1 or marker.bytes.len != 4 or
                std.mem.readInt(u32, marker.bytes[0..4], .little) != core.exl3_load.markers[0].value)
                return error.UnreadableExl3;
            if (m.by_cols) { // every tile row, the rank's tile columns
                const row = (m.n / 16) * tile;
                for (0..kt) |r| try L.host.appendSlice(L.gpa, t.bytes[r * row + (cut.lo / 16) * tile ..][0 .. nt * tile]);
            } else try L.host.appendSlice(L.gpa, t.bytes[(cut.lo / 16) * nt * tile ..][0 .. kt * nt * tile]); // its tile rows
            k2.* = @intCast(h);
        }
        const base = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
        var ptr: u64 = base;
        for (mine) |k2| { // each expert's trellis follows the one before, at its own width
            try L.host.appendSlice(L.gpa, std.mem.asBytes(&ptr));
            ptr += kt * nt * tileBytes(@intCast(k2));
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw ++ "_ptrs", .{i});
        for (0..count) |e| { // suh: the input's scales (gate, up: every input; down: the rank's)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.k) return error.UnreadableExl3;
            const own = if (m.by_cols) t.bytes else t.bytes[cut.lo * 2 .. cut.hi * 2];
            if (m.by_cols) { // one rotation shared by the layer's every gate and up input?
                if (first_suh) |f| (if (!std.mem.eql(u8, f, own)) {
                    out.shx = false;
                }) else first_suh = own;
            }
            try L.host.appendSlice(L.gpa, own);
        }
        @field(out, su) = try L.put("layers.{d}.moe.exl3." ++ su, .{i});
        for (0..count) |e| { // svh: the output's scales (gate, up: the rank's; down: every output)
            const t = try L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ i, e });
            if (t.dtype != .f16 or t.numel() != m.n) return error.UnreadableExl3;
            try L.host.appendSlice(L.gpa, if (m.by_cols) t.bytes[cut.lo * 2 .. cut.hi * 2] else t.bytes);
        }
        @field(out, sv) = try L.put("layers.{d}.moe.exl3." ++ sv, .{i});
    }
    out.uniform_gu = uniform(k2s[0 .. 2 * count]);
    out.uniform_d = uniform(k2s[2 * count ..]);
    const tabs = [3][]const i32{ k2s[0..count], k2s[count .. 2 * count], k2s[2 * count ..] };
    if (std.mem.eql(i32, tabs[0], tabs[1]) and std.mem.eql(i32, tabs[0], tabs[2])) { // one table, as a 4-bit layer has it
        try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(tabs[0]));
        out.k2 = try L.put("layers.{d}.moe.exl3.k2", .{i});
        out.k2_up = out.k2;
        out.k2_down = out.k2;
        return out;
    }
    var ptrs: [3]u64 = undefined;
    inline for (tabs, .{ "k2_gate", "k2_up", "k2_down" }, 0..) |tab, name, p| { // a table a distinct projection
        var same: ?u64 = null;
        for (tabs[0..p], ptrs[0..p]) |before, ptr| if (same == null and std.mem.eql(i32, before, tab)) {
            same = ptr;
        };
        ptrs[p] = same orelse blk: {
            try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(tab));
            break :blk try L.put("layers.{d}.moe.exl3." ++ name, .{i});
        };
    }
    out.k2, out.k2_up, out.k2_down = ptrs;
    return out;
}

/// The selected common EXL3 owner. This file adapts the family's checkpoint shard and slot layout only.
pub const Kernels = cuda.exl3_experts.Experts;
const common = cuda.exl3_experts;
pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };
pub const top_k = 8;
/// Rows a call to the common experts takes: each call reads every expert's weights, so a prompt pass needs few calls.
pub const chunk_rows = 1024;
/// The common module's layer: a width table a projection, and the widths the loader found uniform.
pub fn layer(w: Experts, count: usize, dims: usize, width: usize, limit: f32) common.Layer {
    return .{
        .gate = w.gt,
        .up = w.ut,
        .down = w.dt,
        .gate_k2 = w.tables()[0],
        .up_k2 = w.tables()[1],
        .down_k2 = w.tables()[2],
        .suh_g = w.suh_g,
        .suh_u = w.suh_u,
        .svh_g = w.svh_g,
        .svh_u = w.svh_u,
        .suh_d = w.suh_d,
        .svh_d = w.svh_d,
        .experts = count,
        .dims = dims,
        .width = width,
        .codebook = .mcg,
        .limit = limit,
        .act_mode = 0,
        .uniform_k2 = if (w.uniform_gu == w.uniform_d) w.uniform_gu else 0,
        .uniform_k2_gu = w.uniform_gu,
        .uniform_k2_d = w.uniform_d,
        .shared_input = w.shx,
    };
}
/// A chunk's compacted picks ahead of the common scratch, 256-aligned; the usize keeps @min from narrowing to u7.
pub fn picksBytes(rows: usize) usize {
    const cap: usize = @min(rows, chunk_rows);
    return std.mem.alignForward(usize, cap * top_k * 4, 256);
}
/// The scratch `run` takes: the prompt kernels' when `prefill` and they take the layer, else the grouped path's.
pub fn scratchFor(rows: usize, dims: usize, width: usize, prefill: bool, w: Experts) [4]usize {
    const n: usize = @min(rows, chunk_rows);
    const l = layer(w, cuda.grouped.max_experts, dims, width, 0);
    if (prefill and common.prompt.eligible(l)) return .{ picksBytes(rows) + (common.prompt.bytes(l, n, top_k) catch unreachable), 4, 4, 4 };
    return .{ picksBytes(rows) + (common.scratchBytes(l, n, top_k) catch unreachable), 4, 4, 4 };
}
/// `scratchFor` a 4-bit layer (every projection 8 half-bits), `shx` its shared rotation.
pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool) [4]usize {
    return scratchFor(rows, dims, width, prefill, .{ .gt = 0, .ut = 0, .dt = 0, .k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .uniform_gu = 8, .uniform_d = 8, .shx = shx });
}
/// `most` raised to `w`'s scratch, entry by entry: buffers every layer's experts run in take the largest.
pub fn mostScratch(most: *[4]usize, rows: usize, dims: usize, width: usize, prefill: bool, w: Experts) void {
    for (most, scratchFor(rows, dims, width, prefill, w)) |*m, v| m.* = @max(m.*, v);
}
/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst`, in one call.
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64) !void {
    _ = xd;
    if (slots != top_k + 1) return error.UnsupportedTopK;
    if (dims % 256 != 0 or width % 128 != 0) return error.UnreadableExl3;
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

test {
    _ = @import("cuda_exl3_test.zig");
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
