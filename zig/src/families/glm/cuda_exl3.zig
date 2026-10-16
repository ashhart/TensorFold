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
            _ = L.get("layers.{d}.mlp.experts.{d}." ++ proj ++ ".mcg", .{ i, e }) catch return error.NotExl3Mcg4;
            if (m.by_cols) { // every tile row, the rank's tile columns
                const row = (m.n / 16) * tile;
                for (0..kt) |r| try L.host.appendSlice(L.gpa, t.bytes[r * row + (cut.lo / 16) * tile ..][0 .. nt * tile]);
            } else try L.host.appendSlice(L.gpa, t.bytes[(cut.lo / 16) * nt * tile ..][0 .. kt * nt * tile]); // its tile rows
        }
        @field(out, tw) = try L.put("layers.{d}.moe.exl3." ++ tw, .{i});
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
    return out;
}

/// glm_exl3's kernels.
pub const Kernels = struct {
    rot_rows: cuda.Function,
    rot_in: cuda.Function,
    grouped: cuda.Function,
    gateup: cuda.Function,
    down: cuda.Function,
    prompt_gu: cuda.Function,
    prompt_dn: cuda.Function,
    picks: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name) = try m.function("glm_exl3_" ++ name);
        return k;
    }
};

/// Scratch beside the MoE block's (its eact holds the down inputs): 8-slot picks, rotated inputs, K-split partials.
pub const Scratch = struct { picks: u64, xg: u64, xu: u64, z: u64 };

/// The picks a row holds (the kernels' pair = row * 8 + slot).
pub const top_k = 8;
/// Gate and up K slices in a decode window (down: 1); prompt chunks take the fused kernels, no partials.
pub const decode_splits = 4;

/// Bytes of each Scratch buffer for windows of `rows` rows: (picks, xg, xu, z); `shx`: one input rotation a layer.
pub fn scratchBytes(rows: usize, dims: usize, width: usize, prefill: bool, shx: bool) [4]usize {
    const pairs = rows * top_k;
    const one = prefill and shx; // a prompt rotates each row once
    return .{
        pairs * 4,
        (if (one) rows else pairs) * dims * 2,
        if (one) 16 else pairs * dims * 2,
        if (prefill) 16 else @max(2 * decode_splits * pairs * width, pairs * dims) * 4,
    };
}

/// One launch's arguments, then the launch.
fn go(f: cuda.Function, stream: cuda.Stream, grid: cuda.Dim3, threads: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, stream, &a);
}

/// Routed slots of `rows` rows (picks [rows][slots], shared last) into `ey` [rows][slots][dims] fp32, shared untouched.
pub fn run(k: *const Kernels, ek: *const experts.Kernels, stream: cuda.Stream, w: Experts, sc: Scratch, plan_buf: cuda.grouped.Plan, x: u64, x_stride: usize, picks: u64, slots: usize, rows: usize, count: usize, dims: usize, width: usize, limit: f32, prompt: bool, ey: u64, xd: u64) !void {
    if (slots != top_k + 1) return error.UnsupportedTopK;
    if (dims % 256 != 0 or width % 128 != 0) return error.NotExl3Mcg4;
    const P = rows * top_k;
    const D: i32 = @intCast(dims);
    const N: i32 = @intCast(width);
    const ys: i32 = @intCast(slots);
    try go(k.picks, stream, .{ .x = @intCast((P + 255) / 256) }, 256, .{ picks, sc.picks, @as(i32, @intCast(rows)), @as(i32, @intCast(slots)) });
    const plan: experts.Plan = .{ .route = plan_buf, .slots = top_k, .experts = count, .prompt = prompt };
    try experts.route(ek, stream, sc.picks, P, plan);
    const items: u32 = @intCast(experts.maxItems(P, count, if (prompt) experts.staged_tile else experts.tile));
    const r = plan_buf;
    const one = prompt and w.shx; // the prompt kernels read row pair >> 3: one rotated row a token
    if (one) {
        try go(k.rot_rows, stream, .{ .x = @intCast(rows), .y = @intCast(dims / 128) }, 32, .{ x, @as(i32, @intCast(x_stride)), w.suh_g, sc.xg, D });
    } else {
        try go(k.rot_in, stream, .{ .x = @intCast(P), .y = @intCast(dims / 128), .z = 2 }, 32, .{ x, @as(i32, @intCast(x_stride)), sc.picks, w.suh_g, w.suh_u, sc.xg, sc.xu, D });
    }
    if (prompt) { // gate and up with their epilogue into xd; down with its epilogue into ey
        try go(k.prompt_gu, stream, .{ .x = @intCast(width / 128), .y = items }, 256, .{ sc.xg, if (one) sc.xg else sc.xu, w.gt, w.ut, r.items, r.counts, r.members, w.svh_g, w.svh_u, w.suh_d, xd, D, N, @as(i32, if (one) 3 else 0), limit });
        return go(k.prompt_dn, stream, .{ .x = @intCast(dims / 256), .y = items }, 256, .{ xd, w.dt, r.items, r.counts, r.members, w.svh_d, ey, N, D, ys });
    }
    const sk: i32 = decode_splits;
    const nt = 8; // n tiles a block
    try go(k.grouped, stream, .{ .x = items, .y = @intCast(width / (16 * nt)), .z = @intCast(2 * sk) }, 128, .{ sc.xg, sc.xu, w.gt, w.ut, r.items, r.counts, r.members, sc.z, D, N, @as(i32, @intCast(P)), sk });
    try go(k.gateup, stream, .{ .x = @intCast(P), .y = @intCast(width / 128) }, 32, .{ sc.z, sc.picks, w.svh_g, w.svh_u, w.suh_d, xd, @as(i32, @intCast(P)), N, sk, limit });
    try go(k.grouped, stream, .{ .x = items, .y = @intCast(dims / (16 * nt)), .z = 1 }, 128, .{ xd, xd, w.dt, w.dt, r.items, r.counts, r.members, sc.z, N, D, @as(i32, @intCast(P)), @as(i32, 1) });
    try go(k.down, stream, .{ .x = @intCast(P), .y = @intCast(dims / 128) }, 32, .{ sc.z, sc.picks, w.svh_d, ey, @as(i32, @intCast(P)), D, @as(i32, 1), ys });
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
