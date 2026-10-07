//! Flash Next packs built in Zig from the checkpoint match the dump tools byte for byte, with no model load.
const std = @import("std");
const Io = std.Io;
const st = @import("../../core/safetensors.zig");
const ckpt = @import("../../core/checkpoint.zig");
const affine = @import("affine.zig");
const config_mod = @import("config.zig");
const index_mod = @import("index.zig");
const pio = @import("pack_io.zig");

pub const Report = struct {
    decode_tensors: usize = 0,
    mlx_tensors: usize = 0,
    mtp_mlx_tensors: usize = 0,
    norms_around_one: bool = false,
    draft_ids: usize = 0,
};

const Out = pio.Out;

/// Which of a quantized linear's three tensors.
const Part = enum { weight, scales, biases };

/// A checkpoint quantized linear's three tensors with its verified affine spec.
pub const Q = struct {
    w: st.Tensor,
    s: st.Tensor,
    b: st.Tensor,
    rows: usize,
    k: usize,
    kw: usize,
    spec: affine.Spec,

    fn part(r: Q, comptime which: Part) []const u8 {
        return switch (which) {
            .weight => r.w.bytes,
            .scales => r.s.bytes,
            .biases => r.b.bytes,
        };
    }
};

fn qlinear(ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, path: []const u8) !Q {
    const spec = (try cfg.quantization(path)) orelse return error.NotQuantized;
    var buf: [320]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{path}));
    const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{path}));
    const b = try ck.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{path}));
    if (w.dtype != .u32 or w.rank != 2) return error.UnexpectedTensor;
    if (s.dtype != .bf16 or s.rank != 2 or b.dtype != .bf16 or b.rank != 2) return error.UnexpectedTensor;
    if (spec.group == 128) return error.Group128NotPacked; // the pack never carries group 128; the halve-to-64 widen is unimplemented
    const rows = w.dim(0);
    const kw = w.dim(1);
    if (rows != s.dim(0) or rows != b.dim(0)) return error.UnexpectedTensor;
    if (kw * 32 % spec.bits != 0) return error.UnexpectedTensor;
    const k = kw * 32 / spec.bits;
    if (k % spec.group != 0 or s.dim(1) != k / spec.group or b.dim(1) != k / spec.group) return error.UnexpectedTensor;
    return .{ .w = w, .s = s, .b = b, .rows = rows, .k = k, .kw = kw, .spec = spec };
}

/// The lane pipeline's word permutation: rows tiled nt at a time, a group's words innermost (lane_qmm.tile_weight).
pub fn tileWeight(gpa: std.mem.Allocator, flat: []const u8, rows: usize, kw: usize, nt: usize, w: usize) ![]u8 {
    const words = std.mem.bytesAsSlice(u32, flat);
    const out = try gpa.alloc(u8, flat.len);
    errdefer gpa.free(out);
    const out_words = std.mem.bytesAsSlice(u32, out);
    const groups = kw / w;
    for (0..rows / nt) |t| {
        for (0..groups) |g| {
            for (0..nt) |j| {
                for (0..w) |c| {
                    out_words[((t * groups + g) * nt + j) * w + c] = words[(t * nt + j) * kw + g * w + c];
                }
            }
        }
    }
    return out;
}

/// (K/GS, N, 2) bf16 (scale, bias) pairs, group-major, over a row-concatenated stack (lane_qmm.pack_scales).
pub fn packScales(gpa: std.mem.Allocator, members: []const Q) ![]u8 {
    var rows: usize = 0;
    for (members) |m| rows += m.rows;
    const kg = members[0].s.dim(1);
    const out = try gpa.alloc(u8, kg * rows * 4);
    errdefer gpa.free(out);
    const out16 = std.mem.bytesAsSlice(u16, out);
    var at: usize = 0;
    for (members) |m| {
        const s16 = std.mem.bytesAsSlice(u16, m.part(.scales));
        const b16 = std.mem.bytesAsSlice(u16, m.part(.biases));
        for (0..kg) |g| {
            for (0..m.rows) |j| {
                out16[(g * rows + at + j) * 2] = s16[j * kg + g];
                out16[(g * rows + at + j) * 2 + 1] = b16[j * kg + g];
            }
        }
        at += m.rows;
    }
    return out;
}

/// Row-concatenated members in the checkpoint's own MLX layout: a pure byte move per member.
fn concatPart(gpa: std.mem.Allocator, members: []const Q, comptime part: Part) ![]u8 {
    var n: usize = 0;
    for (members) |m| n += m.part(part).len;
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    var at: usize = 0;
    for (members) |m| {
        @memcpy(out[at..][0..m.part(part).len], m.part(part));
        at += m.part(part).len;
    }
    return out;
}

/// One lane projection's two pack tensors: `.wq` tiled, `.sbt` group-major pairs (decode.py's lane pipeline).
fn putLane(out: *Out, gpa: std.mem.Allocator, name: []const u8, members: []const Q) !void {
    var rows: usize = 0;
    for (members) |m| rows += m.rows;
    const spec = members[0].spec;
    for (members) |m| if (m.spec.bits != spec.bits or m.spec.group != spec.group) return error.MixedAffine;
    const nt: usize = if (spec.bits == 4 and rows % 64 == 0) 64 else if (rows % 32 == 0) 32 else return error.UnsupportedLaneWidth;
    const w = spec.group * spec.bits / 32;
    const flat = try concatPart(gpa, members, .weight);
    defer gpa.free(flat);
    const wq = try tileWeight(gpa, flat, rows, members[0].kw, nt, w);
    const sbt = try packScales(gpa, members);
    const a = out.arena.allocator();
    try out.put(try std.fmt.allocPrint(a, "{s}.wq", .{name}), "U32", &.{ rows, members[0].kw }, wq);
    try out.put(try std.fmt.allocPrint(a, "{s}.sbt", .{name}), "BF16", &.{ members[0].s.dim(1), rows, 2 }, sbt);
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// f32 bits to bf16 bits, round-half-even: the cast MLX's bf16 store does. Infinities and NaNs keep their top half.
fn f32ToBf16(bits: u32) u16 {
    const rounded: u32 = bits +% 0x7FFF +% (bits >> 16 & 1);
    return @intCast(rounded >> 16);
}

/// A centered norm scale is the f32 round trip of decode's 1 + w, after the loader's -1 only when gamma is stored.
pub fn centeredScale(gpa: std.mem.Allocator, ck: *ckpt.Checkpoint, path: []const u8, around_one: bool) ![]u8 {
    const t = try ck.get(path);
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const in16 = std.mem.bytesAsSlice(u16, t.bytes);
    const out = try gpa.alloc(u8, in16.len * 4);
    errdefer gpa.free(out);
    const out32 = std.mem.bytesAsSlice(u32, out);
    for (in16, 0..) |bits, i| {
        var v: f32 = bf16ToF32(bits);
        if (around_one) v = v - 1.0; // the loader's stored form, which decode.py adds one back to
        out32[i] = @bitCast(@as(f32, 1.0) + v);
    }
    return out;
}

/// Gamma versus gamma - 1 follows the attn hc_norm means, and an ambiguous checkpoint is refused.
pub fn normsAroundOne(ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, wide: usize) !bool {
    if (cfg.layers < 8) return false; // model.py:353: fewer than 8 anchors decides nothing and stores around zero
    var means: std.ArrayList(f64) = .empty;
    defer means.deinit(ck.gpa);
    for (0..cfg.layers) |i| {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
        const t = try ck.get(name);
        if (t.dtype != .bf16 or t.dim(0) != wide) return error.UnexpectedTensor;
        const in16 = std.mem.bytesAsSlice(u16, t.bytes);
        var sum: f64 = 0;
        for (in16) |bits| sum += bf16ToF32(bits);
        try means.append(ck.gpa, sum / @as(f64, @floatFromInt(in16.len)));
    }
    var above: usize = 0;
    for (means.items) |m| {
        if (m > 0.5) above += 1;
    }
    std.mem.sort(f64, means.items, {}, std.sort.asc(f64));
    const median = means.items[means.items.len / 2];
    const share = @as(f64, @floatFromInt(above)) / @as(f64, @floatFromInt(means.items.len));
    const around_one = share >= 0.9 and 0.75 <= median and median <= 1.5;
    const around_zero = share <= 0.1 and -0.5 <= median and median <= 0.25;
    if (!around_one and !around_zero) return error.AmbiguousNormStorage;
    return around_one;
}

fn putHc(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, name: []const u8, stem: []const u8, inject: bool, around_one: bool, wide: usize) !void {
    var down_paths: [2][]const u8 = undefined;
    var down_n: usize = 1;
    down_paths[0] = try std.fmt.allocPrint(a, "{s}.input_mix_weight_down", .{stem});
    if (inject) {
        down_paths[1] = try std.fmt.allocPrint(a, "{s}.block_inject_weight", .{stem});
        down_n = 2;
    }
    var storage: [2]Q = undefined;
    for (down_paths[0..down_n], 0..) |p, i| storage[i] = try qlinear(ck, cfg, p);
    const down = storage[0..down_n];
    for (down) |m| if (m.spec.bits != down[0].spec.bits or m.spec.group != down[0].spec.group) return error.MixedAffine;
    var rows: usize = 0;
    for (down) |m| rows += m.rows;
    const scale = try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.hc_norm.weight", .{stem}), around_one);
    try out.put(try std.fmt.allocPrint(a, "{s}.scale", .{name}), "F32", &.{wide}, scale);
    const down_w = try concatPart(gpa, down, .weight);
    const down_s = try concatPart(gpa, down, .scales);
    const down_b = try concatPart(gpa, down, .biases);
    try out.put(try std.fmt.allocPrint(a, "{s}.down.w", .{name}), "U32", &.{ rows, down[0].kw }, down_w);
    try out.put(try std.fmt.allocPrint(a, "{s}.down.s", .{name}), "BF16", &.{ rows, down[0].s.dim(1) }, down_s);
    try out.put(try std.fmt.allocPrint(a, "{s}.down.b", .{name}), "BF16", &.{ rows, down[0].b.dim(1) }, down_b);
    const up = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.input_mix_weight_up", .{stem}));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.w", .{name}), "U32", &.{ up.rows, up.kw }, try gpa.dupe(u8, up.part(.weight)));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.s", .{name}), "BF16", &.{ up.rows, up.s.dim(1) }, try gpa.dupe(u8, up.part(.scales)));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.b", .{name}), "BF16", &.{ up.rows, up.b.dim(1) }, try gpa.dupe(u8, up.part(.biases)));
}

/// A gate's rows are stored bf16, or an affine linear dequantized on the host as code * scale + bias per group.
const Dense = struct { bytes: []const u8, rows: usize, cols: usize, owned: bool };

fn denseRows(gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, stem: []const u8) !Dense {
    const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
    if (w.dtype == .bf16 and w.rank == 2) return .{ .bytes = w.bytes, .rows = w.dim(0), .cols = w.dim(1), .owned = false };
    if (w.dtype != .u32 or w.rank != 2) return error.DenseRouterNotBf16;
    const spec = (try cfg.quantization(stem)) orelse return error.DenseRouterNotBf16;
    const s = try ck.get(try std.fmt.allocPrint(a, "{s}.scales", .{stem}));
    const b = try ck.get(try std.fmt.allocPrint(a, "{s}.biases", .{stem}));
    const we = st.Entry{ .dtype = w.dtype, .rank = w.rank, .shape = w.shape, .begin = 0, .end = w.bytes.len };
    const se = st.Entry{ .dtype = s.dtype, .rank = s.rank, .shape = s.shape, .begin = 0, .end = s.bytes.len };
    const be = st.Entry{ .dtype = b.dtype, .rank = b.rank, .shape = b.shape, .begin = 0, .end = b.bytes.len };
    const shape = try affine.matrix(we, se, be, spec);
    const es: usize = s.dtype.size();
    const out = try gpa.alloc(u8, shape.n * shape.k * 2);
    const words_per_row = shape.words * 4;
    for (0..shape.n) |r| {
        const row = w.bytes[r * words_per_row ..][0 .. words_per_row];
        for (0..shape.k) |c| {
            const g = c / spec.group;
            const scale: f32 = switch (s.dtype) {
                .bf16 => bf16ToF32(std.mem.readInt(u16, s.bytes[(r * shape.groups + g) * es ..][0..2], .little)),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, s.bytes[(r * shape.groups + g) * es ..][0..2], .little)))),
                .f32 => @bitCast(std.mem.readInt(u32, s.bytes[(r * shape.groups + g) * es ..][0..4], .little)),
                else => return error.InvalidAffineMetadataPrecision,
            };
            const bias: f32 = switch (b.dtype) {
                .bf16 => bf16ToF32(std.mem.readInt(u16, b.bytes[(r * shape.groups + g) * es ..][0..2], .little)),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b.bytes[(r * shape.groups + g) * es ..][0..2], .little)))),
                .f32 => @bitCast(std.mem.readInt(u32, b.bytes[(r * shape.groups + g) * es ..][0..4], .little)),
                else => return error.InvalidAffineMetadataPrecision,
            };
            const v: f32 = @as(f32, @floatFromInt(try affine.code(spec, row, c))) * scale + bias;
            const bits: u32 = @bitCast(v);
            const exp = bits & 0x7F800000;
            const rounded: u32 = if (exp == 0x7F800000) bits else bits + 0x7FFF + ((bits >> 16) & 1);
            std.mem.writeInt(u16, out[(r * shape.k + c) * 2 ..][0..2], @intCast(rounded >> 16), .little);
        }
    }
    return .{ .bytes = out, .rows = shape.n, .cols = shape.k, .owned = true };
}

/// A router's rows are the gate then the shared-expert gate, dequantized on the host when affine-stored.
fn putRouter(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, name: []const u8, gate_stem: []const u8, shared_stem: []const u8) !void {
    const gate = try denseRows(gpa, a, ck, cfg, gate_stem);
    const shared = try denseRows(gpa, a, ck, cfg, shared_stem);
    if (gate.cols != shared.cols) return error.UnexpectedTensor;
    const rows = gate.rows + shared.rows;
    const bytes = try gpa.alloc(u8, rows * gate.cols * 2);
    @memcpy(bytes[0..gate.bytes.len], gate.bytes);
    @memcpy(bytes[gate.bytes.len..], shared.bytes);
    if (gate.owned) gpa.free(gate.bytes);
    if (shared.owned) gpa.free(shared.bytes);
    try out.put(name, "BF16", &.{ rows, gate.cols }, bytes);
}

/// A depthwise conv weight is channel-major [C, W], transposed from [W, 1, C], and widened to f32 for the ple gate.
fn putConvSlice(out: *Out, gpa: std.mem.Allocator, ck: *ckpt.Checkpoint, name: []const u8, path: []const u8, widen_f32: bool) !void {
    const t = try ck.get(path);
    if (t.rank != 3 or (t.dtype != .bf16 and t.dtype != .f32)) return error.UnexpectedTensor;
    const es: usize = if (t.dtype == .bf16) 2 else 4;
    if (t.dim(2) == 1) {
        // [C, W, 1]: channel-major rows as the kernel binds them
        if (!widen_f32) {
            try out.put(name, if (t.dtype == .bf16) "BF16" else "F32", &.{ t.dim(0), t.dim(1) }, try gpa.dupe(u8, t.bytes));
            return;
        }
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const bytes = try gpa.alloc(u8, t.dim(0) * t.dim(1) * 4);
        const out32 = std.mem.bytesAsSlice(u32, bytes);
        for (0..t.dim(0)) |ch| for (0..t.dim(1)) |tap| {
            out32[ch * t.dim(1) + tap] = @bitCast(bf16ToF32(std.mem.readInt(u16, t.bytes[(ch * t.dim(1) + tap) * 2 ..][0..2], .little)));
        };
        try out.put(name, "F32", &.{ t.dim(0), t.dim(1) }, bytes);
        return;
    }
    if (t.dim(1) != 1) return error.UnexpectedTensor;
    // [W, 1, C]: the same channels in the packed spelling, transposed into channel-major rows
    const w = t.dim(0);
    const c = t.dim(2);
    if (!widen_f32) {
        const bytes = try gpa.alloc(u8, c * w * es);
        for (0..c) |ch| for (0..w) |tap| {
            @memcpy(bytes[(ch * w + tap) * es ..][0..es], t.bytes[(tap * c + ch) * es ..][0..es]);
        };
        try out.put(name, if (t.dtype == .bf16) "BF16" else "F32", &.{ c, w }, bytes);
        return;
    }
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const bytes = try gpa.alloc(u8, c * w * 4);
    const out32 = std.mem.bytesAsSlice(u32, bytes);
    for (0..c) |ch| for (0..w) |tap| {
        out32[ch * w + tap] = @bitCast(bf16ToF32(std.mem.readInt(u16, t.bytes[(tap * c + ch) * 2 ..][0..2], .little)));
    };
    try out.put(name, "F32", &.{ c, w }, bytes);
}

/// The n-gram tables' group starts: 8 groups of ceil(shards/8), each start the rows before it (PleTables, embed.py).
fn pleStarts(gpa: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, ple_layer: usize, spelling: []const u8) ![]u8 {
    const groups = 8;
    const per = (cfg.ngram_shards + groups - 1) / groups;
    const out = try gpa.alloc(u8, groups * 4);
    errdefer gpa.free(out);
    const out32 = std.mem.bytesAsSlice(u32, out);
    var at: usize = 0;
    for (0..groups) |g| {
        out32[g] = @intCast(at);
        for (0..per) |s| {
            const id = g * per + s;
            if (id >= cfg.ngram_shards) break;
            var buf: [256]u8 = undefined;
            const t = try ck.get(try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}.ple.ple_embedding.ngram_embedding.{s}{d}.weight", .{ ple_layer, spelling, id }));
            at += t.dim(0);
        }
    }
    return out;
}

/// Draft vocabulary ids are sorted and padded to a multiple of 64, and an id past the vocabulary is refused.
pub fn draftIdList(gpa: std.mem.Allocator, io: Io, path: []const u8, vocab: usize) ![]u32 {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    var listed: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer listed.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |token| {
        const id = std.fmt.parseInt(u32, token, 10) catch return error.BadDraftVocab;
        if (id >= vocab) return error.DraftIdPastVocab;
        try listed.put(gpa, id, {});
    }
    if (listed.count() == 0) return error.BadDraftVocab;
    var extra: std.ArrayList(u32) = .empty;
    defer extra.deinit(gpa);
    var candidate: u32 = 0;
    while ((listed.count() + extra.items.len) % 64 != 0) {
        if (candidate >= vocab) return error.DraftVocabExhausted; // a pad past the vocabulary would gather past lm_head
        if (!listed.contains(candidate)) try extra.append(gpa, candidate);
        candidate += 1;
    }
    const all = try gpa.alloc(u32, listed.count() + extra.items.len);
    errdefer gpa.free(all);
    var n: usize = 0;
    var it2 = listed.keyIterator();
    while (it2.next()) |k| {
        all[n] = k.*;
        n += 1;
    }
    @memcpy(all[n..], extra.items);
    std.mem.sort(u32, all, {}, std.sort.asc(u32));
    return all;
}

/// The MTP draft head: lm_head's rows gathered by the id list, then the lane pipeline (draft_head.cut_head).
fn putDraftHead(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, name: []const u8, lm_head: Q, ids: []const u32) !void {
    const rows = ids.len;
    const kw = lm_head.kw;
    const kg = lm_head.s.dim(1);
    const wq = try gpa.alloc(u8, rows * kw * 4);
    const s = try gpa.alloc(u8, rows * kg * 2);
    const b = try gpa.alloc(u8, rows * kg * 2);
    const w32 = std.mem.bytesAsSlice(u32, wq);
    const src32 = std.mem.bytesAsSlice(u32, lm_head.part(.weight));
    const s16 = std.mem.bytesAsSlice(u16, s);
    const src_s16 = std.mem.bytesAsSlice(u16, lm_head.part(.scales));
    const b16 = std.mem.bytesAsSlice(u16, b);
    const src_b16 = std.mem.bytesAsSlice(u16, lm_head.part(.biases));
    for (ids, 0..) |id, r| {
        if (id * kw + kw > src32.len or id * kg + kg > src_s16.len) return error.DraftIdPastVocab;
        @memcpy(w32[r * kw ..][0..kw], src32[id * kw ..][0..kw]);
        @memcpy(s16[r * kg ..][0..kg], src_s16[id * kg ..][0..kg]);
        @memcpy(b16[r * kg ..][0..kg], src_b16[id * kg ..][0..kg]);
    }
    const gathered = [_]Q{.{ .w = .{ .dtype = .u32, .rank = 2, .shape = .{ rows, kw, 0, 0 }, .bytes = wq }, .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0 }, .bytes = s }, .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0 }, .bytes = b }, .rows = rows, .k = lm_head.k, .kw = kw, .spec = lm_head.spec }};
    try putLane(out, gpa, name, gathered[0..]); // putLane copies the bytes, so the gathered buffers free here
    gpa.free(wq);
    gpa.free(s);
    gpa.free(b);
    _ = a;
}

/// Build the three packs into `out_dir`. A null draft vocab omits the mtp tensors. Dims come from the checkpoint.
pub fn build(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, out_dir: []const u8, draft_vocab: ?[]const u8) !Report {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var cfg = try config_mod.Config.read(gpa, io, model_dir);
    defer cfg.deinit();
    // the vision tower's tensors stay unread: a rank-five entry must not stop the no-dump build
    var ck = try ckpt.Checkpoint.openModelPrefix(gpa, io, model_dir, "language_model.");
    defer ck.close();
    const wide = try cfg.wide();
    var report: Report = .{};
    report.norms_around_one = try normsAroundOne(&ck, &cfg, wide);
    // the checkpoint's index drives the shard spelling, the source identity and the recorded shard list
    const index_path = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const index_text = try Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(index_text);
    var parsed_index = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    defer parsed_index.deinit();
    const weight_map = parsed_index.value.object.get("weight_map").?.object;
    const spelling = index_mod.ngramSpelling(weight_map);
    const identity = try pio.sourceIdentity(gpa, io, model_dir);
    defer gpa.free(identity);

    // pack.safetensors: the decode layout, tensor for tensor as build_pack emits it.
    var decode = Out.init(gpa);
    defer decode.deinit();
    const eps = try gpa.alloc(u8, 4);
    std.mem.bytesAsSlice(u32, eps)[0] = @bitCast(cfg.eps);
    try decode.put("eps", "F32", &.{1}, eps);
    var buf: [256]u8 = undefined;
    for (0..cfg.layers) |i| {
        const stem = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{i});
        try putHc(&decode, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.ahc", .{i}), try std.fmt.allocPrint(a, "{s}.attn_hyper_connection", .{stem}), true, report.norms_around_one, wide);
        try putHc(&decode, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.mhc", .{i}), try std.fmt.allocPrint(a, "{s}.mlp_hyper_connection", .{stem}), true, report.norms_around_one, wide);
        try putRouter(&decode, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.moe.router", .{i}), try std.fmt.allocPrint(a, "{s}.mlp.gate", .{stem}), try std.fmt.allocPrint(a, "{s}.mlp.shared_expert_gate", .{stem}));
        if ((try cfg.kind(i)) == .linear_attention) {
            var in_members: [4]Q = undefined;
            const in_names = [_][]const u8{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a" };
            for (in_names, 0..) |n, j| in_members[j] = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, n }));
            try putLane(&decode, gpa, try std.fmt.allocPrint(a, "L{d}.gdn.in", .{i}), in_members[0..]);
            const out_proj = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.linear_attn.out_proj", .{stem}));
            const out_members = [_]Q{out_proj};
            try putLane(&decode, gpa, try std.fmt.allocPrint(a, "L{d}.gdn.out", .{i}), out_members[0..]);
            try putConvSlice(&decode, gpa, &ck, try std.fmt.allocPrint(a, "L{d}.gdn.conv", .{i}), try std.fmt.allocPrint(a, "{s}.linear_attn.conv1d.weight", .{stem}), false);
            const alog = try ck.get(try std.fmt.allocPrint(a, "{s}.linear_attn.A_log", .{stem}));
            if (alog.rank != 1 or (alog.dtype != .bf16 and alog.dtype != .f32)) return error.UnexpectedTensor;
            // Gdn ALOG and DT are bf16 in the pack, copied when stored as bf16 and rounded from f32 otherwise.
            const alog16 = try gpa.alloc(u8, alog.dim(0) * 2);
            if (alog.dtype == .bf16) @memcpy(alog16, alog.bytes) else for (0..alog.dim(0)) |j| std.mem.writeInt(u16, alog16[j * 2 ..][0..2], f32ToBf16(std.mem.readInt(u32, alog.bytes[j * 4 ..][0..4], .little)), .little);
            try decode.put(try std.fmt.allocPrint(a, "L{d}.gdn.alog", .{i}), "BF16", &.{alog.dim(0)}, alog16);
            const dt = try ck.get(try std.fmt.allocPrint(a, "{s}.linear_attn.dt_bias", .{stem}));
            if (dt.rank != 1 or (dt.dtype != .bf16 and dt.dtype != .f32)) return error.UnexpectedTensor;
            const dt16 = try gpa.alloc(u8, dt.dim(0) * 2);
            if (dt.dtype == .bf16) @memcpy(dt16, dt.bytes) else for (0..dt.dim(0)) |j| std.mem.writeInt(u16, dt16[j * 2 ..][0..2], f32ToBf16(std.mem.readInt(u32, dt.bytes[j * 4 ..][0..4], .little)), .little);
            try decode.put(try std.fmt.allocPrint(a, "L{d}.gdn.dt", .{i}), "BF16", &.{dt.dim(0)}, dt16);
            const norm = try ck.get(try std.fmt.allocPrint(a, "{s}.linear_attn.norm.weight", .{stem}));
            if (norm.dtype != .bf16 or norm.rank != 1) return error.UnexpectedTensor;
            try decode.put(try std.fmt.allocPrint(a, "L{d}.gdn.norm", .{i}), "BF16", &.{norm.dim(0)}, try gpa.dupe(u8, norm.bytes));
        } else {
            var proj_members: [4]Q = undefined;
            const proj_names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj" };
            for (proj_names, 0..) |n, j| proj_members[j] = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, n }));
            try putLane(&decode, gpa, try std.fmt.allocPrint(a, "L{d}.att.proj", .{i}), proj_members[0..]);
            const o = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{stem}));
            const o_members = [_]Q{o};
            try putLane(&decode, gpa, try std.fmt.allocPrint(a, "L{d}.att.o", .{i}), o_members[0..]);
            for ([_][]const u8{ "q_norm.weight", "k_norm.weight" }, [_][]const u8{ "qn", "kn" }) |n, pack_name| {
                const scale = try centeredScale(gpa, &ck, try std.fmt.allocPrint(a, "{s}.self_attn.{s}", .{ stem, n }), report.norms_around_one);
                try decode.put(try std.fmt.allocPrint(a, "L{d}.att.{s}", .{ i, pack_name }), "F32", &.{cfg.head_dim}, scale);
            }
            for ([_][]const u8{ "q_layernorm.weight", "k_layernorm.weight" }, [_][]const u8{ "iqn", "pool" }) |n, pack_name| {
                const scale = try centeredScale(gpa, &ck, try std.fmt.allocPrint(a, "{s}.self_attn.indexer.{s}", .{ stem, n }), report.norms_around_one);
                try decode.put(try std.fmt.allocPrint(a, "L{d}.att.{s}", .{ i, pack_name }), "F32", &.{cfg.index_dim}, scale);
            }
        }
    }
    try putHc(&decode, gpa, a, &ck, &cfg, "mix", "language_model.model.hyper_connection_mixer", false, report.norms_around_one, wide);
    const head = try qlinear(&ck, &cfg, "language_model.lm_head");
    const head_members = [_]Q{head};
    try putLane(&decode, gpa, "head", head_members[0..]);
    var ple_layer: ?usize = null;
    for (0..cfg.layers) |i| {
        if (cfg.ple[i]) ple_layer = i;
    }
    if (ple_layer) |pl| {
        const stem = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{pl});
        var kv_members: [2]Q = undefined;
        kv_members[0] = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.ple.key_proj", .{stem}));
        kv_members[1] = try qlinear(&ck, &cfg, try std.fmt.allocPrint(a, "{s}.ple.value_proj", .{stem}));
        try putLane(&decode, gpa, "ple.kv", kv_members[0..]);
        for ([_][]const u8{ "norm_key.weight", "norm_query.weight", "norm_conv.weight" }, [_][]const u8{ "ks", "qs", "cs" }) |n, pack_name| {
            const scale = try centeredScale(gpa, &ck, try std.fmt.allocPrint(a, "{s}.ple.{s}", .{ stem, n }), report.norms_around_one);
            try decode.put(try std.fmt.allocPrint(a, "ple.{s}", .{pack_name}), "F32", &.{wide}, scale);
        }
        try putConvSlice(&decode, gpa, &ck, "ple.conv", try std.fmt.allocPrint(a, "{s}.ple.conv1d.weight", .{stem}), true);
        const starts = try pleStarts(gpa, &ck, &cfg, pl, spelling);
        try decode.put("ple.starts", "U32", &.{8}, starts);
    }
    if (draft_vocab) |vocab_path| {
        const ids = try draftIdList(gpa, io, vocab_path, cfg.vocab);
        report.draft_ids = ids.len;
        try putMtp(&decode, gpa, a, &ck, &cfg, ids, &report, wide);
        try decode.put("mtp.draft_ids", "U32", &.{ids.len}, ids);
    }
    report.decode_tensors = decode.names.items.len;
    try decode.write(gpa, io, try std.fs.path.join(a, &.{ out_dir, "pack.safetensors" }), identity);

    // pack_mlx.safetensors: the prompt path's projections in the checkpoint's own MLX layout.
    var mlx = Out.init(gpa);
    defer mlx.deinit();
    for (0..cfg.layers) |i| {
        const stem = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{i});
        if ((try cfg.kind(i)) == .linear_attention) {
            try putMlxProjection(&mlx, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.gdn.in", .{i}), &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a" }, stem);
            try putMlxProjection(&mlx, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.gdn.out", .{i}), &.{"linear_attn.out_proj"}, stem);
        } else {
            try putMlxProjection(&mlx, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.att.proj", .{i}), &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj" }, stem);
            try putMlxProjection(&mlx, gpa, a, &ck, &cfg, try std.fmt.allocPrint(a, "L{d}.att.o", .{i}), &.{"self_attn.o_proj"}, stem);
        }
    }
    if (ple_layer) |pl| {
        const stem = try std.fmt.bufPrint(&buf, "language_model.model.layers.{d}", .{pl});
        try putMlxProjection(&mlx, gpa, a, &ck, &cfg, "ple.kv", &.{ "ple.key_proj", "ple.value_proj" }, stem);
    }
    report.mlx_tensors = mlx.names.items.len;
    try mlx.write(gpa, io, try std.fs.path.join(a, &.{ out_dir, "pack_mlx.safetensors" }), identity);

    // pack_mtp_mlx.safetensors: the MTP head's prompt projections, independent of the draft vocabulary.
    if (cfg.mtp.layers > 0) {
        var mtp_mlx = Out.init(gpa);
        defer mtp_mlx.deinit();
        const mtp_stem = "language_model.mtp.layers.0";
        try putMlxProjection(&mtp_mlx, gpa, a, &ck, &cfg, "mtp.att.proj", &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj" }, mtp_stem);
        try putMlxProjection(&mtp_mlx, gpa, a, &ck, &cfg, "mtp.fce", &.{"fc_embedding"}, "language_model.mtp");
        try putMlxProjection(&mtp_mlx, gpa, a, &ck, &cfg, "mtp.fch", &.{"fc_hidden"}, "language_model.mtp");
        report.mtp_mlx_tensors = mtp_mlx.names.items.len;
        try mtp_mlx.write(gpa, io, try std.fs.path.join(a, &.{ out_dir, "pack_mtp_mlx.safetensors" }), identity);
    }
    return report;
}

/// One export_mlx_proj-style projection: `.mw` the checkpoint's packed words, `.ms`/`.mb` its scales and biases.
fn putMlxProjection(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, name: []const u8, members: []const []const u8, stem: []const u8) !void {
    var storage: [4]Q = undefined;
    for (members, 0..) |n, i| storage[i] = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, n }));
    const list = storage[0..members.len];
    const mw = try concatPart(gpa, list, .weight);
    const ms = try concatPart(gpa, list, .scales);
    const mb = try concatPart(gpa, list, .biases);
    var rows: usize = 0;
    for (list) |m| rows += m.rows;
    try out.put(try std.fmt.allocPrint(a, "{s}.mw", .{name}), "U32", &.{ rows, list[0].kw }, mw);
    try out.put(try std.fmt.allocPrint(a, "{s}.ms", .{name}), "BF16", &.{ rows, list[0].s.dim(1) }, ms);
    try out.put(try std.fmt.allocPrint(a, "{s}.mb", .{name}), "BF16", &.{ rows, list[0].b.dim(1) }, mb);
}

/// The mtp.* decode tensors: the MTP layer's hc/att/moe classes, its norms, and the draft head over the id list.
fn putMtp(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, ids: []const u32, report: *Report, wide: usize) !void {
    const layer_stem = "language_model.mtp.layers.0";
    try putHc(out, gpa, a, ck, cfg, "mtp.ahc", try std.fmt.allocPrint(a, "{s}.attn_hyper_connection", .{layer_stem}), true, report.norms_around_one, wide);
    try putHc(out, gpa, a, ck, cfg, "mtp.mhc", try std.fmt.allocPrint(a, "{s}.mlp_hyper_connection", .{layer_stem}), true, report.norms_around_one, wide);
    try putRouter(out, gpa, a, ck, cfg, "mtp.moe.router", try std.fmt.allocPrint(a, "{s}.mlp.gate", .{layer_stem}), try std.fmt.allocPrint(a, "{s}.mlp.shared_expert_gate", .{layer_stem}));
    var proj_members: [4]Q = undefined;
    const proj_names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj" };
    for (proj_names, 0..) |n, j| proj_members[j] = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.{s}", .{ layer_stem, n }));
    try putLane(out, gpa, "mtp.att.proj", proj_members[0..]);
    const o = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{layer_stem}));
    const o_members = [_]Q{o};
    try putLane(out, gpa, "mtp.att.o", o_members[0..]);
    for ([_][]const u8{ "q_norm.weight", "k_norm.weight" }, [_][]const u8{ "qn", "kn" }) |n, pack_name| {
        const scale = try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.self_attn.{s}", .{ layer_stem, n }), report.norms_around_one);
        try out.put(try std.fmt.allocPrint(a, "mtp.att.{s}", .{pack_name}), "F32", &.{cfg.head_dim}, scale);
    }
    for ([_][]const u8{ "q_layernorm.weight", "k_layernorm.weight" }, [_][]const u8{ "iqn", "pool" }) |n, pack_name| {
        const scale = try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.self_attn.indexer.{s}", .{ layer_stem, n }), report.norms_around_one);
        try out.put(try std.fmt.allocPrint(a, "mtp.att.{s}", .{pack_name}), "F32", &.{cfg.index_dim}, scale);
    }
    try putHc(out, gpa, a, ck, cfg, "mtp.mix", "language_model.mtp.hyper_connection_mixer", false, report.norms_around_one, wide);
    const fce = try qlinear(ck, cfg, "language_model.mtp.fc_embedding");
    const fce_members = [_]Q{fce};
    try putLane(out, gpa, "mtp.fce", fce_members[0..]);
    const fch = try qlinear(ck, cfg, "language_model.mtp.fc_hidden");
    const fch_members = [_]Q{fch};
    try putLane(out, gpa, "mtp.fch", fch_members[0..]);
    const enorm = try centeredScale(gpa, ck, "language_model.mtp.pre_fc_norm_embedding.weight", report.norms_around_one);
    try out.put("mtp.enorm.scale", "F32", &.{cfg.hidden}, enorm);
    const hnorm = try centeredScale(gpa, ck, "language_model.mtp.pre_fc_norm_hidden.weight", report.norms_around_one);
    try out.put("mtp.hnorm.scale", "F32", &.{wide}, hnorm);
    const head = try qlinear(ck, cfg, "language_model.lm_head");
    try putDraftHead(out, gpa, a, "mtp.draft", head, ids);
}

pub const compareFile = pio.compareFile;

test {
    _ = @import("pack_test.zig");
}
