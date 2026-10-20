//! Flash Next packs built in Zig from the checkpoint match the dump tools byte for byte, with no model load.
const std = @import("std");
const Io = std.Io;
const st = @import("../../core/safetensors.zig");
const ckpt = @import("../../core/checkpoint.zig");
const affine = @import("affine.zig");
const config_mod = @import("config.zig");
const widths = @import("widths.zig");
const index_mod = @import("index.zig");
const pio = @import("pack_io.zig");
const quant_mlx = @import("../../core/quant/mlx.zig");
const quant_widen = @import("../../core/quant/widen.zig");
const widen = @import("pack_widen.zig");
const Module = @This();
pub const testing = if (@import("builtin").is_test) struct {
    pub const putDraftHead = Module.putDraftHead;
} else struct {};

pub const Report = struct {
    decode_tensors: usize = 0,
    mlx_tensors: usize = 0,
    mtp_mlx_tensors: usize = 0,
    norms_around_one: bool = false,
    draft_ids: usize = 0,
};

pub const Out = pio.Out;

pub const lane = @import("pack_lane.zig");
pub const Q = lane.Q;
pub const tileWeight = lane.tileWeight;
pub const packScales = lane.packScales;
const concatPart = lane.concatPart;
const putLane = lane.putLane;

/// A quantized linear read only at its own declared width, refused under its name otherwise.
fn qlinear(ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, path: []const u8) !Q {
    return qlinearAt(ck, cfg, path) catch |err| {
        std.log.warn("{s}: not read at its declared width: {s}", .{ path, @errorName(err) });
        return err;
    };
}

fn qlinearAt(ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, path: []const u8) !Q {
    const spec = (try widths.quantization(cfg, path)) orelse return error.NotQuantized;
    var buf: [320]u8 = undefined;
    const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{path}));
    const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{path}));
    const b = try ck.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{path}));
    if (w.dtype != .u32 or w.rank != 2) return error.UnexpectedTensor;
    if (s.dtype != .bf16 or s.rank != 2 or b.dtype != .bf16 or b.rank != 2) return error.UnexpectedTensor;
    const rows = w.dim(0);
    const kw = w.dim(1);
    if (rows != s.dim(0) or rows != b.dim(0)) return error.UnexpectedTensor;
    if (kw * 32 % spec.bits != 0) return error.UnexpectedTensor;
    const k = kw * 32 / spec.bits;
    if (k % spec.group != 0 or s.dim(1) != k / spec.group or b.dim(1) != k / spec.group) return error.UnexpectedTensor;
    return .{ .w = w, .s = s, .b = b, .rows = rows, .k = k, .kw = kw, .spec = spec };
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
    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();
    const down = try hcDown(scratch.allocator(), storage[0..down_n]); // a pair of two widths widens to the wider (QWeights.of)
    const up = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.input_mix_weight_up", .{stem}));
    const scale_name = try std.fmt.allocPrint(a, "{s}.scale", .{name});
    try out.put(scale_name, "F32", &.{wide}, try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.hc_norm.weight", .{stem}), around_one));
    try out.put(try std.fmt.allocPrint(a, "{s}.down.w", .{name}), "U32", &.{ down.rows, down.kw }, try gpa.dupe(u8, down.part(.weight)));
    try out.put(try std.fmt.allocPrint(a, "{s}.down.s", .{name}), "BF16", &.{ down.rows, down.s.dim(1) }, try gpa.dupe(u8, down.part(.scales)));
    try out.put(try std.fmt.allocPrint(a, "{s}.down.b", .{name}), "BF16", &.{ down.rows, down.b.dim(1) }, try gpa.dupe(u8, down.part(.biases)));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.w", .{name}), "U32", &.{ up.rows, up.kw }, try gpa.dupe(u8, up.part(.weight)));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.s", .{name}), "BF16", &.{ up.rows, up.s.dim(1) }, try gpa.dupe(u8, up.part(.scales)));
    try out.put(try std.fmt.allocPrint(a, "{s}.up.b", .{name}), "BF16", &.{ up.rows, up.b.dim(1) }, try gpa.dupe(u8, up.part(.biases)));
    // the uniform packs' pinned bytes
    if (down.spec.bits == 6 and down.spec.group == 32 and up.spec.bits == 6 and up.spec.group == 32) return;
    const widths_name = try std.fmt.allocPrint(a, "{s}.widths", .{name});
    const w = try gpa.alloc(u32, 4);
    w[0..4].* = .{ down.spec.bits, @intCast(down.spec.group), up.spec.bits, @intCast(up.spec.group) };
    try out.put(widths_name, "U32", &.{4}, w);
}

/// The down projection and the inject rows as one matrix: one width as stored, two widened exactly to the wider.
pub fn hcDown(a: std.mem.Allocator, members: []const Q) !Q {
    var hosts: [2]quant_mlx.Host = undefined;
    for (members, 0..) |m, i| hosts[i] = widen.host(m);
    return widen.fromHost(try quant_widen.fuse(a, hosts[0..members.len]));
}

/// A gate's rows are stored bf16, or an affine linear dequantized on the host as code * scale + bias per group.
const Dense = struct { bytes: []const u8, rows: usize, cols: usize, owned: bool };

fn denseRows(gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, stem: []const u8) !Dense {
    const w = try ck.get(try std.fmt.allocPrint(a, "{s}.weight", .{stem}));
    if (w.dtype == .bf16 and w.rank == 2) return .{ .bytes = w.bytes, .rows = w.dim(0), .cols = w.dim(1), .owned = false };
    if (w.dtype != .u32 or w.rank != 2) return error.DenseRouterNotBf16;
    const spec = (try widths.quantization(cfg, stem)) orelse return error.DenseRouterNotBf16;
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
        const row = w.bytes[r * words_per_row ..][0..words_per_row];
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

/// An n-gram table's weight_scale must be one finite, nonzero bf16 or f32 value: the lookups multiply rows by it.
fn tableScale(ck: *ckpt.Checkpoint, map: std.json.ObjectMap, name: []const u8) !void {
    if (!map.contains(name)) return;
    const t = try ck.get(name);
    const v: ?f32 = switch (t.dtype) {
        .bf16 => if (t.bytes.len == 2) @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[0..2], .little)) << 16) else null,
        .f32 => if (t.bytes.len == 4) @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little)) else null,
        else => null,
    };
    if (v) |x| if (std.math.isFinite(x) and x != 0) return;
    std.log.warn("{s}: an n-gram table scale must be one finite, nonzero value", .{name});
    return error.NgramTableScale;
}

/// Draft vocabulary ids are sorted and padded to a multiple of 64, and an id past the vocabulary is refused.
pub fn draftIdList(gpa: std.mem.Allocator, io: Io, path: []const u8, vocab: usize) ![]u32 {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    return @import("../../core/proposal_vocab.zig").legacy.ids(gpa, text, .{ .vocab = @intCast(vocab), .tile = 64 });
}

/// The MTP draft head: lm_head's rows gathered by the id list, then the lane pipeline (draft_head.cut_head).
fn putDraftHead(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, name: []const u8, lm_head: Q, ids: []const u32) !void {
    const rows = ids.len;
    const kw = lm_head.kw;
    const kg = lm_head.s.dim(1);
    const pv = @import("../../core/proposal_vocab.zig");
    const storage = @import("../../core/proposal_row_storage.zig");
    const pr = storage.Views(storage.Shape);
    const format = try std.fmt.allocPrint(a, "affine{d}", .{lm_head.spec.bits});
    defer a.free(format);
    var artifact = try pv.legacy.artifact(gpa, ids, .{ .vocab = @intCast(lm_head.rows), .hidden = @intCast(lm_head.k), .format = format, .group = @intCast(lm_head.spec.group), .scale_recipe = "existing packed lane" });
    defer artifact.deinit();
    var selected = try (pr.Source{ .identity = artifact.identity(), .storage = .affine, .recipe = .affine_mm, .scale_dtype = .bf16, .bias_dtype = .bf16, .codes = .{ .bytes = lm_head.part(.weight), .stride = kw * 4 }, .scales = .{ .bytes = lm_head.part(.scales), .stride = kg * 2 }, .biases = .{ .bytes = lm_head.part(.biases), .stride = kg * 2 } }).extract(gpa, &artifact, 1, std.math.maxInt(usize));
    defer selected.deinit();
    const wq = selected.codes;
    const s = selected.scales;
    const b = selected.biases;
    const gathered = [_]Q{.{ .w = .{ .dtype = .u32, .rank = 2, .shape = .{ rows, kw, 0, 0, 0 }, .bytes = wq }, .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0, 0 }, .bytes = s }, .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, kg, 0, 0, 0 }, .bytes = b }, .rows = rows, .k = lm_head.k, .kw = kw, .spec = lm_head.spec }};
    try putLane(out, gpa, name, gathered[0..]);
}

/// Build the three packs into `out_dir`. A null draft vocab omits the mtp tensors. Dims come from the checkpoint.
pub fn build(gpa: std.mem.Allocator, io: Io, model_dir: []const u8, out_dir: []const u8, draft_vocab: ?[]const u8) !Report {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var cfg = try config_mod.Config.read(gpa, io, model_dir);
    defer cfg.deinit();
    var strict = try widths.Widths.init(gpa, &cfg);
    defer strict.deinit();
    cfg.widths = &strict;
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
    const weight_map = try st.index.weightMap(parsed_index.value);
    const spelling = index_mod.ngramSpelling(weight_map);
    const mtp_root = index_mod.mtpRoot(weight_map); // oMLX keeps the head outside language_model.
    var ck_mtp = if (std.mem.eql(u8, mtp_root, "mtp")) try ckpt.Checkpoint.openModelPrefix(gpa, io, model_dir, "mtp.") else null;
    defer if (ck_mtp) |*m| m.close();
    const ckm = if (ck_mtp) |*m| m else &ck;
    try @import("pack_raw.zig").checkAll(&ck, ckm, &strict, &cfg, mtp_root, spelling); // the directly read tensors
    const identity = try pio.sourceIdentity(gpa, io, model_dir);
    defer gpa.free(identity);

    var buf: [256]u8 = undefined;
    var ple_layer: ?usize = null;
    for (0..cfg.layers) |i| {
        if (cfg.ple[i]) ple_layer = i;
    }
    // Each output is written and freed before the next pack is constructed.
    {
        // pack.safetensors: the decode layout, tensor for tensor as build_pack emits it.
        var decode = Out.init(gpa);
        defer decode.deinit();
        const eps = try gpa.alloc(u8, 4);
        std.mem.bytesAsSlice(u32, eps)[0] = @bitCast(cfg.eps);
        try decode.put("eps", "F32", &.{1}, eps);
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
            try tableScale(&ck, weight_map, try std.fmt.allocPrint(a, "{s}.ple.ple_embedding.ngram_embedding.weight_scale", .{stem}));
            const starts = try pleStarts(gpa, &ck, &cfg, pl, spelling);
            try decode.put("ple.starts", "U32", &.{8}, starts);
        }
        if (draft_vocab) |vocab_path| {
            const ids = try draftIdList(gpa, io, vocab_path, cfg.vocab);
            report.draft_ids = ids.len;
            try putMtp(&decode, gpa, a, ckm, &ck, mtp_root, &cfg, ids, &report, wide);
            try decode.put("mtp.draft_ids", "U32", &.{ids.len}, ids);
        }
        report.decode_tensors = decode.names.items.len;
        try decode.write(gpa, io, try std.fs.path.join(a, &.{ out_dir, "pack.safetensors" }), identity);
    }
    {
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
    }
    // pack_mtp_mlx.safetensors: the MTP head's prompt projections, independent of the draft vocabulary.
    if (cfg.mtp.layers > 0) {
        var mtp_mlx = Out.init(gpa);
        defer mtp_mlx.deinit();
        const mtp_stem = try std.fmt.allocPrint(a, "{s}.layers.0", .{mtp_root});
        try putMlxProjection(&mtp_mlx, gpa, a, ckm, &cfg, "mtp.att.proj", &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.indexer.index_qk_proj" }, mtp_stem);
        try putMlxProjection(&mtp_mlx, gpa, a, ckm, &cfg, "mtp.fce", &.{"fc_embedding"}, mtp_root);
        try putMlxProjection(&mtp_mlx, gpa, a, ckm, &cfg, "mtp.fch", &.{"fc_hidden"}, mtp_root);
        report.mtp_mlx_tensors = mtp_mlx.names.items.len;
        try mtp_mlx.write(gpa, io, try std.fs.path.join(a, &.{ out_dir, "pack_mtp_mlx.safetensors" }), identity);
    }
    return report;
}

/// One export_mlx_proj-style projection (`.mw` words, `.ms`/`.mb` tables); `.parts` gives other widths per member.
fn putMlxProjection(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, cfg: *const config_mod.Config, name: []const u8, members: []const []const u8, stem: []const u8) !void {
    var storage: [4]Q = undefined;
    for (members, 0..) |n, i| storage[i] = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.{s}", .{ stem, n }));
    return mlxProjection(out, gpa, a, name, storage[0..members.len]);
}

/// The members' MLX bytes in member order; 6/32 alone keeps its 2-D shapes, other widths go flat with `.parts`.
pub fn mlxProjection(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, name: []const u8, list: []const Q) !void {
    var rows: usize = 0;
    for (list) |m| rows += m.rows;
    const six = for (list) |m| {
        if (m.spec.bits != 6 or m.spec.group != 32) break false;
    } else true;
    // the uniform packs keep their bytes and shapes; members of other widths go flat, in member order
    try mlxPart(out, gpa, a, name, "mw", "U32", list, .weight, if (six) .{ rows, list[0].kw } else null);
    try mlxPart(out, gpa, a, name, "ms", "BF16", list, .scales, if (six) .{ rows, list[0].s.dim(1) } else null);
    try mlxPart(out, gpa, a, name, "mb", "BF16", list, .biases, if (six) .{ rows, list[0].b.dim(1) } else null);
    if (six) return;
    const parts_name = try std.fmt.allocPrint(a, "{s}.parts", .{name}); // each member's rows, bits and group
    const parts = try gpa.alloc(u32, list.len * 3);
    for (list, 0..) |m, i| parts[i * 3 ..][0..3].* = .{ @intCast(m.rows), m.spec.bits, @intCast(m.spec.group) };
    try out.put(parts_name, "U32", &.{ list.len, 3 }, parts);
}

/// One of a projection's three tensors, members concatenated: at `shape` when given, else flat.
fn mlxPart(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, name: []const u8, suffix: []const u8, dtype: []const u8, list: []const Q, comptime part: lane.Part, shape: ?[2]usize) !void {
    const full = try std.fmt.allocPrint(a, "{s}.{s}", .{ name, suffix });
    const bytes = try concatPart(gpa, list, part);
    if (shape) |sh| return out.put(full, dtype, &sh, bytes);
    return out.put(full, dtype, &.{bytes.len / @as(usize, if (part == .weight) 4 else 2)}, bytes);
}

/// The mtp.* decode tensors: the MTP layer's hc/att/moe classes, its norms, and the draft head over the id list.
fn putMtp(out: *Out, gpa: std.mem.Allocator, a: std.mem.Allocator, ck: *ckpt.Checkpoint, text: *ckpt.Checkpoint, root: []const u8, cfg: *const config_mod.Config, ids: []const u32, report: *Report, wide: usize) !void {
    const layer_stem = try std.fmt.allocPrint(a, "{s}.layers.0", .{root});
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
    try putHc(out, gpa, a, ck, cfg, "mtp.mix", try std.fmt.allocPrint(a, "{s}.hyper_connection_mixer", .{root}), false, report.norms_around_one, wide);
    const fce = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.fc_embedding", .{root}));
    const fce_members = [_]Q{fce};
    try putLane(out, gpa, "mtp.fce", fce_members[0..]);
    const fch = try qlinear(ck, cfg, try std.fmt.allocPrint(a, "{s}.fc_hidden", .{root}));
    const fch_members = [_]Q{fch};
    try putLane(out, gpa, "mtp.fch", fch_members[0..]);
    const enorm = try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.pre_fc_norm_embedding.weight", .{root}), report.norms_around_one);
    try out.put("mtp.enorm.scale", "F32", &.{cfg.hidden}, enorm);
    const hnorm = try centeredScale(gpa, ck, try std.fmt.allocPrint(a, "{s}.pre_fc_norm_hidden.weight", .{root}), report.norms_around_one);
    try out.put("mtp.hnorm.scale", "F32", &.{wide}, hnorm);
    const head = try qlinear(text, cfg, "language_model.lm_head"); // the head sits with the text model
    try putDraftHead(out, gpa, a, "mtp.draft", head, ids);
}

pub const compareFile = pio.compareFile;

test {
    _ = @import("pack_test.zig");
    _ = @import("pack_hc_test.zig");
}
