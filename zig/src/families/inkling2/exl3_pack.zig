//! Host pack validation extending exl3_config.zig for single-Mac Metal EXL3 packs (no GPU).
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const exl3 = core.exl3_format;
const config = @import("config.zig");
const exl3_cfg = @import("exl3_config.zig");
const names = @import("names.zig");

pub const format_id = exl3_cfg.format;
pub const min_bits: u8 = 2;
pub const max_bits: u8 = 4;

pub const Layout = enum { single_dense, split_dense };

pub const Options = struct {
    vision: bool = false,
    cfg: bool = false,
    loops: ?u32 = null,
    precision: ?[]const u8 = null,
};

pub const Batch = struct {
    start: u32,
    end: u32,
    file: []const u8,
    bytes: u64,
};

pub const LayerManifest = struct {
    layer: u32,
    batches: []Batch,
    k_by_expert: ?[]u8,
};

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    layers: []LayerManifest,

    pub fn deinit(self: *Manifest) void {
        self.arena.deinit();
    }
};

pub const Summary = struct {
    layout: Layout,
    layers: u32,
    experts: u32,
    dense_tensors: usize,
    expert_shards: usize,
    width_counts: [max_bits + 1]u64 = @splat(0),
    dense_bytes: u64 = 0,
    expert_bytes: u64 = 0,
    has_mtp: bool = false,
    mtp_bytes: u64 = 0,
};

/// Serve-option refusals for text-only Inkling 2 Small EXL3 packs.
pub fn refusal(o: Options) ?[]const u8 {
    if (o.vision) return "inkling2 EXL3 pack is text-only; serve without --vision";
    if (o.cfg) return "inkling2 EXL3 pack: classifier-free guidance is unsupported";
    if (o.loops != null) return "inkling2 Small has no core loop; serve without --loops";
    if (o.precision != null) return "inkling2 EXL3 pack precision is fixed to stored EXL3 mul1 experts and BF16 dense weights; serve without --precision";
    return null;
}

/// Exact non-looped Inkling 2 Small geometry and constants admitted for Metal EXL3 serving.
pub fn admitSmallConfig(c: *const config.Config, why: *config.Why) !void {
    if (c.quant != .exl3_kortexa_v1) {
        why.set("inkling2 Small EXL3 requires quantization_config format {s}", .{format_id});
        return error.UnsupportedQuant;
    }
    const t = c.text;
    if (t.loop != null) {
        why.set("inkling2 Small EXL3 expects a non-looped checkpoint (loop_config must be absent)", .{});
        return error.NotInkling2Small;
    }
    if (t.hidden != exl3_cfg.hidden or t.layers != exl3_cfg.layers or t.vocab != exl3_cfg.vocab or
        t.unpadded_vocab != exl3_cfg.vocabulary or t.window != exl3_cfg.local_window or
        t.d_rel != config.d_rel or t.rel_extent != exl3_cfg.local_window or
        (t.log_floor != 8192 and t.log_floor != 128000) or
        t.block != exl3_cfg.block_size or t.experts != exl3_cfg.experts or t.topk != exl3_cfg.topk or
        t.expert_inter != exl3_cfg.intermediate or t.max_context != exl3_cfg.maximum_context)
    {
        why.set("inkling2 Small EXL3 geometry mismatch: expected 48 layers, hidden 4608, 384 experts, inter 2304", .{});
        return error.UnsupportedInkling2Geometry;
    }
    for ([_]config.Heads{ t.local, t.global }) |h| {
        if (h.q != 48 or h.kv != 8 or h.dim != config.head_dim) {
            why.set("inkling2 Small EXL3 expects 48 query heads, 8 KV heads, head_dim 128", .{});
            return error.UnsupportedInkling2Geometry;
        }
    }
    if (t.eps != 1e-6 or t.log_alpha != 0.1 or t.qk_clip != 22.627416997969522 or
        t.rel_clip != 8 or t.route_scale != 8 or t.gate_cap != 4 or t.up_cap != 25 or
        t.logit_cap != 30)
    {
        why.set("inkling2 Small EXL3 constants differ from the published checkpoint", .{});
        return error.UnsupportedInkling2Constants;
    }
    _ = try exl3_cfg.layerPlan(c);
}

fn cleanRelativePath(path: []const u8) bool {
    if (path.len == 0 or path[0] == '/' or path[0] == '\\') return false;
    if (std.mem.indexOfAny(u8, path, "\\:\x00") != null) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return true;
}

fn jsonString(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn jsonU32(o: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = o.get(key) orelse return null;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return null;
    return @intCast(v.integer);
}

fn jsonU64(o: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = o.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return null;
    return @intCast(v.integer);
}

/// Parse and validate a kortexa-inkling2-exl3-v1 checkpoint.json manifest against `text`.
pub fn parseManifest(gpa: std.mem.Allocator, json: []const u8, text: *const config.Text, why: *config.Why) !Manifest {
    if (text.loop != null) {
        why.set("EXL3 pack manifest is refused for looped checkpoints", .{});
        return error.UnsupportedPack;
    }
    if (text.hidden % 16 != 0 or text.expert_inter % 16 != 0) {
        why.set("EXL3 pack requires hidden_size and intermediate_size divisible by 16", .{});
        return error.BadManifest;
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, a, json, .{}) catch {
        why.set("checkpoint.json is not valid JSON", .{});
        return error.BadManifest;
    };
    if (parsed.value != .object) {
        why.set("checkpoint.json root must be a JSON object", .{});
        return error.BadManifest;
    }
    const root = parsed.value.object;
    const fmt = jsonString(root, "format") orelse "";
    if (!std.mem.eql(u8, fmt, format_id)) {
        why.set("checkpoint.json format must be {s}", .{format_id});
        return error.UnsupportedFormat;
    }
    const status = jsonString(root, "status") orelse "";
    if (!std.mem.eql(u8, status, "complete")) {
        why.set("checkpoint.json status is not complete", .{});
        return error.IncompletePack;
    }
    const layers_val = root.get("layers") orelse {
        why.set("checkpoint.json has no layers object", .{});
        return error.BadManifest;
    };
    if (layers_val != .object) {
        why.set("checkpoint.json layers must be an object keyed by layer index", .{});
        return error.BadManifest;
    }
    const layers_obj = layers_val.object;
    if (layers_obj.count() != text.layers) {
        why.set("checkpoint.json layers count {d} does not match config layers {d}", .{ layers_obj.count(), text.layers });
        return error.BadManifest;
    }

    const out_layers = try a.alloc(LayerManifest, text.layers);
    const covered = try a.alloc(bool, text.experts);
    for (0..text.layers) |li| {
        var key_buf: [16]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{d}", .{li}) catch unreachable;
        const lv = layers_obj.get(key) orelse {
            why.set("checkpoint.json is missing layer {d}", .{li});
            return error.BadManifest;
        };
        if (lv != .object) {
            why.set("checkpoint.json layer {d} must be an object", .{li});
            return error.BadManifest;
        }
        const lo = lv.object;
        if (lo.get("status")) |sv| {
            if (sv != .string or !std.mem.eql(u8, sv.string, "complete")) {
                why.set("checkpoint.json layer {d} status is not complete", .{li});
                return error.IncompletePack;
            }
        }
        const batches_val = lo.get("batches") orelse {
            why.set("checkpoint.json layer {d} has no batches array", .{li});
            return error.BadManifest;
        };
        if (batches_val != .array or batches_val.array.items.len == 0) {
            why.set("checkpoint.json layer {d} batches must be a nonempty array", .{li});
            return error.BadManifest;
        }

        @memset(covered, false);
        const batches = try a.alloc(Batch, batches_val.array.items.len);
        for (batches_val.array.items, 0..) |bv, bi| {
            if (bv != .object) {
                why.set("checkpoint.json layer {d} batch {d} must be an object", .{ li, bi });
                return error.BadManifest;
            }
            const bo = bv.object;
            const start = jsonU32(bo, "start") orelse {
                why.set("checkpoint.json layer {d} batch {d} has invalid start", .{ li, bi });
                return error.BadManifest;
            };
            const end = jsonU32(bo, "end") orelse {
                why.set("checkpoint.json layer {d} batch {d} has invalid end", .{ li, bi });
                return error.BadManifest;
            };
            if (start >= end or end > text.experts) {
                why.set("checkpoint.json layer {d} batch {d} range {d}..{d} is out of bounds", .{ li, bi, start, end });
                return error.BadManifest;
            }
            const file = jsonString(bo, "file") orelse "";
            if (!cleanRelativePath(file)) {
                why.set("checkpoint.json layer {d} batch {d} has unsafe or empty file path", .{ li, bi });
                return error.UnsafePath;
            }
            const bytes = jsonU64(bo, "bytes") orelse 0;
            if (bytes == 0) {
                why.set("checkpoint.json layer {d} batch {d} must declare positive bytes", .{ li, bi });
                return error.BadManifest;
            }
            for (start..end) |e| {
                if (covered[e]) {
                    why.set("checkpoint.json layer {d} expert {d} is covered more than once", .{ li, e });
                    return error.DuplicateExpert;
                }
                covered[e] = true;
            }
            batches[bi] = .{ .start = start, .end = end, .file = try a.dupe(u8, file), .bytes = bytes };
        }
        for (covered, 0..) |ok, e| {
            if (!ok) {
                why.set("checkpoint.json layer {d} is missing expert {d}", .{ li, e });
                return error.MissingExpert;
            }
        }

        var k_by_expert: ?[]u8 = null;
        if (lo.get("K_by_expert")) |kv| {
            if (kv != .array or kv.array.items.len != text.experts) {
                why.set("checkpoint.json layer {d} K_by_expert length does not match experts", .{li});
                return error.BadManifest;
            }
            const ks = try a.alloc(u8, text.experts);
            for (kv.array.items, 0..) |item, e| {
                if (item != .integer or item.integer < min_bits or item.integer > max_bits) {
                    why.set("checkpoint.json layer {d} expert {d} declares unsupported bit width", .{ li, e });
                    return error.UnsupportedWidth;
                }
                ks[e] = @intCast(item.integer);
            }
            k_by_expert = ks;
        }
        out_layers[li] = .{ .layer = @intCast(li), .batches = batches, .k_by_expert = k_by_expert };
    }
    return .{ .arena = arena, .layers = out_layers };
}

fn fileExists(io: std.Io, path: []const u8) !bool {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    f.close(io);
    return true;
}

const OpenedHeader = struct {
    header: st.Header,
    header_len: usize,
    file_size: u64,
};

fn readSafetensorsHeader(a: std.mem.Allocator, io: std.Io, path: []const u8) !OpenedHeader {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const size = try f.length(io);
    const size_usize = std.math.cast(usize, size) orelse return error.BadSafetensors;
    var prefix: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &prefix, 0) != 8) return error.BadSafetensors;
    const hlen = try st.headerSize(std.mem.readInt(u64, &prefix, .little), size_usize);
    const json = try a.alloc(u8, hlen);
    if (try f.readPositionalAll(io, json, 8) != hlen) return error.BadSafetensors;
    const data_len = size_usize - 8 - hlen;
    return .{
        .header = try st.parseHeader(a, json, data_len),
        .header_len = hlen,
        .file_size = size,
    };
}

fn expectScale(hdr: st.Header, name: []const u8, dim: usize, why: *config.Why) !void {
    const e = hdr.get(name) orelse {
        why.set("expert shard is missing {s}", .{name});
        return error.MissingTensor;
    };
    if (e.dtype != .f16 or e.rank != 1 or e.shape[0] != dim) {
        why.set("expert shard scale {s} has unexpected dtype or shape", .{name});
        return error.WrongTensor;
    }
}

fn validateProjection(
    io: std.Io,
    file: std.Io.File,
    data_offset: u64,
    hdr: st.Header,
    expert: u32,
    comptime proj: []const u8,
    k: usize,
    n: usize,
    declared_bits: ?u8,
    why: *config.Why,
) !u8 {
    var nb: [64]u8 = undefined;
    const t_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".trellis", .{expert}) catch unreachable;
    const te = hdr.get(t_name) orelse {
        why.set("expert shard is missing {s}", .{t_name});
        return error.MissingTensor;
    };
    if (te.dtype != .i16 or te.rank != 3 or te.shape[0] != k / 16 or te.shape[1] != n / 16) {
        why.set("expert trellis {s} has unexpected dtype or shape", .{t_name});
        return error.WrongTensor;
    }
    if (te.shape[2] == 0 or te.shape[2] % 16 != 0) {
        why.set("expert trellis {s} last dim must be 16 * bits (whole widths 2..4)", .{t_name});
        return error.UnsupportedWidth;
    }
    const bits_usize = te.shape[2] / 16;
    if (bits_usize < min_bits or bits_usize > max_bits) {
        why.set("expert trellis {s} width {d} is outside 2..4 bits", .{ t_name, bits_usize });
        return error.UnsupportedWidth;
    }
    const bits: u8 = @intCast(bits_usize);
    const width: exl3.Width = .{ .halves = @as(u16, bits) * 2 };
    if (width.tileWords() != te.shape[2]) return error.UnsupportedWidth;
    if (declared_bits) |want| if (want != bits) {
        why.set("expert {d} {s} width {d} disagrees with manifest K_by_expert {d}", .{ expert, proj, bits, want });
        return error.WidthMismatch;
    };

    const su_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".suh", .{expert}) catch unreachable;
    try expectScale(hdr, su_name, k, why);
    const sv_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".svh", .{expert}) catch unreachable;
    try expectScale(hdr, sv_name, n, why);

    const m_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".mul1", .{expert}) catch unreachable;
    const me = hdr.get(m_name) orelse {
        why.set("expert shard is missing mul1 codebook marker {s}", .{m_name});
        return error.UnsupportedCodebook;
    };
    if (me.dtype != .i32 or me.rank != 0 or me.end - me.begin != 4) {
        why.set("expert marker {s} must be a scalar I32", .{m_name});
        return error.UnsupportedCodebook;
    }
    var raw: [4]u8 = undefined;
    if (try file.readPositionalAll(io, &raw, data_offset + me.begin) != 4) return error.BadSafetensors;
    const marker = std.mem.readInt(u32, &raw, .little);
    if (marker != exl3.Mul1Mul) {
        why.set("expert marker {s} has value 0x{x}, expected mul1 (0x{x})", .{ m_name, marker, exl3.Mul1Mul });
        return error.UnsupportedCodebook;
    }
    return bits;
}

fn validateExpertShard(
    a: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    batch: Batch,
    text: *const config.Text,
    k_by_expert: ?[]const u8,
    width_counts: *[max_bits + 1]u64,
    why: *config.Why,
) !u64 {
    const opened = try readSafetensorsHeader(a, io, path);
    if (opened.file_size != batch.bytes) {
        why.set("expert shard {s} size {d} does not match manifest bytes {d}", .{ batch.file, opened.file_size, batch.bytes });
        return error.ShardSizeMismatch;
    }
    const expected_count = @as(usize, batch.end - batch.start) * 8;
    if (opened.header.count() != expected_count) {
        why.set("expert shard {s} has {d} tensors, expected {d}", .{ batch.file, opened.header.count(), expected_count });
        return error.UnexpectedTensor;
    }
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const data_off: u64 = 8 + @as(u64, opened.header_len);
    const h: usize = text.hidden;
    const m: usize = text.expert_inter;
    for (batch.start..batch.end) |e_usize| {
        const e: u32 = @intCast(e_usize);
        const want = if (k_by_expert) |ks| ks[e] else null;
        const b13 = try validateProjection(io, f, data_off, opened.header, e, "w13", h, 2 * m, want, why);
        const b2 = try validateProjection(io, f, data_off, opened.header, e, "w2", m, h, want, why);
        width_counts[b13] += 1;
        width_counts[b2] += 1;
    }
    return opened.file_size;
}

/// Inspect and validate an on-disk kortexa-inkling2-exl3-v1 pack directory without loading weight data.
pub fn inspectDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: *config.Why) !Summary {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const single_manifest = try std.fs.path.join(a, &.{ dir, "experts", "checkpoint.json" });
    const split_manifest = try std.fs.path.join(a, &.{ dir, "checkpoint.json" });
    const has_single = try fileExists(io, single_manifest);
    const has_split = try fileExists(io, split_manifest);
    if (has_single and has_split) {
        why.set("pack directory has both checkpoint.json and experts/checkpoint.json", .{});
        return error.AmbiguousLayout;
    }
    if (!has_single and !has_split) {
        why.set("pack directory has neither checkpoint.json nor experts/checkpoint.json", .{});
        return error.MissingManifest;
    }
    const layout: Layout = if (has_single) .single_dense else .split_dense;
    const cfg_path = if (layout == .single_dense)
        try std.fs.path.join(a, &.{ dir, "config.json" })
    else
        try std.fs.path.join(a, &.{ dir, "auxiliary", "config.json" });
    const cfg_json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
    const parsed_cfg = try config.parse(a, cfg_json, why);
    if (parsed_cfg.quant != .exl3_kortexa_v1) {
        why.set("EXL3 pack config.json must declare quantization_config format {s}", .{format_id});
        return error.UnsupportedQuant;
    }
    if (parsed_cfg.text.loop != null) {
        why.set("EXL3 pack requires a non-looped Inkling 2 checkpoint", .{});
        return error.UnsupportedPack;
    }

    const manifest_path = if (layout == .single_dense) single_manifest else split_manifest;
    const manifest_json = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, a, .limited(64 << 20));
    var manifest = try parseManifest(gpa, manifest_json, &parsed_cfg.text, why);
    defer manifest.deinit();

    var dense_header: st.Header = .empty;
    var dense_bytes: u64 = 0;
    if (layout == .single_dense) {
        const dense_path = try std.fs.path.join(a, &.{ dir, "model-dense.safetensors" });
        const opened = try readSafetensorsHeader(a, io, dense_path);
        dense_header = opened.header;
        dense_bytes = opened.file_size;
    } else {
        const global_path = try std.fs.path.join(a, &.{ dir, "dense", "global.safetensors" });
        const g_open = try readSafetensorsHeader(a, io, global_path);
        try names.merge(a, &dense_header, g_open.header);
        dense_bytes += g_open.file_size;
        for (0..parsed_cfg.text.layers) |li| {
            var lb: [32]u8 = undefined;
            const fname = std.fmt.bufPrint(&lb, "layer-{d:0>2}.safetensors", .{li}) catch unreachable;
            const lpath = try std.fs.path.join(a, &.{ dir, "dense", fname });
            const l_open = try readSafetensorsHeader(a, io, lpath);
            try names.merge(a, &dense_header, l_open.header);
            dense_bytes += l_open.file_size;
        }
    }
    const dense_report = try names.check(a, &parsed_cfg, dense_header, .dense_text);
    if (!dense_report.clean()) {
        why.set("dense text weights failed validation: {d} missing, {d} wrong, {d} unknown", .{
            dense_report.missing.items.len,
            dense_report.wrong.items.len,
            dense_report.unknown.items.len,
        });
        return error.BadDenseWeights;
    }

    const expert_root = if (layout == .single_dense) try std.fs.path.join(a, &.{ dir, "experts" }) else dir;
    var summary: Summary = .{
        .layout = layout,
        .layers = parsed_cfg.text.layers,
        .experts = parsed_cfg.text.experts,
        .dense_tensors = dense_report.expected,
        .expert_shards = 0,
        .dense_bytes = dense_bytes,
    };
    for (manifest.layers) |lm| {
        for (lm.batches) |batch| {
            var shard_arena = std.heap.ArenaAllocator.init(gpa);
            defer shard_arena.deinit();
            const sa = shard_arena.allocator();
            const spath = try std.fs.path.join(sa, &.{ expert_root, batch.file });
            summary.expert_bytes += try validateExpertShard(sa, io, spath, batch, &parsed_cfg.text, lm.k_by_expert, &summary.width_counts, why);
            summary.expert_shards += 1;
        }
    }

    const mtp_path = if (layout == .single_dense)
        try std.fs.path.join(a, &.{ dir, "mtp.safetensors" })
    else
        try std.fs.path.join(a, &.{ dir, "auxiliary", "mtp.safetensors" });
    if (try fileExists(io, mtp_path)) {
        if (parsed_cfg.mtp == null) {
            why.set("mtp.safetensors is present but config.json has no mtp_config", .{});
            return error.UnexpectedMtp;
        }
        const m_open = try readSafetensorsHeader(a, io, mtp_path);
        const m_report = try names.check(a, &parsed_cfg, m_open.header, .mtp);
        if (!m_report.clean()) {
            why.set("mtp.safetensors failed validation: {d} missing, {d} wrong, {d} unknown", .{
                m_report.missing.items.len,
                m_report.wrong.items.len,
                m_report.unknown.items.len,
            });
            return error.BadMtpWeights;
        }
        summary.has_mtp = true;
        summary.mtp_bytes = m_open.file_size;
    }
    return summary;
}
