//! Metadata-only upper bounds for transformed packs and their overlapping host construction storage.
const std = @import("std");
const add = std.math.add;
const mul = std.math.mul;
pub const conversion_scratch_copies = 2;
pub const Width = struct { bits: u8, group: u16 };

pub const Inventory = struct {
    raw: u64 = 0,
    source: u64 = 0,
    decode: u64 = 0,
    prompt: u64 = 0,
    tensors: u64 = 0,
    routed: u64 = 0,
    ple: u64 = 0,
    expert_parts: [9]u64 = @splat(0),

    /// Source pages, one retained output phase and operation-local fusion/regroup/gather/tile storage.
    pub fn building(p: Inventory) !u64 {
        return add(u64, p.source, try add(u64, @max(p.decode, p.prompt), try mul(u64, p.decode, conversion_scratch_copies)));
    }

    pub fn resident(p: Inventory) !u64 {
        return add(u64, p.decode, p.prompt);
    }

    /// Raw expert, embedding and PLE tensors already fit in the original checkpoint-byte charge.
    pub fn include(p: *Inventory, name: []const u8, e: anytype) !void {
        return p.includeWidth(name, e, .{ .bits = 2, .group = 128 });
    }

    pub fn includeWidth(p: *Inventory, name: []const u8, e: anytype, width: Width) !void {
        const bytes: u64 = e.end - e.begin;
        if (!packedTensor(name)) {
            if (rawTensor(name)) {
                p.raw = try add(u64, p.raw, bytes);
                if (expertPart(name)) |part| {
                    if (bytes == 0 or bytes % 512 != 0) return error.BadExpertTensor;
                    p.routed = try add(u64, p.routed, bytes);
                    p.expert_parts[part] = @max(p.expert_parts[part], bytes / 512);
                }
                if (std.mem.indexOf(u8, name, ".ngram_embedding.") != null)
                    p.ple = try add(u64, p.ple, bytes);
            }
            return;
        }
        const bound: u64 = switch (e.dtype) {
            .u32 => if (std.mem.endsWith(u8, name, ".weight")) try codeBytes(name, bytes, width) else bytes,
            .bf16 => try mul(u64, bytes, if (std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases")) @as(u64, width.group / 32) else @as(u64, 2)),
            else => bytes,
        };
        p.source = try add(u64, p.source, bytes);
        p.decode = try add(u64, p.decode, if (std.mem.startsWith(u8, name, "language_model.lm_head.")) try mul(u64, bound, 2) else bound);
        // Include an extra full head for every possible draft vocabulary, not just its selected rows.
        p.prompt = try add(u64, p.prompt, if (std.mem.startsWith(u8, name, "language_model.lm_head.")) try mul(u64, bound, 2) else bytes);
        p.tensors = try add(u64, p.tensors, 1);
    }
};

/// Stored routed components; shared experts and projection packs stay resident.
fn expertPart(name: []const u8) ?usize {
    const bases = [_][]const u8{ ".mlp.switch_mlp.gate_proj.", ".mlp.switch_mlp.up_proj.", ".mlp.switch_mlp.down_proj." };
    for (bases, 0..) |base, i| if (std.mem.indexOf(u8, name, base) != null) {
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, j|
            if (std.mem.endsWith(u8, name, suffix)) return i * 3 + j;
    };
    return null;
}

fn rawTensor(name: []const u8) bool {
    const payload = std.mem.endsWith(u8, name, ".weight") or std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases");
    return payload and (std.mem.indexOf(u8, name, ".ngram_embedding.") != null or
        std.mem.startsWith(u8, name, "language_model.model.embed_tokens.") or
        std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null or std.mem.indexOf(u8, name, ".mlp.shared_expert.") != null);
}

fn packedTensor(name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, "language_model.") and !std.mem.startsWith(u8, name, "mtp.")) return false;
    return std.mem.indexOf(u8, name, ".ngram_embedding.") == null and
        std.mem.indexOf(u8, name, ".embed_tokens.") == null and
        std.mem.indexOf(u8, name, ".mlp.switch_mlp.") == null and
        std.mem.indexOf(u8, name, ".mlp.shared_expert.") == null;
}

fn codeBytes(name: []const u8, bytes: u64, width: Width) !u64 {
    const bits = width.bits;
    if (bits < 2 or bits > 8 or bits == 7) return error.UnsupportedQuantization;
    if (std.mem.endsWith(u8, name, ".gate.weight") or std.mem.endsWith(u8, name, ".shared_expert_gate.weight"))
        return std.math.divCeil(u64, try mul(u64, bytes, 16), bits);
    if (std.mem.indexOf(u8, name, "hyper_connection") != null)
        return std.math.divCeil(u64, try mul(u64, bytes, 8), bits);
    return bytes;
}

/// Checkpoint maps are unpopulated; only their common validated headers are accessed, never tensor payloads.
pub fn read(comptime Checkpoint: type, comptime Mlx: type, a: std.mem.Allocator, io: std.Io, dir: []const u8) !Inventory {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    const cfg_path = try std.fs.path.join(alloc, &.{ dir, "config.json" });
    const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, alloc, .limited(1 << 26));
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, alloc, cfg_bytes, .{});
    if (cfg != .object) return error.InvalidFlashConfig;
    const root = cfg.object;
    const text = if (root.get("text_config")) |v| if (v == .object and v.object.count() > 0) v.object else root else root;
    const quant = quant: {
        for ([_]std.json.ObjectMap{ root, text }) |source| for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
            if (source.get(key)) |v| if (v == .object and v.object.count() > 0) break :quant v.object;
        };
        return error.MissingAffineConfig;
    };
    const widths = (try Mlx.detect(alloc, .{ .root = root, .quantization = quant })) orelse return error.UnsupportedQuantization;
    var out: Inventory = .{};
    for ([_][]const u8{ "language_model.", "mtp." }) |prefix| {
        var ck = try Checkpoint.openModelPrefix(a, io, dir, prefix);
        defer ck.close();
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(a);
        for (ck.files.items) |*file| {
            var it = file.names.iterator();
            while (it.next()) |item| {
                if ((try seen.getOrPut(a, item.key_ptr.*)).found_existing) return error.DuplicateTensor;
                const name = item.key_ptr.*;
                const stem = for ([_][]const u8{ ".weight", ".scales", ".biases" }) |suffix| {
                    if (std.mem.endsWith(u8, name, suffix)) break name[0 .. name.len - suffix.len];
                } else name;
                const entry = item.value_ptr.*;
                const q = if (packedTensor(name) and (entry.dtype == .u32 or std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases"))) try widths.width(stem) else widths.global;
                try out.includeWidth(name, entry, .{ .bits = q.bits, .group = q.group });
            }
        }
    }
    return out;
}

test "Flash Next memory: transformed weights and conversion overlap are not hidden in the runtime reserve" {
    const Entry = struct { dtype: enum { u32, bf16, f32 }, begin: usize = 0, end: usize };
    var p: Inventory = .{};
    try p.include("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight", Entry{ .dtype = .u32, .end = 1 << 30 });
    try p.include("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", Entry{ .dtype = .u32, .end = 1 << 30 });
    try std.testing.expectEqual(@as(u64, 0), p.source);
    try p.include("language_model.model.layers.0.self_attn.q_proj.weight", Entry{ .dtype = .u32, .end = 100 });
    try p.include("language_model.model.layers.0.self_attn.q_proj.scales", Entry{ .dtype = .bf16, .end = 20 });
    try p.include("language_model.model.layers.0.mlp.gate.weight", Entry{ .dtype = .u32, .end = 10 });
    try std.testing.expectEqual(@as(u64, 260), p.decode);
    try std.testing.expectEqual(@as(u64, 130), p.prompt);
    try std.testing.expectEqual(@as(u64, 910), try p.building());
    try std.testing.expectEqual(@as(u64, 390), try p.resident());
    var overflow: Inventory = .{ .decode = std.math.maxInt(u64) };
    try std.testing.expectError(error.Overflow, overflow.building());
}

test "Flash Next memory: declared code widths and groups price each transformed tensor" {
    const Entry = struct { dtype: enum { u32, bf16, f32 }, begin: usize = 0, end: usize };
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }, [_]u64{ 960, 640, 480, 384, 320, 240 }) |bits, router| {
        var native: Inventory = .{};
        try native.includeWidth("language_model.model.layers.0.self_attn.q_proj.weight", Entry{ .dtype = .u32, .end = 120 }, .{ .bits = bits, .group = 32 });
        try std.testing.expectEqual(@as(u64, 120), native.decode);
        var gate: Inventory = .{};
        try gate.includeWidth("language_model.model.layers.0.mlp.gate.weight", Entry{ .dtype = .u32, .end = 120 }, .{ .bits = bits, .group = 32 });
        try std.testing.expectEqual(router, gate.decode);
    }
    for ([_]u16{ 32, 64, 128 }, [_]u64{ 20, 40, 80 }) |group, bytes| {
        var scale: Inventory = .{};
        try scale.includeWidth("language_model.model.layers.0.self_attn.q_proj.scales", Entry{ .dtype = .bf16, .end = 20 }, .{ .bits = 4, .group = group });
        try std.testing.expectEqual(bytes, scale.decode);
    }
}
