//! GLM-5.3-Flash's checkpoint index in either layout (the original or mlx-lm's), each tensor by its original name.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const load_fix = @import("weights_fix.zig");

const Src = load_fix.Src;
const lm = "model.language_model.";

/// A stored tensor's name in the original layout, or null for another tower's (the vision tower is never read).
pub fn original(buf: []u8, name: []const u8, mtp_layer: u32) error{NoSpaceLeft}!?[]const u8 {
    if (std.mem.startsWith(u8, name, lm) or std.mem.startsWith(u8, name, "lm_head.")) return name;
    const mlx = "language_model.";
    if (!std.mem.startsWith(u8, name, mlx)) return null;
    var short = name[mlx.len..];
    if (std.mem.startsWith(u8, short, "lm_head.")) return short;
    var num: [16]u8 = undefined;
    var head: []const u8 = "";
    if (std.mem.startsWith(u8, short, "model.")) {
        short = short["model.".len..];
    } else if (std.mem.startsWith(u8, short, "mtp.0.")) { // the MTP block as the layer after the backbone's
        short = short["mtp.0.".len..];
        if (std.mem.startsWith(u8, short, "block.")) short = short["block.".len..];
        if (std.mem.eql(u8, short, "norm.weight")) short = "shared_head.norm.weight";
        head = try std.fmt.bufPrint(&num, "layers.{d}.", .{mtp_layer});
    } else return null;
    inline for (.{ "attn", "ffn" }) |site| inline for (.{ "fn", "base", "scale" }) |part| {
        const tail = "." ++ site ++ "_hc." ++ part;
        if (std.mem.endsWith(u8, short, tail)) return try std.fmt.bufPrint(buf, lm ++ "{s}{s}.hc_" ++ site ++ "_" ++ part, .{ head, short[0 .. short.len - tail.len] });
    };
    const gate = ".self_attn.forget_gate.";
    if (std.mem.indexOf(u8, short, gate)) |at| return try std.fmt.bufPrint(buf, lm ++ "{s}{s}.self_attn.{s}", .{ head, short[0..at], short[at + gate.len ..] });
    return try std.fmt.bufPrint(buf, lm ++ "{s}{s}", .{ head, short });
}

/// The index and every shard's header: each language-model tensor's shard and byte range, by its original name.
pub fn read(gpa: std.mem.Allocator, arena: std.mem.Allocator, dir: []const u8, mtp_layer: u32, shards: *std.ArrayList([:0]const u8), names: *std.StringHashMapUnmanaged(Src)) !void {
    const path = try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{dir}, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    if (f.size > st.index.max_bytes) return error.BadIndex;
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = (try st.index.weightMap(doc)).iterator();
    while (it.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    for (files.keys(), 0..) |name, si| {
        const full = try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ dir, name }, 0);
        try shards.append(gpa, full);
        const fd = platform.c.open(full, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = platform.c.close(fd);
        var head: [8]u8 = undefined;
        try load_fix.readAll(fd, &head, 0);
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (end < 8) return error.BadSafetensors;
        const size = std.math.cast(usize, end) orelse return error.BadSafetensors;
        const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), size);
        const header = try arena.alloc(u8, hlen);
        try load_fix.readAll(fd, header, 8);
        const data = 8 + hlen;
        const entries = try st.parseHeaderPrefixes(arena, header, size - data, &.{ lm, "language_model.", "lm_head." });
        for (entries.keys(), entries.values()) |name_in_file, entry| {
            var buf: [240]u8 = undefined;
            const tensor = (try original(&buf, name_in_file, mtp_layer)) orelse continue;
            const got = try names.getOrPut(arena, try arena.dupe(u8, tensor));
            if (got.found_existing) { // one tensor stored under both layouts' names, or twice
                @import("quant.zig").say("glm: {s} is stored twice in the checkpoint", .{tensor});
                return error.BadSafetensors;
            }
            got.value_ptr.* = .{
                .shard = @intCast(si),
                .off = data + entry.begin,
                .len = entry.end - entry.begin,
                .dtype = entry.dtype,
                .shape = entry.shape,
                .rank = entry.rank,
            };
        }
    }
}

test "mlx-lm's names map onto the original layout's, the MTP block onto layer 45" {
    var buf: [240]u8 = undefined;
    const cases = [_][2][]const u8{
        .{ "language_model.model.embed_tokens.weight", "model.language_model.embed_tokens.weight" },
        .{ "language_model.lm_head.scales", "lm_head.scales" },
        .{ "lm_head.weight", "lm_head.weight" },
        .{ "model.language_model.layers.3.self_attn.kv_b_proj.weight", "model.language_model.layers.3.self_attn.kv_b_proj.weight" },
        .{ "language_model.model.layers.7.attn_hc.fn", "model.language_model.layers.7.hc_attn_fn" },
        .{ "language_model.model.layers.7.ffn_hc.scale", "model.language_model.layers.7.hc_ffn_scale" },
        .{ "language_model.model.layers.3.mlp.switch_mlp.up_proj.biases", "model.language_model.layers.3.mlp.switch_mlp.up_proj.biases" },
        .{ "language_model.model.layers.3.self_attn.embed_q.weight", "model.language_model.layers.3.self_attn.embed_q.weight" },
        .{ "language_model.model.layers.0.self_attn.forget_gate.A_log", "model.language_model.layers.0.self_attn.A_log" },
        .{ "language_model.mtp.0.block.self_attn.o_proj.weight", "model.language_model.layers.45.self_attn.o_proj.weight" },
        .{ "language_model.mtp.0.eh_proj.weight", "model.language_model.layers.45.eh_proj.weight" },
        .{ "language_model.mtp.0.enorm.weight", "model.language_model.layers.45.enorm.weight" },
        .{ "language_model.mtp.0.norm.weight", "model.language_model.layers.45.shared_head.norm.weight" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], (try original(&buf, c[0], 45)).?);
    for ([_][]const u8{ "vision_tower.blocks.0.attn.qkv.weight", "language_model.mtp.1.eh_proj.weight", "model.visual.x" }) |skip|
        try std.testing.expect((try original(&buf, skip, 45)) == null);
}
