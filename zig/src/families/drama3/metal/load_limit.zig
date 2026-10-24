//! The one-Mac load ceiling for Drama 3: 70% of RAM, or DRAMA_LOAD_LIMIT_GB up to the GPU's working set.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("../config.zig");
const shards = @import("../shards.zig");

/// The row buffers, the codec and the frames the load holds beside the weights and the caches.
pub const margin: u64 = 64 << 20;

pub const Limit = struct {
    bytes: usize,
    default: usize, // 70% of RAM in GiB read as GB (179.2 at 256 GiB)
    asked: bool = false, // DRAMA_LOAD_LIMIT_GB set it
    capped: bool = false, // DRAMA_LOAD_LIMIT_GB asked for more than the GPU's recommended working set

    /// Where the limit came from, for the refusal.
    pub fn source(l: Limit) []const u8 {
        return if (l.asked) "DRAMA_LOAD_LIMIT_GB" else "70% of RAM";
    }
};

/// 70% of `ram`, or DRAMA_LOAD_LIMIT_GB's GB (`text`) up to the GPU's working set; text that isn't a number is refused.
pub fn limitFor(ram: u64, working_set: u64, text: ?[]const u8) error{BadLoadLimit}!Limit {
    const default: usize = @intFromFloat(@as(f64, @floatFromInt(ram)) / (1 << 30) * 0.7 * 1e9);
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return .{ .bytes = default, .default = default };
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
    const most: f64 = @floatFromInt(@min(ram, working_set));
    return .{ .bytes = @intFromFloat(@min(gb * 1e9, most)), .default = default, .asked = true, .capped = gb * 1e9 > most };
}

/// This Mac's limit (RAM, GPU working set, DRAMA_LOAD_LIMIT_GB); past the 70% default the log names the panic risk.
pub fn loadLimit(device: mtl.Device) error{BadLoadLimit}!Limit {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &mem, &len, null, 0) != 0) mem = 0; // RAM unknown: a zero limit refuses every load
    const text: ?[]const u8 = if (std.c.getenv("DRAMA_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const l = limitFor(mem, device.maxWorkingSet(), text) catch |err| {
        std.log.err("drama3: DRAMA_LOAD_LIMIT_GB={s} is not a load limit: give a number of GB above 0, or unset it for 70% of RAM", .{text.?});
        return err;
    };
    if (l.capped) std.log.info("drama3: DRAMA_LOAD_LIMIT_GB={s} passes this GPU's recommended working set; the limit is {d:.1} GB", .{ text.?, toGb(l.bytes) });
    if (l.bytes > l.default) std.log.warn("drama3: DRAMA_LOAD_LIMIT_GB lets this Mac load {d:.1} GB, past the default {d:.1} GB (70% of RAM): a load that wires most of the RAM can stall macOS until its watchdog restarts the Mac, so run nothing else large beside it", .{ toGb(l.bytes), toGb(l.default) });
    return l;
}

/// The caches a load holds: the KV cache of every full-attention layer and the tower's, and the linear state.
pub fn cacheBytes(c: cfg.Config, context: u32) u64 {
    const kvb: u64 = @as(u64, context) * cfg.kv_heads * cfg.head_dim * 2;
    var full: u64 = 0;
    var gdn: u64 = 0;
    for (c.full_attention) |f| {
        if (f) full += 1 else gdn += 1;
    }
    const kv = (full + cfg.tower_layers) * 2 * kvb;
    const state = gdn * (cfg.conv_channels * 3 * 2 + cfg.lin_v_heads * cfg.lin_dim * cfg.lin_dim * 4);
    return kv + state + margin;
}

/// The bytes a load needs: the checkpoint's tensors (read in place, the bf16 ones copied too) and the caches.
pub fn planBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, context: u32) !u64 {
    const c = try cfg.Config.load(gpa, io, dir);
    var sh = try shards.Shards.open(gpa, io, dir);
    defer sh.close(io);
    var resident: u64 = 0;
    for (sh.files) |f| for (f.names.values()) |e| {
        const bytes: u64 = e.end - e.begin;
        resident += bytes;
        if (e.dtype == .bf16) resident += bytes; // copied to the device beside the mapped pages
    };
    return resident + cacheBytes(c, context);
}

/// A plan past the limit is refused before a weight is read.
pub fn admitUnder(plan: u64, limit: Limit) error{OverMemoryLimit}!void {
    if (plan > limit.bytes) return error.OverMemoryLimit;
}

/// This Mac's ceiling and the load's plan: a load that does not fit is refused before a weight is read.
pub fn admit(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, context: u32) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    const limit = try loadLimit(device);
    const plan = try planBytes(gpa, io, dir, context);
    if (plan > limit.bytes) {
        std.log.err("drama3: {d:.1} GB of weights and caches pass this Mac's {d:.1} GB load limit ({s}); free memory, or set DRAMA_LOAD_LIMIT_GB (past 70% of RAM a load risks a watchdog panic)", .{ toGb(plan), toGb(limit.bytes), limit.source() });
        return error.OverMemoryLimit;
    }
}

fn toGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

test "the Drama 3 load limit: 70% of RAM, or DRAMA_LOAD_LIMIT_GB up to the GPU's working set" {
    const ram: u64 = 256 << 30;
    const ws: u64 = 239_000_000_000;
    const default = try limitFor(ram, ws, null);
    try std.testing.expectEqual(@as(usize, 179_200_000_000), default.bytes);
    try std.testing.expect(!default.asked and std.mem.eql(u8, "70% of RAM", default.source()));
    const asked = try limitFor(ram, ws, "10");
    try std.testing.expectEqual(@as(usize, 10_000_000_000), asked.bytes);
    try std.testing.expect(asked.asked and std.mem.eql(u8, "DRAMA_LOAD_LIMIT_GB", asked.source()));
    const capped = try limitFor(ram, ws, "1000");
    try std.testing.expectEqual(@as(usize, 239_000_000_000), capped.bytes);
    try std.testing.expect(capped.capped);
    try std.testing.expectError(error.BadLoadLimit, limitFor(ram, ws, "x"));
    try std.testing.expectError(error.BadLoadLimit, limitFor(ram, ws, "0"));
    try std.testing.expectError(error.BadLoadLimit, limitFor(ram, ws, "-1"));
}

test "the Drama 3 plan counts the checkpoint's tensors and its caches, and a plan past the limit is refused" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = config_text });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.embed_tokens.weight\":\"model-00001-of-00001.safetensors\"}}" });
    const header = "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    const shard = try std.testing.allocator.alloc(u8, 8 + header.len + 4);
    defer std.testing.allocator.free(shard);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    @memset(shard[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = shard });
    const plan = try planBytes(std.testing.allocator, std.testing.io, dir, 8192);
    const c = try cfg.Config.parse(std.testing.allocator, config_text);
    // the tensor's 4 bytes are counted twice (the mapped pages and the device copy) beside the caches
    try std.testing.expectEqual(cacheBytes(c, 8192) + 8, plan);
    try std.testing.expectError(error.OverMemoryLimit, admitUnder(plan, .{ .bytes = @intCast(plan - 1), .default = 1 }));
    try admitUnder(plan, .{ .bytes = @intCast(plan), .default = 1 });
}

const config_text =
    \\{"eos_token_id": 248046, "pad_token_id": 248044, "semantic_start_token_id": 248099,
    \\ "semantic_end_token_id": 252194, "audio_pad_token_id": 248076,
    \\ "text_config": {"hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 252288,
    \\  "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "num_experts": 256,
    \\  "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    \\  "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    \\  "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    \\  "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention",
    \\   "linear_attention","linear_attention","linear_attention","full_attention"]},
    \\ "audio_decoder_config": {"dim": 2048, "n_head": 16, "n_local_heads": 2, "head_dim": 256,
    \\  "intermediate_size": 6144, "num_codebooks": 10, "vocab_size": 4096, "shared_enrichment_layers": 4,
    \\  "vq_backbone_codebooks": 1, "rotary_interleaved": true}}
;
