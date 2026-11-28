//! Metadata fixtures exercise admission without allocating tensor payloads or a GPU device.
const std = @import("std");
const w = @import("weights.zig");
const c = @import("config.zig");
const st = @import("safetensors");
const layout = @import("vision_layout.zig");

fn config(bits: u8, group: u16) c.Config {
    return .{ .arena = std.heap.ArenaAllocator.init(std.testing.allocator), .kind = .text, .context = 262144, .quant = .{ .global = .{ .bits = bits, .group = group } } };
}

fn header(a: std.mem.Allocator, cfg: *const c.Config, namespace: w.Namespace) !st.Header {
    var h: st.Header = .empty;
    var offset: usize = 0;
    for (try w.schema(a, cfg, namespace)) |e| {
        var bytes: usize = e.dtype.size();
        for (e.shape[0..e.rank]) |d| bytes *= d;
        try h.put(a, e.name, .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .begin = offset, .end = offset + bytes });
        offset += bytes;
    }
    return h;
}

test "official text4 tensor count and bytes; both namespaces and every target bit width" {
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]w.Namespace{ .bare, .wrapped }) |namespace| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(bits, 64);
        defer cfg.deinit();
        const h = try header(arena.allocator(), &cfg, namespace);
        const m = try w.validate(std.testing.allocator, &cfg, &h);
        try std.testing.expectEqual(@as(usize, 594), m.tensors);
        try std.testing.expectEqual(namespace, m.namespace);
        if (bits == 4) try std.testing.expectEqual(@as(usize, 973008384), m.bytes);
    };
}

test "VL text uses mixed8 embedding with group32 without admitting its image tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cfg = config(4, 32);
    defer cfg.deinit();
    cfg.kind = .vision_language;
    try cfg.quant.modules.put(cfg.arena.allocator(), "embed_tokens", .{ .bits = 8, .group = 32 });
    var h = try header(a, &cfg, .wrapped);
    const embed = h.get("language_model.model.embed_tokens.weight").?;
    try std.testing.expectEqual(@as(usize, 512), embed.shape[1]);
    const gate = h.get("language_model.model.layers.2.self_attn.gate_proj.scales").?;
    try std.testing.expectEqualSlices(usize, &.{ 16, 64 }, gate.shape[0..2]);
    try h.put(a, "vision_tower.unqualified.weight", .{ .dtype = .bf16, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 0, .end = 2 });
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 594), m.tensors);
    try std.testing.expectEqual(@as(usize, 1212106240), m.bytes);
}

test "missing head gate, wrong packed width and malformed convolution are rejected" {
    const keys = [_][]const u8{
        "language_model.model.layers.2.self_attn.gate_proj.weight",
        "language_model.model.layers.0.conv.in_proj.weight",
        "language_model.model.layers.0.conv.conv.weight",
        "language_model.model.layers.0.post_operator_norm.weight",
    };
    for (keys, 0..) |key, i| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        if (i == 0) {
            _ = h.swapRemove(key);
            try std.testing.expectError(error.MissingTensor, w.validate(std.testing.allocator, &cfg, &h));
        } else {
            h.getPtr(key).?.shape[if (i == 3) 0 else 1] += 1;
            try std.testing.expectError(error.TensorShape, w.validate(std.testing.allocator, &cfg, &h));
        }
    }
}

test "bad dtype, byte range, namespace and unexpected text tensors are rejected" {
    for (0..4) |mode| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        const key = "language_model.model.embed_tokens.weight";
        switch (mode) {
            0 => h.getPtr(key).?.dtype = .bf16,
            1 => h.getPtr(key).?.end -= 1,
            2 => try h.put(arena.allocator(), "model.embed_tokens.weight", h.get(key).?),
            3 => try h.put(arena.allocator(), "language_model.lm_head.weight", h.get(key).?),
            else => unreachable,
        }
        const err = switch (mode) {
            0 => error.TensorDType,
            1 => error.TensorBytes,
            2 => error.MixedNamespaces,
            3 => error.UnexpectedTensor,
            else => unreachable,
        };
        try std.testing.expectError(err, w.validate(std.testing.allocator, &cfg, &h));
    }
}

test "module names of both namespaces are names of the schema" {
    for ([_]w.Namespace{ .bare, .wrapped }) |namespace| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var names: std.StringHashMapUnmanaged(void) = .empty;
        for (try w.schema(arena.allocator(), &cfg, namespace)) |e| try names.put(arena.allocator(), e.name, {});
        var buf: [128]u8 = undefined;
        var full: [160]u8 = undefined;
        for (std.enums.values(w.Module)) |m| {
            const layer: ?usize = switch (m) {
                .embed_tokens, .embedding_norm => null,
                .conv_in, .conv_out, .conv => 0,
                .q_proj, .k_proj, .v_proj, .o_proj, .gate_proj, .q_norm, .k_norm => 2,
                else => 23,
            };
            const name = try w.moduleName(&buf, namespace, layer, m);
            const affine = !std.mem.endsWith(u8, name, ".weight");
            const tensor = if (affine) try std.fmt.bufPrint(&full, "{s}.scales", .{name}) else name;
            try std.testing.expect(names.contains(tensor));
        }
        var e: [64]u8 = undefined;
        try std.testing.expect(names.contains(try w.embedName(&e, namespace)));
        try std.testing.expectEqual(namespace, try w.detect(namespace == .bare, namespace == .wrapped));
    }
    try std.testing.expectError(error.MixedNamespaces, w.detect(true, true));
}

test "the ModelOpt NVFP4 export: 502 bare tensors, FP4 FFN with its scales, everything else bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cfg = try c.parse(std.testing.allocator, @import("config_test.zig").nvfp4);
    defer cfg.deinit();
    const h = try header(arena.allocator(), &cfg, .bare);
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 502), m.tensors);
    try std.testing.expectEqual(@as(usize, 1829689408), m.bytes);
    const codes = h.get("model.layers.0.feed_forward.w2.weight").?;
    try std.testing.expect(codes.dtype == .u8 and codes.shape[0] == 2048 and codes.shape[1] == 3840);
    try std.testing.expectEqual(@as(u8, 0), h.get("model.layers.0.feed_forward.w2.weight_scale_2").?.rank);
    try std.testing.expect(h.get("model.layers.2.self_attn.gate_proj.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.embed_tokens.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.layers.0.conv.in_proj.scales") == null);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1, 7 }, h.get("model.layers.0.conv.conv.weight").?.shape[0..3]);
}

test "the block FP8 export: 402 bare tensors, FP8 FFN and q/k/v/o with block scales, convs and gates bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cfg = try c.parse(std.testing.allocator, @import("config_fp8_test.zig").fp8);
    defer cfg.deinit();
    const h = try header(arena.allocator(), &cfg, .bare);
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 402), m.tensors);
    try std.testing.expectEqual(@as(usize, 2209949952), m.bytes);
    const down = h.get("model.layers.0.feed_forward.w2.weight_scale").?;
    try std.testing.expect(down.dtype == .bf16 and down.shape[0] == 16 and down.shape[1] == 60);
    try std.testing.expect(h.get("model.layers.2.self_attn.k_proj.weight").?.dtype == .f8_e4m3);
    try std.testing.expect(h.get("model.layers.2.self_attn.gate_proj.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.layers.0.conv.in_proj.weight").?.dtype == .bf16);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1, 7 }, h.get("model.layers.0.conv.conv.weight").?.shape[0..3]);
}

/// One header entry at `offset` into `json`; returns the offset past its bytes.
fn entry(a: std.mem.Allocator, json: *std.ArrayList(u8), name: []const u8, dtype: st.DType, shape: []const usize, offset: usize) !usize {
    var bytes: usize = dtype.size();
    for (shape) |d| bytes *= d;
    var tag: [16]u8 = undefined;
    try json.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, std.ascii.upperString(&tag, @tagName(dtype)) });
    for (shape, 0..) |d, i| try json.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
    try json.print(a, "],\"data_offsets\":[{d},{d}]}},", .{ offset, offset + bytes });
    return offset + bytes;
}

/// A VL shard's header: its text decoder in `namespace`, then the official tower and projector; returns data bytes.
fn vlShard(a: std.mem.Allocator, cfg: *const c.Config, namespace: w.Namespace, json: *std.ArrayList(u8)) !usize {
    try json.append(a, '{');
    var offset: usize = 0;
    for (try w.schema(a, cfg, namespace)) |e| offset = try entry(a, json, e.name, e.dtype, e.shape[0..e.rank], offset);
    var lines = std.mem.splitScalar(u8, @embedFile("fixtures/vision_tensors.tsv"), '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const name = fields.next().?;
        try std.testing.expectEqualStrings("BF16", fields.next().?);
        var shape: [st.max_rank]usize = undefined;
        var rank: usize = 0;
        var dims = std.mem.splitScalar(u8, fields.next().?, ',');
        while (dims.next()) |d| : (rank += 1) shape[rank] = try std.fmt.parseInt(usize, d, 10);
        offset = try entry(a, json, name, .bf16, shape[0..rank], offset);
    }
    json.items[json.items.len - 1] = '}';
    return offset;
}

/// The tensors `Model.load` packs from that shard: each namespace `w.packs` names, or the whole file.
fn loaded(a: std.mem.Allocator, json: []const u8, len: usize, kind: c.Kind, tower: bool) !st.Header {
    const prefixes = w.packs(kind, tower) orelse return st.parseHeaderPrefixes(a, json, len, &.{});
    var out: st.Header = .empty;
    for (prefixes) |p| {
        const part = try st.parseHeaderPrefixes(a, json, len, &.{p});
        for (part.keys(), part.values()) |k, v| try out.put(a, k, v);
    }
    return out;
}

/// The packed tensors as the tower's layout reads them; each ref is the tensor's offset.
const Packed = struct {
    header: *const st.Header,
    len: usize,

    const Names = struct {
        keys: []const []const u8,
        at: usize = 0,

        pub fn next(self: *Names) ?[]const u8 {
            if (self.at == self.keys.len) return null;
            defer self.at += 1;
            return self.keys[self.at];
        }
    };

    pub fn find(self: Packed, name: []const u8) !?layout.Found(usize) {
        const e = self.header.get(name) orelse return null;
        return .{ .ref = e.begin, .bf16 = e.dtype == .bf16, .rank = e.rank, .shape = e.shape, .offset = e.begin, .bytes = e.end - e.begin, .storage = self.len };
    }

    pub fn keys(self: Packed) Names {
        return .{ .keys = self.header.keys() };
    }
};

test "an LFM3-VL load packs its tower and projector with --vision, its text alone without; a text load packs all" {
    for ([_]w.Namespace{ .bare, .wrapped }) |namespace| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var cfg = config(4, 32);
        defer cfg.deinit();
        cfg.kind = .vision_language;
        try cfg.quant.modules.put(cfg.arena.allocator(), "embed_tokens", .{ .bits = 8, .group = 32 });
        var json: std.ArrayList(u8) = .empty;
        const len = try vlShard(a, &cfg, namespace, &json);
        const all = try st.parseHeaderPrefixes(a, json.items, len, &.{});
        const lean = try loaded(a, json.items, len, .vision_language, false);
        const vision = try loaded(a, json.items, len, .vision_language, true);
        // without --vision: the text decoder alone; the tower and projector stay on disk
        try std.testing.expectEqual(@as(usize, 594), lean.count());
        try std.testing.expectEqual(@as(usize, 594), (try w.validate(std.testing.allocator, &cfg, &lean)).tensors);
        const none = layout.load(usize, std.testing.allocator, Packed{ .header = &lean, .len = len });
        try std.testing.expectError(error.MissingTensor, none);
        // with --vision: the whole shard, so the tower's 138 tensors load beside the same text
        _ = try layout.load(usize, std.testing.allocator, Packed{ .header = &vision, .len = len });
        try std.testing.expectEqual(@as(usize, 594 + layout.tensor_count), all.count());
        try std.testing.expectEqual(all.count(), vision.count());
        try std.testing.expectEqual(@as(usize, 594), (try w.validate(std.testing.allocator, &cfg, &vision)).tensors);
    }
    const text: []const []const u8 = &.{ "language_model.", "model." };
    try std.testing.expectEqualDeep(text, w.packs(.vision_language, false).?);
    try std.testing.expect(w.packs(.text, false) == null and w.packs(.text, true) == null);
}
