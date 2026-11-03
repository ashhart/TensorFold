//! Disposable mixed BF16/group-64 checkpoints exercise the admitted learner without any real model or device.
const std = @import("std");
const core = @import("core");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const cfg = @import("config.zig");
const a = std.testing.allocator;
pub const E = @import("engine.zig").Engine(n.G);
pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    b: *n.B,
    e: *E,
    path: []const u8,
    pub fn open() !Fixture {
        var arena: std.heap.ArenaAllocator = .init(a);
        errdefer arena.deinit();
        const mem = arena.allocator();
        var doc = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/native-dense.json"), .{});
        defer doc.deinit();
        const dm = doc.arena.allocator();
        try doc.value.object.put(dm, "ple_layer_ids", .{ .array = std.array_list.Managed(std.json.Value).init(dm) });
        try doc.value.object.put(dm, "num_hidden_layers", .{ .integer = 3 });
        try doc.value.object.getPtr("layer_types").?.array.append(.{ .string = "linear_attention" });
        var p: cfg.Problem = .{};
        const plain = try std.json.Stringify.valueAlloc(mem, doc.value, .{});
        var c = try cfg.parse(a, plain, &p);
        defer c.deinit();
        const Sink = struct {
            a: std.mem.Allocator,
            quant: std.json.ObjectMap = .empty,
            pub fn take(s: *@This(), name: []const u8, w: @import("weights.zig").Want) !void {
                if (w.role != .matrix and w.role != .embedding) return;
                try s.quant.put(s.a, try s.a.dupe(u8, name), .{ .bool = std.mem.indexOf(u8, name, ".mlp.") != null });
            }
        };
        var sink: Sink = .{ .a = dm };
        try sink.quant.put(dm, "bits", .{ .integer = 4 });
        try sink.quant.put(dm, "group_size", .{ .integer = 64 });
        try @import("weights.zig").walk(&c, .causal, &sink, &p);
        try doc.value.object.put(dm, "quantization", .{ .object = sink.quant });
        const text = try std.json.Stringify.valueAlloc(mem, doc.value, .{});
        var final = try cfg.parse(a, text, &p);
        defer final.deinit();
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = text });
        try @import("reader_fixture.zig").writeNonzero(tmp, mem, &final);
        const image = try tmp.dir.readFileAlloc(std.testing.io, "model.safetensors", mem, .unlimited);
        const hn = std.mem.readInt(u64, image[0..8], .little);
        const h = try core.safetensors.parseHeader(mem, image[8..][0..hn], image.len - 8 - hn);
        for (h.keys(), h.values(), 0..) |name, entry, ordinal| {
            if (std.mem.indexOf(u8, name, ".mlp.") == null) continue;
            const bytes = image[8 + hn + entry.begin ..][0 .. entry.end - entry.begin];
            if (std.mem.endsWith(u8, name, ".weight")) {
                for (bytes, 0..) |*v, i| v.* = @truncate((i * 37 + ordinal * 13) ^ (i >> 3));
            } else {
                const value: f32 = if (std.mem.endsWith(u8, name, ".scales")) 0.00390625 else -0.03125;
                for (0..bytes.len / 2) |i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], core.quant.convert.bf16(value), .little);
            }
        }
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = image });
        const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        const b = try mem.create(n.B);
        b.* = .{};
        const e = try E.open(a, std.testing.io, path, b, .{ .capacity = 32, .window = 16, .streams = 2, .budget = std.math.maxInt(u64), .learning = true }, &p, n.Loader{});
        return .{ .arena = arena, .tmp = tmp, .b = b, .e = e, .path = path };
    }
    pub fn deinit(f: *Fixture) void {
        f.e.close() catch unreachable;
        std.testing.expectEqual(@as(usize, 0), f.b.held) catch unreachable;
        f.tmp.cleanup();
        f.arena.deinit();
    }
};
