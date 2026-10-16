//! nvidia/GLM-5.3-Flash-NVFP4's whole header: every linear is NVFP4 or BF16 as nvfp4.dims reads it.

const std = @import("std");
const nvfp4 = @import("nvfp4.zig");
const DType = @import("safetensors.zig").DType;

const Meta = nvfp4.Meta;

/// name -> [dtype, shape] from the gzip'd fixture, kept in `arena`.
fn headers(arena: std.mem.Allocator) !std.json.ObjectMap {
    var in: std.Io.Reader = .fixed(@embedFile("fixtures/glm53_flash_nvfp4/headers.json.gz"));
    var gz: std.compress.flate.Decompress = .init(&in, .gzip, &.{});
    var out: std.Io.Writer.Allocating = .init(arena);
    _ = try gz.reader.streamRemaining(&out.writer);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, out.written(), .{});
    return v.object;
}

fn meta(arena: std.mem.Allocator, h: std.json.ObjectMap, name: []const u8) !?Meta {
    const v = h.get(name) orelse return null;
    const pair = v.array.items;
    const dims = pair[1].array.items;
    const shape = try arena.alloc(usize, dims.len);
    for (dims, shape) |d, *s| s.* = @intCast(d.integer);
    return .{ .dtype = DType.parse(pair[0].string) orelse return error.UnsupportedDType, .shape = shape };
}

fn at(arena: std.mem.Allocator, h: std.json.ObjectMap, module: []const u8, suffix: []const u8) !?Meta {
    return meta(arena, h, try std.fmt.allocPrint(arena, "{s}.{s}", .{ module, suffix }));
}

test "every NVFP4 linear of the checkpoint reads as ModelOpt stores it, and the MTP layer stays BF16" {
    var state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    const h = try headers(a);
    try std.testing.expectEqual(@as(usize, 147661), h.count());
    var quantized: usize = 0;
    var experts: usize = 0;
    var it = h.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        if (std.mem.endsWith(u8, name, ".weight_scale")) {
            // no scale without its codes
            try std.testing.expect((try at(a, h, name[0 .. name.len - ".weight_scale".len], "weight")) != null);
            continue;
        }
        if (!std.mem.endsWith(u8, name, ".weight")) continue;
        const module = name[0 .. name.len - ".weight".len];
        const scale = try at(a, h, module, "weight_scale") orelse continue;
        const scale2 = (try at(a, h, module, "weight_scale_2")).?;
        const d = try nvfp4.dims((try meta(a, h, name)).?, scale, scale2, try at(a, h, module, "input_scale"));
        quantized += 1;
        try std.testing.expect(std.mem.indexOf(u8, module, ".mlp.") != null);
        try std.testing.expect(std.mem.indexOf(u8, module, ".layers.45.") == null);
        const down = std.mem.endsWith(u8, module, "down_proj");
        if (std.mem.indexOf(u8, module, ".experts.") != null) {
            experts += 1;
            const want: nvfp4.Dims = if (down) .{ .n = 4096, .k = 2048 } else .{ .n = 2048, .k = 4096 };
            try std.testing.expectEqual(want, d);
        } else {
            const want: nvfp4.Dims = if (down) .{ .n = 4096, .k = 12288 } else .{ .n = 12288, .k = 4096 };
            try std.testing.expectEqual(want, d);
        }
    }
    try std.testing.expectEqual(@as(usize, 42 * 288 * 3 + 3 * 3), quantized);
    try std.testing.expectEqual(@as(usize, 42 * 288 * 3), experts);
    // the MTP layer's routed experts are BF16 as stored (a family may re-pack them for drafting)
    const mtp = (try meta(a, h, "model.language_model.layers.45.mlp.experts.0.gate_proj.weight")).?;
    try std.testing.expectEqual(DType.bf16, mtp.dtype);
    const mtp_scale = "model.language_model.layers.45.mlp.experts.0.gate_proj.weight_scale";
    try std.testing.expect((try meta(a, h, mtp_scale)) == null);
}
