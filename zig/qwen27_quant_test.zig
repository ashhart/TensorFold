//! Actual projection cache construction and encoding at a recording Metal boundary.
const std = @import("std");
const mtl = @import("metal");
const quant = @import("src/families/qwen27/quant_gpu.zig");
const projection = @import("src/families/qwen27/projection.zig");
pub const lane_projection = @import("src/core/lane_projection.zig");
pub const row_projection = @import("src/core/row_projection.zig");
pub const checkpoint_metal = @import("src/core/checkpoint_metal.zig");
pub const gpu_profile = @import("src/core/gpu_profile.zig");
const a = std.testing.allocator;
const t = std.testing;

fn matrix(r: *mtl.Record, n: u32) projection.Linear {
    return .{ .words = .{ .buffer = r.virtual() }, .pairs = .{ .buffer = r.virtual() }, .n = n, .k = 5120, .tile = 32, .slices = 1 };
}

fn launch(k: *quant.Kernels, r: *mtl.Record, p: projection.Linear, rows: u32) !void {
    const ref = quant.Ref{ .buffer = r.virtual() };
    try k.quant(r.encoder(), p, ref, null, ref, ref, rows);
}

test "qwen27 quant: prepared head is the first launch entry after short and long prompts" {
    for ([_]bool{ false, true }) |long_prompt| for ([_]u32{ 128, 129, 2048 }) |prompt_rows| {
        var r: mtl.Record = .{};
        {
            var k = quant.Kernels{ .allocator = a, .device = r.device(), .long_prompt = long_prompt };
            defer k.deinit();
            const head = matrix(&r, @import("src/families/qwen27/dflash/runtime_head.zig").width);
            try k.prepareDecode(head);
            try t.expectEqual(@as(u32, 1), k.lanes.count());
            var entries = k.lanes.iterator();
            const first = entries.next().?;
            const prepared = first.key_ptr.*;
            const pipe = first.value_ptr.pipe.id;
            try t.expectEqualDeep(@TypeOf(prepared){ .kind = .projection, .n = 98592, .k = 5120, .slices = 1 }, prepared);
            try t.expectEqual(@as(u32, lane_projection.reg_rows), k.lanes.get(prepared).?.max_rows);
            try t.expectEqual(@as(usize, 1), r.libraries);
            try t.expectEqual(@as(usize, 2), r.pipelines);
            // Intervening target prompt traffic may build long entries, but cannot rename the decoded head.
            k.prompt = true;
            var body = matrix(&r, 34816);
            body.slices = 2;
            try launch(&k, &r, body, prompt_rows);
            k.prompt = false;
            const libraries = r.libraries;
            const pipelines = r.pipelines;
            for ([_]u32{ 1, 31, 128 }) |rows| {
                r.count = 0;
                try launch(&k, &r, head, rows);
                try t.expectEqual(@as(usize, 1), r.count);
                try t.expectEqual(pipe, r.launches[0].pipeline);
                try t.expectEqual(rows, r.launches[0].rows);
                try t.expect(k.lanes.contains(prepared));
                try t.expectEqual(@as(u32, 2), k.lanes.count());
                try t.expectEqual(libraries, r.libraries);
                try t.expectEqual(pipelines, r.pipelines);
            }
            try k.prepareDecode(head);
            try t.expectEqual(libraries, r.libraries);
        }
        try t.expectEqual(@as(usize, 0), r.live_bytes);
        try t.expectEqual(r.pipelines, r.released);
    };
}

test "qwen27 quant: prompt long key stays separate and dispatch takes the entry row capacity" {
    var r: mtl.Record = .{};
    var k = quant.Kernels{ .allocator = a, .device = r.device(), .long_prompt = true };
    defer k.deinit();
    const head = matrix(&r, 98592);
    try k.prepareDecode(head);
    try t.expectEqual(@as(u32, 1), k.lanes.count());
    var entries = k.lanes.iterator();
    const first = entries.next().?;
    const short = first.key_ptr.*;
    const pipe = first.value_ptr.pipe.id;
    k.prompt = true;
    try launch(&k, &r, head, 2048);
    try t.expectEqual(@as(u32, 2), k.lanes.count());
    var long = short;
    long.long = true;
    try t.expect(k.lanes.contains(short));
    try t.expect(k.lanes.contains(long));
    const entry = k.lanes.get(long).?;
    try t.expectEqual(@as(u32, lane_projection.max_reg_rows), entry.max_rows);
    try t.expectEqual(@as(usize, 1), r.count);
    try t.expectEqual(entry.pipe.id, r.launches[0].pipeline);
    try t.expectEqual(@as(u32, 2048), r.launches[0].rows);
    try t.expect(entry.pipe.id != pipe);
    k.prompt = false;
    r.count = 0;
    try launch(&k, &r, head, 129);
    try t.expectEqual(@as(usize, 2), r.count);
    try t.expectEqual(pipe, r.launches[0].pipeline);
    try t.expectEqual(pipe, r.launches[1].pipeline);
    try t.expectEqual(@as(u32, 128), r.launches[0].rows);
    try t.expectEqual(@as(u32, 1), r.launches[1].rows);
}

test "qwen27 quant: failed preparation leaves no entry and a retry is reused" {
    var r = mtl.Record{ .fail_pipeline = true };
    var k = quant.Kernels{ .allocator = a, .device = r.device() };
    defer k.deinit();
    const head = matrix(&r, 98592);
    try t.expectError(error.PipelineBuild, k.prepareDecode(head));
    try t.expectEqual(@as(u32, 0), k.lanes.count());
    try t.expectEqual(@as(usize, 0), r.live_bytes);
    r.fail_pipeline = false;
    try k.prepareDecode(head);
    const count = r.pipelines;
    try launch(&k, &r, head, 1);
    try t.expectEqual(count, r.pipelines);
}

test "qwen27 quant: preparation skips projections that cannot take a decode lane" {
    var r: mtl.Record = .{};
    var k = quant.Kernels{ .allocator = a, .device = r.device() };
    defer k.deinit();
    const head = matrix(&r, 98592);
    k.prompt = true;
    try k.prepareDecode(head);
    k.prompt = false;
    var other = head;
    other.tile = 16;
    try k.prepareDecode(other);
    other = head;
    other.raw = .{ .w = r.virtual(), .scales = r.virtual(), .biases = r.virtual(), .n = head.n, .k = head.k };
    try k.prepareDecode(other);
    other = head;
    other.runs = &.{.{ .linear = head, .first = 0 }};
    try k.prepareDecode(other);
    try t.expectEqual(@as(u32, 0), k.lanes.count());
    try t.expectEqual(@as(usize, 0), r.libraries);
}
