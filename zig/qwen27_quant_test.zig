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

const wide_formats = [_]lane_projection.Format{
    .{ .bits = 8, .group = 64 }, .{ .bits = 6, .group = 32 }, .{ .bits = 5, .group = 64 },
    .{ .bits = 3, .group = 32 }, .{ .bits = 2, .group = 64 }, .{ .bits = 8, .group = 128 },
};

test "qwen27 quant: non-reg layouts admit whole prompt chunks with row-block source variants" {
    for (wide_formats) |format| {
        const long = lane_projection.Layout{ .n = 64, .k = 5120, .format = format, .rows = 2048, .row_blocks = true };
        try long.validate();
        var short = long;
        short.rows = 16;
        const text = try lane_projection.source(a, long);
        defer a.free(text);
        const old = try lane_projection.source(a, short);
        defer a.free(old);
        try t.expect(std.mem.indexOf(u8, old, "#define TF_ROW_BLOCKS 0\n") != null);
        try t.expect(std.mem.indexOf(u8, text, "#define TF_ROW_BLOCKS 1\n") != null);
        for ([_]usize{ 0, 15, 17, 2064 }) |rows| {
            short.rows = rows;
            try t.expectError(error.InvalidSchedule, short.validate());
        }
    }
}

test "qwen27 quant: each non-reg whole prompt is one projection dispatch" {
    for (wide_formats) |format| for ([_]u32{ 17, 1065, 2048 }) |rows| {
        var r: mtl.Record = .{};
        var k = quant.Kernels{ .allocator = a, .device = r.device(), .long_prompt = true, .prompt = true };
        defer k.deinit();
        var p = matrix(&r, 64);
        p.bits = format.bits;
        p.group = @intCast(format.group);
        const ref = quant.Ref{ .buffer = r.virtual() };
        const producer: ?quant.Sums = if (format.bits == 8 and format.group == 64) .{ .ref = ref, .stride = 2080 } else null;
        try k.quant(r.encoder(), p, ref, producer, ref, ref, rows);
        try t.expectEqual(@as(usize, 1), r.count);
        try t.expectEqual(rows, r.launches[0].rows);
        try t.expectEqual(@as(usize, (rows + 15) / 16), r.launches[0].grid.height);
        try t.expectEqual(@as(i32, if (producer != null) 2080 else @intCast((rows + 15) / 16 * 16)), r.launches[0].stride);
        var entries = k.lanes.valueIterator();
        const entry = entries.next().?;
        try t.expectEqual(@as(u32, 2048), entry.max_rows);
        if (entry.sums_buffer) |b| try t.expectEqual(@as(usize, 2048 * (5120 / format.group) * 4), b.length());
        // A decode or verify row still uses the short key and its original 16-row steps.
        k.prompt = false;
        r.count = 0;
        try k.quant(r.encoder(), p, ref, producer, ref, ref, 17);
        try t.expectEqual(@as(usize, 2), r.count);
        try t.expectEqual(@as(u32, 16), r.launches[0].rows);
        try t.expectEqual(@as(u32, 1), r.launches[1].rows);
        try t.expectEqual(@as(usize, 1), r.launches[0].grid.height);
        try t.expectEqual(@as(u32, 2), k.lanes.count());
        k.prompt = true;
        k.long_prompt = false;
        r.count = 0;
        try k.quant(r.encoder(), p, ref, producer, ref, ref, 2048);
        // Tensor units take the row blocks a dispatch at every width; elsewhere 16 rows a dispatch.
        if (r.device().tensorUnits()) {
            try t.expectEqual(@as(usize, 1), r.count);
            try t.expectEqual(@as(u32, 2048), r.launches[0].rows);
            try t.expectEqual(@as(usize, (2048 + 15) / 16), r.launches[0].grid.height);
        } else {
            try t.expectEqual(@as(usize, 128), r.count);
            for (r.launches[0..r.count]) |call| {
                try t.expectEqual(@as(u32, 16), call.rows);
                try t.expectEqual(@as(usize, 1), call.grid.height);
            }
        }
    };
}

test "qwen27 quant: cooperative and older-chip projections keep their 16-row limits" {
    var r: mtl.Record = .{};
    var layout = lane_projection.Layout{ .n = 64, .k = 512, .format = .{ .bits = 4, .group = 64 }, .cooperative = true };
    const coop = try lane_projection.Projection.init(a, r.device(), layout);
    defer coop.deinit();
    try t.expectEqual(@as(u32, 16), coop.max_rows);
    const ref = lane_projection.Ref{ .buf = r.virtual() };
    try coop.encode(r.encoder(), ref, ref, ref, ref, 16);
    try t.expectEqual(@as(usize, 1), r.launches[0].grid.height);
    try t.expectError(error.InvalidRows, coop.encode(r.encoder(), ref, ref, ref, ref, 17));
    layout.rows = 2048;
    try t.expectError(error.InvalidSchedule, layout.validate());
    layout.cooperative = false;
    layout.format = .{ .bits = 6, .group = 32 };
    r.tensor_units = false;
    const old = try lane_projection.Projection.init(a, r.device(), layout);
    defer old.deinit();
    try t.expectEqual(@as(u32, 16), old.max_rows);
    try t.expectError(error.InvalidRows, old.encode(r.encoder(), ref, ref, ref, ref, 17));
    const over_budget = lane_projection.Layout{ .n = 2048, .k = 5120, .format = .{ .bits = 8, .group = 128 }, .sk = 8 };
    try t.expectError(error.ThreadgroupMemoryLimit, over_budget.validate());
}

test "qwen27 quant: default non-reg callers retain scalar source and 16-row capacity" {
    for (wide_formats) |format| {
        var r: mtl.Record = .{};
        const layout = lane_projection.Layout{ .n = 64, .k = 5120, .format = format };
        const text = try lane_projection.source(a, layout);
        defer a.free(text);
        try t.expect(std.mem.indexOf(u8, text, "#define TF_ROW_BLOCKS 0\n") != null);
        const p = try lane_projection.Projection.init(a, r.device(), layout);
        defer p.deinit();
        try t.expectEqual(@as(u32, 16), p.max_rows);
        const ref = lane_projection.Ref{ .buf = r.virtual() };
        try p.encode(r.encoder(), ref, ref, ref, ref, 16);
        try t.expectEqual(@as(usize, 1), r.launches[0].grid.height);
        try t.expectError(error.InvalidRows, p.encode(r.encoder(), ref, ref, ref, ref, 17));
    }
}

test "qwen27 quant: non-reg prompt key opts in only above 16 rows" {
    for (wide_formats) |format| for ([_]u32{ 16, 17 }) |rows| {
        var r: mtl.Record = .{};
        var k = quant.Kernels{ .allocator = a, .device = r.device(), .long_prompt = true, .prompt = true };
        defer k.deinit();
        var p = matrix(&r, 64);
        p.bits = format.bits;
        p.group = @intCast(format.group);
        const ref = quant.Ref{ .buffer = r.virtual() };
        const sums: ?quant.Sums = if (format.bits == 8 and format.group == 64) .{ .ref = ref, .stride = 32 } else null;
        try k.quant(r.encoder(), p, ref, sums, ref, ref, rows);
        var entries = k.lanes.iterator();
        const entry = entries.next().?;
        try t.expectEqual(rows > 16, entry.key_ptr.long);
        try t.expectEqual(@as(u32, if (rows > 16) 2048 else 16), entry.value_ptr.max_rows);
        try t.expectEqual(@as(usize, 1), r.count);
        try t.expectEqual(@as(usize, if (rows > 16) 2 else 1), r.launches[0].grid.height);
    };
}
