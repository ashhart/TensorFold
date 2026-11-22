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
        try t.expectEqual(@as(usize, 128), r.count);
        for (r.launches[0..r.count]) |call| {
            try t.expectEqual(@as(u32, 16), call.rows);
            try t.expectEqual(@as(usize, 1), call.grid.height);
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

fn rawMatrix(r: *mtl.Record, n: u32, bits: u8, group: u32) projection.Linear {
    var p = matrix(r, n);
    p.raw = .{ .w = r.virtual(), .scales = r.virtual(), .biases = r.virtual(), .n = n, .k = p.k, .bits = bits, .group = group };
    return p;
}

/// Cache decisions only: the recording boundary cannot run the chip's one-time twin byte comparison.
fn twins(k: *quant.Kernels, p: projection.Linear) !void {
    try k.twins.put(a, .{ p.n, p.k }, false);
    try k.pair_twins.put(a, .{ p.n, p.k }, false);
    if (p.raw) |raw| try k.bit_twins.put(a, .{ raw.bits, p.n, p.k, p.gn }, false);
    for (p.runs) |run| try twins(k, run.linear);
}

test "qwen27 quant: independent q/kv and qkv/zba raw writes have one caller barrier" {
    for ([_][2]u32{ .{ 8192, 1024 }, .{ 12288, 6240 } }) |widths| {
        for ([_]u8{ 4, 8 }) |bits| for ([_]u32{ 32, 64 }) |group| {
            var r = mtl.Record{ .tensor_units = false };
            var k = quant.Kernels{ .allocator = a, .device = r.device() };
            defer k.deinit();
            const left = rawMatrix(&r, widths[0], bits, group);
            const right = rawMatrix(&r, widths[1], bits, group);
            try twins(&k, left);
            try twins(&k, right);
            const input = quant.Ref{ .buffer = r.virtual() };
            const q_out = quant.Ref{ .buffer = r.virtual(), .offset = 64 };
            const kv_out = quant.Ref{ .buffer = r.virtual(), .offset = 128 };
            for ([_]u32{ 1, 8, 128, 129 }) |rows| {
                r.count = 0;
                r.barriers = 0;
                const e = r.encoder();
                try k.quant(e, left, input, null, input, q_out, rows);
                const middle = r.count;
                try k.quant(e, right, input, null, input, kv_out, rows);
                e.barrier();
                try t.expect(middle > 0 and r.count > middle);
                for (r.launches[0..r.count], 0..) |l, i| {
                    try t.expectEqual(@as(usize, 0), l.barriers);
                    try t.expectEqual(input.buffer.id, l.input.?.id);
                    try t.expectEqual(if (i < middle) q_out.buffer.id else kv_out.buffer.id, l.output.?.id);
                }
                try t.expectEqual(@as(usize, 1), r.barriers);
            }
        };
    }
}

test "qwen27 quant: mixed raw columns remain unordered until the caller barrier" {
    for ([_]u32{ 32, 64 }) |group| {
        var r = mtl.Record{ .tensor_units = false };
        var k = quant.Kernels{ .allocator = a, .device = r.device() };
        defer k.deinit();
        const members = [_]projection.Run{
            .{ .linear = rawMatrix(&r, 512, 4, group), .first = 0 },
            .{ .linear = rawMatrix(&r, 512, 8, group), .first = 512 },
        };
        var mixed = matrix(&r, 1024);
        mixed.runs = &members;
        try twins(&k, mixed);
        const x = quant.Ref{ .buffer = r.virtual() };
        const y = quant.Ref{ .buffer = r.virtual(), .offset = 64 };
        try k.quant(r.encoder(), mixed, x, null, x, y, 129);
        try t.expectEqual(@as(usize, 4), r.count);
        for (r.launches[0..r.count], 0..) |l, i| {
            try t.expectEqual(@as(usize, 0), l.barriers);
            try t.expectEqual(y.buffer.id, l.output.?.id);
            try t.expectEqual(64 + (i % 2) * 128 * 1024 * 2 + (i / 2) * 512 * 2, l.output.?.offset);
        }
        r.encoder().barrier();
        try t.expectEqual(@as(usize, 1), r.barriers);
    }
}

test "qwen27 quant: norm and MLP compile XS only on tensor units and pad only its grid" {
    var r: mtl.Record = .{};
    var k = quant.Kernels{ .allocator = a, .device = r.device(), .raw_tiles = false };
    defer k.deinit();
    const ref = quant.Ref{ .buffer = r.virtual() };
    for ([_]bool{ false, true, false }) |units| {
        r.tensor_units = units;
        for ([_]u32{ 1, 7, 16, 17 }) |rows| {
            r.count = 0;
            try k.norm(r.encoder(), ref, null, ref, ref, ref, ref, ref, ref, 512, rows);
            try k.norm(r.encoder(), ref, ref, ref, ref, ref, ref, ref, ref, 512, rows);
            try k.mlp(r.encoder(), ref, ref, ref, ref, ref, 1024, rows);
            try t.expectEqual(@as(usize, 3), r.count);
            for (r.launches[0..r.count]) |l| {
                try t.expectEqual(@as(?bool, units), l.xs);
                try t.expectEqual(@as(usize, if (units) quant.padded(rows) else rows), l.grid.height);
            }
        }
    }
    try t.expectEqual(@as(u32, 6), k.cache.count());
    try t.expectEqual(@as(usize, 6), r.libraries);
}

test "qwen27 quant: block source is appended only when opted in" {
    const off = lane_projection.Layout{ .n = 64, .k = 5120, .format = .{ .bits = 8, .group = 64 }, .precompute_sums = true, .row_blocks = true, .rows = 2048 };
    const plain = try lane_projection.source(a, off);
    defer a.free(plain);
    try t.expect(std.mem.indexOf(u8, plain, "tf_lane_block") == null);
    try t.expect(std.mem.indexOf(u8, plain, "TF_TM") == null);
    var on = off;
    on.tm = 4;
    const wide = try lane_projection.source(a, on);
    defer a.free(wide);
    try t.expect(std.mem.indexOf(u8, wide, "#define TF_TM 4\n") != null);
    try t.expect(std.mem.endsWith(u8, wide, @import("kernel_sources").core_lane_block));
    on.tm = 3;
    try t.expectError(error.InvalidSchedule, on.validate());
    on.tm = 2;
    on.format.bits = 5;
    try t.expectError(error.UnsupportedBlockFormat, on.validate());
}

test "qwen27 quant: a split that cannot fit takes fewer blocks and never changes" {
    try t.expectEqual(@as(u8, 4), lane_projection.blockTm(1, 4));
    try t.expectEqual(@as(u8, 4), lane_projection.blockTm(4, 4));
    try t.expectEqual(@as(u8, 2), lane_projection.blockTm(8, 4));
    try t.expectEqual(@as(u8, 2), lane_projection.blockTm(8, 2));
    try t.expectEqual(@as(u8, 1), quant.promptTm(quant.longPrompt("Apple M3 Ultra", false), 4));
}

test "qwen27 quant: M5 prompt schedules from explicit and default block counts" {
    const env = @import("sys").environment;
    const shapes = [_]struct { n: u32, k: u32, prompt_sk: u32, decode_sk: u32 }{
        .{ .n = 34816, .k = 5120, .prompt_sk = 1, .decode_sk = 2 },
        .{ .n = 10240, .k = 5120, .prompt_sk = 1, .decode_sk = 4 },
        .{ .n = 5120, .k = 17408, .prompt_sk = 2, .decode_sk = 8 },
        .{ .n = 5120, .k = 6144, .prompt_sk = 2, .decode_sk = 8 },
        .{ .n = 12288, .k = 5120, .prompt_sk = 2, .decode_sk = 4 },
        .{ .n = 6240, .k = 5120, .prompt_sk = 4, .decode_sk = 8 },
        .{ .n = 2048, .k = 5120, .prompt_sk = 8, .decode_sk = 8 },
    };
    const long = quant.longPrompt("Apple M5 Ultra", true);
    for ([_]?[]const u8{ "1", "2", "4", null, "" }, [_]u8{ 1, 2, 4, 4, 1 }) |raw, want| {
        const blocks = try env.value(u8, "TF_Q27_PROMPT_TM", raw, if (raw == null) 4 else 1);
        try t.expectEqual(want, blocks);
        for (shapes) |shape| for ([_]u8{ 4, 6, 8 }) |bits| {
            var r: mtl.Record = .{ .tensor_units = true };
            var k = quant.Kernels{
                .allocator = a,
                .device = r.device(),
                .long_prompt = long,
                .prompt = true,
                .prompt_tm = quant.promptTm(long, blocks),
            };
            defer k.deinit();
            var p = matrix(&r, shape.n);
            p.k = shape.k;
            p.bits = bits;
            p.slices = if (bits == 4 and shape.n == 6240) 4 else shape.decode_sk;
            const sk = if (bits == 4) shape.prompt_sk else shape.decode_sk;
            const tm = if (sk == 8 and want == 4) 2 else want;
            for ([_]u32{ 129, 512, 1024, 2048 }) |rows| {
                r.count = 0;
                try launch(&k, &r, p, rows);
                var it = k.lanes.keyIterator();
                const key = it.next().?.*;
                try t.expectEqual(tm, key.tm);
                try t.expectEqual(sk, key.slices);
                try t.expectEqual(@as(usize, 1), r.count);
                try t.expectEqual(@as(usize, (rows + 16 * @as(u32, tm) - 1) / (16 * @as(u32, tm))), r.launches[0].grid.height);
            }
        };
    }
}

test "qwen27 quant: decoded rows and an eight-way split stay within their block limits" {
    var r: mtl.Record = .{};
    var k = quant.Kernels{ .allocator = a, .device = r.device(), .long_prompt = true, .prompt = true, .prompt_tm = 4 };
    defer k.deinit();
    var p = matrix(&r, 2048);
    p.bits = 8;
    p.slices = 8;
    try launch(&k, &r, p, 2048);
    var it = k.lanes.keyIterator();
    try t.expectEqual(@as(u8, 2), it.next().?.tm);
    try t.expectEqual(@as(usize, 64), r.launches[0].grid.height);
    k.prompt = false;
    r.count = 0;
    try launch(&k, &r, p, 16);
    try t.expectEqual(@as(usize, 1), r.launches[0].grid.height);
}

test "qwen27 quant: block code addresses reach every tiled 6- and 8-bit code once" {
    var prng = std.Random.DefaultPrng.init(0x212);
    for ([_]u8{ 6, 8 }) |bits| {
        const wp: usize = 64 * @as(usize, bits) / 32;
        var codes: [32][64]u8 = undefined;
        var words: [32 * 16]u32 = @splat(0);
        for (0..32) |col| for (0..64) |code| {
            const v = prng.random().int(u8) & @as(u8, @intCast((@as(u16, 1) << @intCast(bits)) - 1));
            codes[col][code] = v;
            const at = col * wp * 32 + code * bits;
            for (0..bits) |bit| words[(at + bit) / 32] |= @as(u32, (v >> @intCast(bit)) & 1) << @intCast((at + bit) % 32);
        };
        var seen: [32][64]bool = @splat(@splat(false));
        for (0..32) |l| for (0..4) |nj| for (0..4) |kc| {
            const k0 = 4 * (l & 1) + 8 * ((l >> 3) & 1);
            const c = ((l >> 1) & 3) + 4 * ((l >> 4) & 1) + 8 * nj;
            const col = words[c * wp ..];
            var v: u32 = undefined;
            if (bits == 8) v = col[(k0 + 16 * kc) / 4] else {
                const bit = 6 * (k0 + 16 * kc);
                const sh: u5 = @intCast(bit & 31);
                v = col[bit >> 5] >> sh;
                if (sh > 8) v |= col[(bit >> 5) + 1] << @intCast(32 - @as(u6, sh));
            }
            for (0..4) |e| {
                const got = (v >> @intCast(bits * e)) & ((@as(u32, 1) << @intCast(bits)) - 1);
                try t.expectEqual(@as(u32, codes[c][k0 + 16 * kc + e]), got);
                try t.expect(!seen[c][k0 + 16 * kc + e]);
                seen[c][k0 + 16 * kc + e] = true;
            }
        };
        for (seen) |column| for (column) |hit| try t.expect(hit);
    }
}
