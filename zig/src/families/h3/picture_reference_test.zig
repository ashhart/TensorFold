//! MiniMax-H3-Image's edit reference rules (picture_reference.zig): sizes, crops, grids, layers, weights, rows.
const std = @import("std");
const st = @import("safetensors");
const r = @import("picture_reference.zig");
const vae = @import("picture_vae.zig");
const testing = std.testing;

fn arena() std.heap.ArenaAllocator {
    return std.heap.ArenaAllocator.init(testing.allocator);
}

fn expectSize(want: [2]usize, got: r.Size) !void {
    try testing.expectEqual(want[0], got.width);
    try testing.expectEqual(want[1], got.height);
}

test "picture reference sizes: the VAE bound's and the Qwen bound's examples, halves to even, refusals" {
    try expectSize(.{ 1024, 1536 }, try r.bounded(512, 768, r.vae_bound));
    try expectSize(.{ 2048, 2720 }, try r.bounded(3000, 4000, r.vae_bound));
    try expectSize(.{ 1536, 2048 }, try r.bounded(1536, 2048, r.vae_bound));
    try expectSize(.{ 1024, 1376 }, try r.bounded(1500, 2000, r.qwen_bound));
    // inside the VAE's range a side need not be a multiple of 32
    try expectSize(.{ 1500, 2000 }, try r.bounded(1500, 2000, r.vae_bound));
    try expectSize(.{ 1024, 1024 }, try r.bounded(1024, 1024, r.qwen_bound));
    // 520 x 2 / 32 = 32.5 goes down to 32, 536 x 2 / 32 = 33.5 up to 34
    try expectSize(.{ 1024, 1024 }, try r.bounded(520, 512, r.vae_bound));
    try expectSize(.{ 1088, 1024 }, try r.bounded(536, 512, r.vae_bound));
    try expectSize(.{ 1024, 4096 }, try r.bounded(32, 128, r.vae_bound));
    try testing.expectError(error.H3ReferenceTooSmall, r.bounded(31, 2000, r.vae_bound));
    try testing.expectError(error.H3ReferenceTooSmall, r.bounded(2000, 20, r.qwen_bound));
    try testing.expectError(error.H3ReferenceTooLarge, r.bounded(32, 8192, r.vae_bound));
    try testing.expectError(error.H3ReferenceTooLarge, r.bounded(2048, 32769, r.vae_bound));
}

test "picture reference grids: 16-pixel crop, full grid, even grid" {
    const square = try r.cropped(.{ .width = 1024, .height = 1024 });
    try testing.expectEqual(r.Grid{ .rows = 64, .columns = 64 }, try r.evenGrid(r.fullGrid(square)));
    // width 1050, height 1000: 1040 x 992 pixels, 62 x 65 cells, then 62 rows by 64 columns
    const wide = try r.cropped(.{ .width = 1050, .height = 1000 });
    try expectSize(.{ 1040, 992 }, wide);
    try testing.expectEqual(r.Grid{ .rows = 62, .columns = 65 }, r.fullGrid(wide));
    try testing.expectEqual(r.Grid{ .rows = 62, .columns = 64 }, try r.evenGrid(r.fullGrid(wide)));
    try testing.expectEqual(r.Grid{ .rows = 2, .columns = 4 }, try r.evenGrid(.{ .rows = 3, .columns = 5 }));
    try testing.expectError(error.H3ReferenceTooSmall, r.cropped(.{ .width = 15, .height = 40 }));
    try testing.expectError(error.H3ReferenceTooSmall, r.evenGrid(.{ .rows = 1, .columns = 8 }));
}

test "picture reference input: cropped top-left to 16s, x / 255 then ImageNet's statistics, in fp32" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const w = 18;
    const h = 17;
    const rgb = try x.alloc(u8, w * h * 3);
    for (rgb, 0..) |*b, i| b.* = @intCast((i * 37) % 256);
    const out = try x.alloc(f32, 16 * 16 * 3);
    try expectSize(.{ 16, 16 }, try r.input(rgb, w, h, out));
    for (0..16) |y| for (0..16) |col| for (0..3) |c| {
        const b: f32 = @floatFromInt(rgb[(y * w + col) * 3 + c]);
        try testing.expectEqual((b / 255.0 - vae.pixel_mean[c]) / vae.pixel_std[c], out[(y * 16 + col) * 3 + c]);
    };
    try testing.expectError(error.H3BadPictureSize, r.input(rgb, w, h, out[1..]));
    try testing.expectError(error.H3BadPictureSize, r.input(rgb[3..], w, h, out));
}

test "picture reference reflect padding: -1 reads 1, n reads n - 2, the inside as is" {
    try testing.expectEqual(@as(usize, 1), r.reflect(-1, 5));
    try testing.expectEqual(@as(usize, 3), r.reflect(5, 5));
    for (0..5) |i| try testing.expectEqual(i, r.reflect(@intCast(i), 5));
    // a downsample's right and bottom pad, and the smallest axis reflect allows
    try testing.expectEqual(@as(usize, 254), r.reflect(256, 256));
    try testing.expectEqual(@as(usize, 1), r.reflect(-1, 2));
    try testing.expectEqual(@as(usize, 0), r.reflect(2, 2));
}

fn shapeJson(a: std.mem.Allocator, dims: []const usize) ![]const u8 {
    return std.json.Stringify.valueAlloc(a, dims, .{});
}

const Entry = struct { []const u8, []const u8, []const usize };

/// A safetensors header for `entries` (name, dtype, shape), laid end to end; returns it parsed.
fn header(a: std.mem.Allocator, entries: []const Entry) !st.Header {
    var json: []const u8 = "{";
    var at: usize = 0;
    for (entries, 0..) |e, i| {
        var n: usize = if (std.mem.eql(u8, e[1], "BF16")) 2 else 4;
        for (e[2]) |d| n *= d;
        json = try std.fmt.allocPrint(a, "{s}{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[{d},{d}]}}", .{ json, if (i == 0) "" else ",", e[0], e[1], try shapeJson(a, e[2]), at, at + n });
        at += n;
    }
    return st.parseHeader(a, try std.fmt.allocPrint(a, "{s}}}", .{json}), at);
}

/// The released encoder's entries, with `change` applied to the one named (null drops it), plus the decoder's.
fn released(a: std.mem.Allocator, name: []const u8, change: ?Entry) ![]Entry {
    var list: std.ArrayList(Entry) = .empty;
    try list.append(a, .{ "decoder.norm_out.weight", "F32", &.{1024} });
    try list.append(a, .{ "post_quant_conv.weight", "F32", &.{ 24, 24, 1, 1 } });
    for (try r.items(a, r.released)) |it| for ([_][]const u8{ ".weight", ".bias" }) |part| {
        const full = try std.fmt.allocPrint(a, "{s}{s}", .{ it.name, part });
        const shape = r.weightShape(it);
        const dims = if (part[1] == 'w') try a.dupe(usize, shape.slice()) else try a.dupe(usize, &.{it.cout});
        const e: Entry = .{ full, "F32", dims };
        if (!std.mem.eql(u8, full, name)) try list.append(a, e) else if (change) |c| try list.append(a, c);
    };
    return list.items;
}

test "picture reference layers: the released names and shapes pass the header check, every difference refused" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const list = try r.items(x, r.released);
    try testing.expectEqual(@as(usize, 59), list.len);
    try testing.expectEqualStrings("encoder.conv_in", list[0].name);
    try testing.expectEqual(@as(usize, 3), list[0].cin);
    var shortcuts: usize = 0;
    var downs: usize = 0;
    for (list) |it| {
        if (std.mem.endsWith(u8, it.name, "nin_shortcut")) {
            shortcuts += 1;
            try testing.expect(it.kind == .conv1 and it.cin * 2 == it.cout);
        }
        if (std.mem.endsWith(u8, it.name, "downsample.conv")) downs += 1;
    }
    try testing.expectEqual(@as(usize, 3), shortcuts);
    try testing.expectEqual(@as(usize, 4), downs);
    try testing.expectEqualStrings("encoder.down.1.block.0.nin_shortcut", list[14].name);
    try testing.expectEqualStrings("quant_conv", list[58].name);
    try testing.expectEqual(@as(usize, 1024), list[57].cin);
    var culprit: []const u8 = "";
    const good = try header(x, try released(x, "", null));
    try r.checkHeader(x, &good, &culprit);
    const conv_in = "encoder.conv_in.weight";
    try testing.expectError(error.H3MissingEncoderTensor, r.checkHeader(x, &(try header(x, try released(x, conv_in, null))), &culprit));
    try testing.expectEqualStrings(conv_in, culprit);
    const flat: Entry = .{ conv_in, "F32", &.{ 128, 3, 3, 3 } };
    try testing.expectError(error.H3BadEncoderTensor, r.checkHeader(x, &(try header(x, try released(x, conv_in, flat))), &culprit));
    const half: Entry = .{ "quant_conv.bias", "BF16", &.{48} };
    try testing.expectError(error.H3BadEncoderTensor, r.checkHeader(x, &(try header(x, try released(x, "quant_conv.bias", half))), &culprit));
    try testing.expectEqualStrings("quant_conv.bias", culprit);
    const mid = [_]Entry{.{ "encoder.mid.block_1.conv1.weight", "F32", &.{ 8, 8, 3, 3, 3 } }};
    const extra = try std.mem.concat(x, Entry, &.{ try released(x, "", null), &mid });
    try testing.expectError(error.H3UnexpectedEncoderTensor, r.checkHeader(x, &(try header(x, extra)), &culprit));
    try testing.expectError(error.H3BadEncoderWidths, r.items(x, .{ .widths = .{ 128, 256, 256, 500, 512, 1024 } }));
}

test "picture reference weights: a 3x3x3 kernel's last time slice, a 1x1x1 kernel, rows padded to 64" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    const conv: r.Item = .{ .name = "c", .kind = .conv3, .cin = 2, .cout = 3 };
    const w = try x.alloc(f32, 3 * 2 * 27);
    for (0..3) |co| for (0..2) |ci| for (0..3) |t| for (0..9) |tap| {
        w[((co * 2 + ci) * 3 + t) * 9 + tap] = @floatFromInt(co * 1000 + ci * 100 + t * 10 + tap);
    };
    const out = try x.alloc(f32, 9 * 2 * 64);
    try r.pack(conv, w, out);
    for (0..9) |tap| for (0..2) |ci| for (0..64) |co| {
        const want: f32 = if (co < 3) @floatFromInt(co * 1000 + ci * 100 + 20 + tap) else 0;
        try testing.expectEqual(want, out[(tap * 2 + ci) * 64 + co]);
    };
    const one: r.Item = .{ .name = "q", .kind = .conv1, .cin = 2, .cout = 65 };
    const w1 = try x.alloc(f32, 65 * 2);
    for (w1, 0..) |*v, i| v.* = @floatFromInt(i);
    const out1 = try x.alloc(f32, 2 * 128);
    try r.pack(one, w1, out1);
    try testing.expectEqual(@as(f32, 64 * 2 + 1), out1[128 + 64]);
    try testing.expectEqual(@as(f32, 0), out1[128 + 65]);
    try testing.expectError(error.H3BadEncoderTensor, r.pack(conv, w[1..], out));
    try testing.expectError(error.H3BadEncoderTensor, r.pack(.{ .name = "n", .kind = .norm, .cin = 2, .cout = 2 }, w1[0..2], out1[0..2]));
}

test "picture reference config: encoder tiling must stay on" {
    var a = arena();
    defer a.deinit();
    try r.checkTiling(a.allocator(), "{\"vae_encoder_tiling\": 1}");
    try testing.expectError(error.H3UnsupportedVae, r.checkTiling(a.allocator(), "{\"vae_encoder_tiling\": 0}"));
    try testing.expectError(error.H3BadVaeConfig, r.checkTiling(a.allocator(), "{\"vae_decoder_tiling\": 1}"));
}

test "picture reference clean rows: sampled as fp16, normalised, the full grid cropped top-left to even, patchified" {
    var a = arena();
    defer a.deinit();
    const x = a.allocator();
    var c: vae.Config = .{ .mean = undefined, .std = undefined, .tile = 256, .overlap = 64, .rope_theta = 100 };
    for (&c.mean, &c.std, 0..) |*mu, *sd, i| {
        mu.* = @as(f32, @floatFromInt(i)) * 0.1 - 1;
        sd.* = 0.5 + @as(f32, @floatFromInt(i)) * 0.05;
    }
    const full: r.Grid = .{ .rows = 3, .columns = 5 };
    const n = 24 * 15;
    const m = try x.alloc(f32, 2 * n);
    const noise = try x.alloc(f32, n);
    for (m, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 97)) * 0.0137 - 0.5;
    for (noise, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.21 - 1.3;
    const got = try r.clean(x, &c, m, full, noise);
    try testing.expectEqual(r.Grid{ .rows = 2, .columns = 4 }, got.grid);
    try testing.expectEqual(@as(usize, 2 * 96), got.rows.len);
    for (0..2) |pc| for (0..24) |ch| for (0..2) |dy| for (0..2) |dx| {
        const at = (ch * 3 + dy) * 5 + 2 * pc + dx;
        const s = @exp(0.5 * std.math.clamp(m[n + at], -30.0, 20.0));
        const kept: f32 = @floatCast(@as(f16, @floatCast(m[at] + s * noise[at])));
        try testing.expectEqual((kept - c.mean[ch]) / c.std[ch], got.rows[pc * 96 + ch * 4 + dy * 2 + dx]);
    };
    try testing.expectError(error.H3BadLatentLength, r.clean(x, &c, m[1..], full, noise));
    try testing.expectError(error.H3ReferenceTooSmall, r.clean(x, &c, m[0 .. 2 * 24 * 5], .{ .rows = 1, .columns = 5 }, noise[0 .. 24 * 5]));
}

test "picture reference anchors: 0.999 of the clean row and 1 - 0.999 in fp32 of the noise" {
    try testing.expectEqual(@as(f64, 1) - @as(f64, @as(f32, 0.999)), @as(f64, r.anchor_noise));
    try testing.expect(r.anchor_noise != @as(f32, 0.001));
    var out: [2]f32 = undefined;
    try r.anchors(&.{ 2, -1 }, &.{ 4, 0.5 }, &out);
    try testing.expectEqual(@as(f32, 0.999) * 2 + r.anchor_noise * 4, out[0]);
    try testing.expectEqual(@as(f32, 0.999) * -1 + r.anchor_noise * 0.5, out[1]);
    try testing.expectError(error.H3BadLatentLength, r.anchors(&.{1}, &.{ 1, 2 }, &out));
}
