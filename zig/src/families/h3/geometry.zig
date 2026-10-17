//! H3's host admission, before shader compilation, buffer allocation or indexing caller data.
const std = @import("std");
const st = @import("safetensors");

pub const scalar_max = std.math.maxInt(i32);
pub const tile_rows = 64;
pub const tile_max = 32768;

pub fn product(values: []const usize) !usize {
    var n: usize = 1;
    for (values) |v| n = std.math.mul(usize, n, v) catch return error.H3BufferSizeOverflow;
    if (n > std.math.maxInt(isize)) return error.H3BufferSizeOverflow;
    return n;
}

pub fn sum(values: []const usize) !usize {
    var n: usize = 0;
    for (values) |v| n = std.math.add(usize, n, v) catch return error.H3RowCountOverflow;
    if (n > scalar_max) return error.H3RowCountOverflow;
    return n;
}

pub fn dimension(n: usize) !void {
    if (n == 0 or n > scalar_max) return error.H3BadDimension;
}

pub const Rows = struct {
    prefix_tiles: usize,
    tiles: usize,
    keep: usize,
    text: usize,
    condition: usize,
    audio: usize,
    held_audio: usize,
    video: usize,
    rows: usize,
};

pub fn counts(v: []const i32) !Rows {
    if (v.len < 6 or v.len > 8) return error.H3BadCounts;
    for (v) |n| if (n < 0) return error.H3NegativeCount;
    const prefix: usize = @intCast(v[0]);
    const video_tiles: usize = @intCast(v[1]);
    const keep: usize = @intCast(v[2]);
    const text: usize = @intCast(v[3]);
    const audio: usize = @intCast(v[4]);
    const video: usize = @intCast(v[5]);
    const condition: usize = if (v.len >= 7) @intCast(v[6]) else 0;
    const held: usize = if (v.len == 8) @intCast(v[7]) else 0;
    if (held > audio) return error.H3HeldAudioExceedsTotal;
    if (audio == 0 or video == 0) return error.H3EmptyMedia;
    const tiles = try sum(&.{ prefix, video_tiles });
    if (prefix == 0 or video_tiles == 0 or tiles > tile_max) return error.H3BadTiles;
    if (keep > video_tiles) return error.H3BadKeep;
    const before = try sum(&.{ text, condition, audio });
    const rows = try sum(&.{ before, video });
    if (before < prefix or before > prefix * tile_rows or video < video_tiles or video > video_tiles * tile_rows)
        return error.H3TileCapacity;
    return .{ .prefix_tiles = prefix, .tiles = tiles, .keep = keep, .text = text, .condition = condition, .audio = audio, .held_audio = held, .video = video, .rows = rows };
}

pub const Geometry = struct {
    layout: Rows,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    video_in: usize,
    audio_in: usize,
    lines: usize,
    rot: usize,
    times: usize,
    steps: usize,

    pub fn buffers(g: Geometry, base: bool) !void {
        const r = g.layout;
        const slots = r.tiles * tile_rows;
        const products = [_][]const usize{
            &.{ r.text, g.hidden, 2 },                 &.{ g.times, 2, g.hidden, 2 },
            &.{ g.hidden, g.video_in, 4 },             &.{ g.hidden, g.audio_in, 4 },
            &.{ r.rows, g.rot, 4 },                    &.{ g.steps, r.rows, 4 },
            &.{ slots, 4 },                            &.{ r.tiles, 4 },
            &.{ r.rows, 4 },                           &.{ r.rows, g.hidden, 2 },
            &.{ r.rows, g.inner, 2 },                  &.{ r.rows, if (base) @as(usize, 3) else 1, g.inner, 2 },
            &.{ r.rows, 2, g.mlp, 2 },                 &.{ r.rows, g.mlp, 2 },
            &.{ g.heads, slots, 128, 2 },              &.{ g.heads, r.tiles, 128, 4 },
            &.{ g.heads, r.tiles, r.tiles, 4 },        &.{ g.heads, r.tiles - r.prefix_tiles, r.prefix_tiles + r.keep, 4 },
            &.{ @max(r.condition, 1), g.video_in, 4 }, &.{ r.video, g.video_in, 4 },
            &.{ r.audio, g.audio_in, 4 },              &.{ 3, g.inner, g.hidden, 2 },
            &.{ 2, g.mlp, g.hidden, 2 },               &.{ g.hidden, g.mlp, 2 },
            &.{ 6, g.lines, g.hidden, 2 },
        };
        for (products) |p| _ = try product(p);
    }
};

pub const Case = struct {
    names: st.Header,
    data: []const u8,

    pub fn entry(c: Case, name: []const u8) !st.Entry {
        return c.names.get(name) orelse error.MissingTensor;
    }

    pub fn bytes(c: Case, name: []const u8) ![]const u8 {
        const e = try c.entry(name);
        if (e.end < e.begin or e.end > c.data.len) return error.H3BadTensorRange;
        return c.data[e.begin..e.end];
    }

    pub fn expect(c: Case, name: []const u8, dtype: st.DType, shape: []const usize) !void {
        const e = try c.entry(name);
        if (e.dtype != dtype or e.rank != shape.len or !std.mem.eql(usize, e.shape[0..e.rank], shape))
            return error.H3BadTensorShape;
        const n = try product(shape);
        if ((try c.bytes(name)).len != try product(&.{ n, dtype.size() })) return error.H3BadTensorRange;
    }

    pub fn int(c: Case, name: []const u8, i: usize) !i32 {
        const b = try c.bytes(name);
        const at = try product(&.{ i, 4 });
        if (at > b.len or b.len - at < 4) return error.H3BadTensorRange;
        return std.mem.readInt(i32, b[at..][0..4], .little);
    }
};

fn matrix(e: st.Entry) !void {
    if (e.rank != 2 or e.dtype != .bf16) return error.H3BadCheckpointShape;
    for (e.shape[0..2]) |n| dimension(n) catch return error.H3BadCheckpointShape;
}

/// `data` starts after the safetensors header; the same reader serves the Metal loader and portable fixtures.
pub fn read(a: std.mem.Allocator, c: Case, q: st.Entry, fc2: st.Entry, layers: usize, base: bool) !Geometry {
    const e = try c.entry("geometry");
    if (e.rank != 1 or e.dtype != .i32 or e.shape[0] < 6 or e.shape[0] > 8) return error.H3BadCounts;
    try c.expect("geometry", .i32, &.{e.shape[0]});
    var v: [8]i32 = undefined;
    for (v[0..e.shape[0]], 0..) |*n, i| n.* = try c.int("geometry", i);
    const r = try counts(v[0..e.shape[0]]);
    try matrix(q);
    try matrix(fc2);
    const divisor: usize = if (base) 3 else 1;
    if (q.shape[0] % divisor != 0) return error.H3BadCheckpointShape;
    const inner = q.shape[0] / divisor;
    const hidden = q.shape[1];
    const mlp = fc2.shape[1];
    if (layers == 0 or layers > scalar_max or fc2.shape[0] != hidden or
        inner == 0 or inner % 256 != 0 or hidden % 256 != 0 or mlp % 256 != 0)
        return error.H3BadCheckpointShape;
    _ = try sum(&.{ inner, inner, inner });
    _ = try sum(&.{ mlp, mlp });
    const vi = try c.entry("video_in.weight");
    const ai = try c.entry("audio_in.weight");
    const tab = try c.entry("tables.0");
    const cos = try c.entry("cos");
    const final = try c.entry("final");
    const adaln = try c.entry("adaln");
    if (vi.rank != 2 or ai.rank != 2 or tab.rank != 3 or cos.rank != 2 or final.rank != 2 or adaln.rank != 2)
        return error.H3BadTensorShape;
    const g = Geometry{ .layout = r, .inner = inner, .hidden = hidden, .heads = inner / 128, .mlp = mlp, .video_in = vi.shape[1], .audio_in = ai.shape[1], .lines = tab.shape[1], .rot = cos.shape[1], .times = final.shape[0], .steps = adaln.shape[0] };
    for ([_]usize{ g.video_in, g.audio_in, g.lines, g.times, g.steps }) |n| try dimension(n);
    if (g.rot == 0 or g.rot > 128 or g.rot % 2 != 0) return error.H3BadRotary;
    try g.buffers(base);
    try c.expect("text", .bf16, &.{ r.text, hidden });
    try c.expect("final", .bf16, &.{ g.times, 2 * hidden });
    try c.expect("final_norm.weight", .bf16, &.{hidden});
    inline for (.{ "video", "audio" }) |kind| {
        const width = if (comptime std.mem.eql(u8, kind, "video")) g.video_in else g.audio_in;
        const n = if (comptime std.mem.eql(u8, kind, "video")) r.video else r.audio;
        try c.expect(kind ++ "_in.weight", .f32, &.{ hidden, width });
        try c.expect(kind ++ "_in.bias", .f32, &.{hidden});
        try c.expect(kind ++ "_out.weight", .f32, &.{ width, hidden });
        try c.expect(kind ++ "_out.bias", .f32, &.{width});
        try c.expect(kind, .f32, &.{ n, width });
        try c.expect(kind ++ "_velocity", .f32, &.{ n, width });
        try c.expect(kind ++ "_step", .f32, &.{ g.steps, 2 });
    }
    if (r.condition != 0) try c.expect("condition", .f32, &.{ r.condition, g.video_in });
    try c.expect("cos", .f32, &.{ r.rows, g.rot });
    try c.expect("sin", .f32, &.{ r.rows, g.rot });
    try c.expect("adaln", .i32, &.{ g.steps, r.rows });
    try c.expect("times", .i32, &.{ g.steps, r.rows });
    try c.expect("tile_slot", .i32, &.{r.rows});
    try c.expect("tile_sizes", .i32, &.{r.tiles});
    var name: [64]u8 = undefined;
    for (0..layers) |i| try c.expect(try std.fmt.bufPrint(&name, "tables.{d}", .{i}), .bf16, &.{ 6, g.lines, hidden });
    for (0..g.steps * r.rows) |i| {
        const line = try c.int("adaln", i);
        const time = try c.int("times", i);
        if (line < 0 or line >= g.lines) return error.H3BadModulationIndex;
        if (time < 0 or time >= g.times) return error.H3BadTimeIndex;
    }
    var real: usize = 0;
    for (0..r.tiles) |i| {
        const size = try c.int("tile_sizes", i);
        if (size < 1 or size > tile_rows) return error.H3BadTileSize;
        real += @intCast(size);
    }
    if (real != r.rows) return error.H3TileSizeMismatch;
    var seen = try std.DynamicBitSetUnmanaged.initEmpty(a, r.tiles * tile_rows);
    defer seen.deinit(a);
    for (0..r.rows) |row| {
        const slot = try c.int("tile_slot", row);
        if (slot < 0 or slot >= r.tiles * tile_rows) return error.H3BadTileSlot;
        const at: usize = @intCast(slot);
        if (seen.isSet(at)) return error.H3DuplicateTileSlot;
        seen.set(at);
        const tile = at / tile_rows;
        if (at % tile_rows >= try c.int("tile_sizes", tile)) return error.H3BadTileSlot;
        if ((row < r.text + r.condition + r.audio) != (tile < r.prefix_tiles)) return error.H3BadTilePartition;
    }
    return g;
}

pub fn forward(rows: usize, video_n: usize, audio_n: usize, adaln_n: usize, times_n: usize, step: usize, video: usize, audio: usize, vv: usize, av: usize) !void {
    if (rows == 0 or adaln_n % rows != 0 or times_n != adaln_n or step >= adaln_n / rows)
        return error.H3BadStep;
    if (video != video_n or vv != video_n or audio != audio_n or av != audio_n) return error.H3BadLatentLength;
}

pub fn stepRange(from: usize, steps: usize) !void {
    if (from >= steps) return error.H3BadStep;
}
