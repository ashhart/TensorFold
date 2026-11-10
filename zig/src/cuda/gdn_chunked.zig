//! Opt-in chunkwise GDN operator; the caller supplies the image and retains the serial production fallback.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const launch = @import("launch.zig");

pub const width = 128;
pub const chunk_rows = 16;
pub const value_rows = 16;
pub const score_bytes = 2048;
pub const coefficient_bytes = 2176;
pub const scan_shared = 28928;
pub const symbols = [_][:0]const u8{
    "tf_gdn_chunk_classify", "tf_gdn_chunk_scores",   "tf_gdn_chunk_coefficients",
    "tf_gdn_chunk_scan",     "tf_gdn_chunk_fallback",
};
pub const Shape = struct {
    rows: usize,
    hk: usize,
    hv: usize,
    pub fn valid(s: Shape) bool {
        return s.rows > 0 and s.rows <= 131072 and s.hk > 0 and s.hv > 0 and s.hv <= 128 and s.hv % s.hk == 0;
    }
};
pub const Layout = struct {
    scores: usize = 0,
    coefficients: usize,
    flags: usize,
    bytes: usize,
    pub fn init(s: Shape) !Layout {
        if (!s.valid()) return error.UnsupportedGdnChunkShape;
        const chunks = (s.rows + chunk_rows - 1) / chunk_rows;
        const coefficients = try std.math.mul(usize, try std.math.mul(usize, chunks, s.hk), score_bytes);
        const flags = try std.math.add(usize, coefficients, try std.math.mul(usize, try std.math.mul(usize, chunks, s.hv), coefficient_bytes));
        return .{ .coefficients = coefficients, .flags = flags, .bytes = std.mem.alignForward(usize, try std.math.add(usize, flags, s.hv * 4), 256) };
    }
};
pub const Buffers = struct { q: u64, k: u64, v: u64, decay: u64, beta: u64, initial: u64, state: u64, output: u64 };
pub const Chunked = struct {
    module: Module,
    functions: [symbols.len]Function,
    scratch: DeviceBuffer,
    capacity: Shape,
    value_tile: usize = value_rows,
    pub fn load(d: *const Driver, image: []const u8, capacity: Shape) !Chunked {
        const layout = try Layout.init(capacity);
        var module = try Module.load(d, image);
        errdefer module.unload();
        var functions: [symbols.len]Function = undefined;
        for (&functions, symbols) |*f, name| f.* = try module.function(name);
        return .{ .module = module, .functions = functions, .scratch = try DeviceBuffer.alloc(d, layout.bytes), .capacity = capacity };
    }
    pub fn loadTensor(d: *const Driver, image: []const u8, capacity: Shape, tile: usize) !Chunked {
        const name: [:0]const u8 = switch (tile) {
            16 => "tf_gdn_chunk_tensor_16",
            32 => "tf_gdn_chunk_tensor_32",
            else => return error.UnsupportedGdnChunkTile,
        };
        var result = try load(d, image, capacity);
        errdefer result.unload();
        result.functions[3] = try result.module.function(name);
        result.value_tile = tile;
        return result;
    }
    pub fn unload(self: *Chunked) void {
        self.scratch.free();
        self.module.unload();
    }
    pub fn run(self: *const Chunked, stream: Stream, s: Shape, b: Buffers) !void {
        const layout = try Layout.init(s);
        if (s.hk != self.capacity.hk or s.hv != self.capacity.hv or s.rows > self.capacity.rows or layout.bytes > self.scratch.len)
            return error.UnsupportedGdnChunkShape;
        for ([_]u64{ b.q, b.k, b.v, b.decay, b.beta, b.initial, b.state, b.output }) |p|
            if (p == 0 or p % 4 != 0) return error.InvalidGdnChunkBuffer;
        const scores = self.scratch.ptr + layout.scores;
        const coefficients = self.scratch.ptr + layout.coefficients;
        const flags = self.scratch.ptr + layout.flags;
        const chunks = (s.rows + chunk_rows - 1) / chunk_rows;
        var a: launch.Args = .{};
        for ([_]u64{ b.q, b.k, b.v, b.decay, b.beta, b.initial, flags }) |p| a.add(p);
        dimensions(&a, s, true);
        try launch.launch(self.functions[0], .{ .grid = .{ .x = @intCast(s.hv) }, .block = .{ .x = 256 } }, stream, &a);
        a = .{};
        for ([_]u64{ b.q, b.k, scores }) |p| a.add(p);
        dimensions(&a, s, false);
        try launch.launch(self.functions[1], .{ .grid = .{ .x = @intCast(chunks), .y = @intCast(s.hk) }, .block = .{ .x = 128 } }, stream, &a);
        a = .{};
        for ([_]u64{ scores, b.decay, b.beta, flags, coefficients }) |p| a.add(p);
        dimensions(&a, s, true);
        try launch.launch(self.functions[2], .{ .grid = .{ .x = @intCast(chunks), .y = @intCast(s.hv) }, .block = .{ .x = 32 } }, stream, &a);
        a = .{};
        for ([_]u64{ b.q, b.k, b.v, coefficients, flags, b.initial, b.state, b.output }) |p| a.add(p);
        dimensions(&a, s, true);
        try launch.launch(self.functions[3], .{ .grid = .{ .x = @intCast(s.hv), .y = @intCast(width / self.value_tile) }, .block = .{ .x = 256 } }, stream, &a);
        a = .{};
        for ([_]u64{ b.q, b.k, b.v, b.decay, b.beta, flags, b.initial, b.state, b.output }) |p| a.add(p);
        dimensions(&a, s, true);
        try launch.launch(self.functions[4], .{ .grid = .{ .x = @intCast(s.hv), .y = 2 }, .block = .{ .x = 128 } }, stream, &a);
    }
};
fn dimensions(a: *launch.Args, s: Shape, heads: bool) void {
    a.add(@as(i32, @intCast(s.rows)));
    a.add(@as(i32, @intCast(s.hk)));
    if (heads) a.add(@as(i32, @intCast(s.hv)));
}

test "GDN chunk layout covers short tails and 128K rows without overlap" {
    for ([_]usize{ 1, 15, 16, 17, 4096, 131072 }) |rows| {
        const layout = try Layout.init(.{ .rows = rows, .hk = 16, .hv = 48 });
        try std.testing.expect(layout.coefficients > layout.scores);
        try std.testing.expect(layout.flags > layout.coefficients);
        try std.testing.expect(layout.bytes >= layout.flags + 48 * 4);
        try std.testing.expectEqual(@as(usize, 0), layout.bytes % 256);
    }
    try std.testing.expectEqual(@as(usize, 35_127_552), (try Layout.init(.{ .rows = 4096, .hk = 16, .hv = 48 })).bytes);
    try std.testing.expectEqual(@as(usize, 28928), scan_shared);
}
test "GDN chunk layout refuses empty and inconsistent head geometry" {
    for ([_]Shape{ .{ .rows = 0, .hk = 16, .hv = 48 }, .{ .rows = 17, .hk = 0, .hv = 48 }, .{ .rows = 17, .hk = 17, .hv = 48 }, .{ .rows = 131073, .hk = 16, .hv = 48 } }) |s|
        try std.testing.expectError(error.UnsupportedGdnChunkShape, Layout.init(s));
}

test "GDN tensor selection rejects unsupported tiles before opening CUDA" {
    var driver: Driver = undefined;
    try std.testing.expectError(error.UnsupportedGdnChunkTile, Chunked.loadTensor(&driver, &.{}, .{ .rows = 17, .hk = 16, .hv = 48 }, 64));
}
