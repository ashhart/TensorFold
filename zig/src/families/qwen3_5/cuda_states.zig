//! A stream's state table on CUDA: the addresses of its state, rollback copy and log rows (qwen35.cuh Seg).
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");

const Engine = engine.Engine;
const DeviceBuffer = cuda.DeviceBuffer;

/// A stream's state table (qwen35.cuh Seg.rec, .conv, .before, .log): addresses of its state, copy and log rows.
pub const States = struct {
    dev: DeviceBuffer,
    host: []u64,
    /// The state, before and log addresses `dev` holds the entries of (null: not written yet).
    from: ?[3]u64 = null,

    pub fn len(g: geometry.Geometry) usize {
        return 4 * g.deltas() + engine.most_window;
    }

    pub fn bytes(g: geometry.Geometry) usize {
        return len(g) * 8;
    }

    pub fn init(gpa: std.mem.Allocator, e: *const Engine) !States {
        const host = try gpa.alloc(u64, len(e.g));
        errdefer gpa.free(host);
        return .{ .dev = try DeviceBuffer.alloc(e.d, bytes(e.g)), .host = host };
    }

    pub fn deinit(self: *States, gpa: std.mem.Allocator) void {
        self.dev.free();
        gpa.free(self.host);
    }

    /// The entries of a state block (recurrences, then tails), its copy and a log of whole rows; 0 gives 0 entries.
    pub fn fill(t: []u64, g: geometry.Geometry, state: u64, before: u64, log: u64) void {
        const n = g.deltas();
        const in = struct {
            fn f(base: u64, off: usize) u64 {
                return if (base == 0) 0 else base + off;
            }
        }.f;
        for (0..n) |l| {
            t[l] = in(state, l * g.recBytes());
            t[n + l] = in(state, n * g.recBytes() + l * g.convBytes());
            t[2 * n + l] = in(before, l * g.recBytes());
            t[3 * n + l] = in(before, n * g.recBytes() + l * g.convBytes());
        }
        for (0..engine.most_window) |r| t[4 * n + r] = in(log, r * n * g.logRowBytes());
    }

    /// Where the tables start in `dev` (Seg.rec, .conv, .before, .log).
    pub fn at(self: States, g: geometry.Geometry) [4]u64 {
        return parts(self.dev.ptr, g);
    }

    /// A table at `base`'s recurrence, tail, copy and log row entries (qwen35.cuh Seg.rec, .conv, .before, .log).
    pub fn parts(base: u64, g: geometry.Geometry) [4]u64 {
        const n = g.deltas();
        return .{ base, base + n * 8, base + 2 * n * 8, base + 4 * n * 8 };
    }

    /// Window row `row`'s record base (cuda_cache.zig's pending copy).
    pub fn logRow(self: States, g: geometry.Geometry, row: usize) u64 {
        return self.host[4 * g.deltas() + row];
    }
};

/// The dense 27B's and the 35B-A3B's linear widths: every fourth layer attention.
fn stateGeometry(layers: usize, v_heads: usize) geometry.Geometry {
    var linear: [geometry.max_layers]bool = @splat(false);
    for (linear[0..layers], 0..) |*l, i| l.* = i % 4 != 3;
    return .{ .hidden = 256, .intermediate = 256, .layers = layers, .vocab = 8, .k_heads = 16, .v_heads = v_heads, .q_heads = 16, .kv_heads = 2, .eps = 1e-6, .tied = true, .context = 64, .linear = linear };
}

test "Qwen state table: every entry is the address the kernels computed from one block before the table" {
    const t = std.testing;
    for ([_][2]usize{ .{ 64, 48 }, .{ 40, 32 } }) |shape| {
        const g = stateGeometry(shape[0], shape[1]);
        const nl = g.deltas();
        const host = try t.allocator.alloc(u64, States.len(g));
        defer t.allocator.free(host);
        const state: u64 = 0x7f00_0000_0000;
        const before: u64 = 0x7f10_0000_0100;
        const log: u64 = 0x7f20_0000_0200;
        States.fill(host, g, state, before, log);
        // qwen35.cuh before the table: rec() floats and conv_rows() bf16 a layer, the log's (row * nl + layer) records
        const rec: u64 = g.v_heads * 128 * 128;
        const conv_rows: u64 = 3 * (2 * g.k_heads + g.v_heads) * 128;
        const log_bytes: u64 = g.v_heads * 4 + g.v_heads * 128 * 4 + g.k_heads * 128 * 2 + conv_rows * 2;
        for (0..nl) |l| {
            try t.expectEqual(state + l * rec * 4, host[l]);
            try t.expectEqual(state + nl * rec * 4 + l * conv_rows * 2, host[nl + l]);
            try t.expectEqual(before + l * rec * 4, host[2 * nl + l]);
            try t.expectEqual(before + nl * rec * 4 + l * conv_rows * 2, host[3 * nl + l]);
        }
        for (0..engine.most_window) |r| try t.expectEqual(log + r * nl * log_bytes, host[4 * nl + r]);
        // the last entries end inside their blocks
        try t.expectEqual(state + g.stateBytes(), host[2 * nl - 1] + g.convBytes());
        try t.expectEqual(log + g.logBytes(engine.most_window), host[4 * nl + engine.most_window - 1] + nl * g.logRowBytes());
    }
}

test "Qwen state table: a block that is not there gives zero entries, the others stay" {
    const t = std.testing;
    const g = stateGeometry(64, 48);
    const nl = g.deltas();
    const host = try t.allocator.alloc(u64, States.len(g));
    defer t.allocator.free(host);
    @memset(host, 0xaa);
    States.fill(host, g, 0x1000_0000, 0, 0);
    for (host[2 * nl ..]) |x| try t.expectEqual(@as(u64, 0), x);
    try t.expectEqual(@as(u64, 0x1000_0000), host[0]);
    try t.expectEqual(@as(u64, 4 * nl + engine.most_window), States.len(g));
}
