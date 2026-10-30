//! The production Metal QSA adapter is exercised through recording commands, without a shader or device.
const std = @import("std");
const family = @import("qwen4").family;
const qsa = family.cuda_attention;
const t = std.testing;
const R = struct {
    pub const Size = struct {
        width: usize,
        height: usize,
        depth: usize,
        pub fn of(x: usize, y: usize, z: usize) @This() {
            return .{ .width = x, .height = y, .depth = z };
        }
    };
    pub const Device = struct { id: usize = 1 };
    pub const Queue = struct {
        owner: usize = 1,
        pub fn deviceId(q: Queue) usize {
            return q.owner;
        }
    };
    pub const Buffer = struct {
        address: u64,
        size: usize,
        owner: usize = 1,
        pub fn deviceId(b: Buffer) usize {
            return b.owner;
        }
        pub fn gpuAddress(b: Buffer) u64 {
            return b.address;
        }
        pub fn length(b: Buffer) usize {
            return b.size;
        }
    };
    pub const CompileOptions = struct {
        pub fn mlx() @This() {
            return .{};
        }
    };
    pub var live: usize = 0;
    pub var calls: usize = 0;
    pub var fail_at: ?usize = null;
    pub var selected: usize = 0;
    pub var p: qsa.Call = undefined;
    pub var binds: [11]?u64 = @splat(null);
    pub var inline_values: [11][128]i32 = undefined;
    pub var inline_len: [11]usize = @splat(0);
    pub var grids: [6]?Size = @splat(null);
    pub var copies: usize = 0;
    pub var saw_parts = false;
    pub const Library = struct {
        pub fn fromSource(_: Device, text: []const u8, _: CompileOptions) !Library {
            std.debug.assert(std.mem.indexOf(u8, text, "constexpr int NKV = 4;") != null);
            return .{};
        }
        pub fn deinit(_: Library) void {}
    };
    pub const Pipeline = struct {
        index: usize,
        pub fn init(_: Device, _: Library, name: []const u8, _: bool) !Pipeline {
            for (family.metal_attention.Kernels(R).names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
                if (fail_at == i) return error.PlantedPipeline;
                live += 1;
                return .{ .index = i };
            };
            return error.UnownedSymbol;
        }
        pub fn maxThreads(_: Pipeline) usize {
            return 1024;
        }
        pub fn simdWidth(_: Pipeline) usize {
            return 32;
        }
        pub fn deinit(_: Pipeline) void {
            live -= 1;
        }
    };
    pub const ComputeEncoder = struct {
        pub fn setPipeline(_: ComputeEncoder, pipe: Pipeline) void {
            selected = pipe.index;
            binds = @splat(null);
            inline_len = @splat(0);
            calls += 1;
        }
        pub fn setBuffer(_: ComputeEncoder, b: Buffer, off: usize, index: usize) void {
            binds[index] = b.address + off;
            calls += 1;
        }
        pub fn setBytes(_: ComputeEncoder, bytes: []const u8, index: usize) void {
            std.debug.assert(bytes.len % 4 == 0 and bytes.len <= 512);
            inline_len[index] = bytes.len / 4;
            @memcpy(std.mem.sliceAsBytes(&inline_values[index])[0..bytes.len], bytes);
            calls += 1;
        }
        pub fn dispatchGroups(_: ComputeEncoder, grid: Size, threads: Size) void {
            const want: usize = if (selected == 3) 1024 else if (selected == 1) 128 else 256;
            std.debug.assert(threads.width == want and threads.height == 1 and threads.depth == 1);
            grids[selected] = grid;
            const plan = family.attention_plan.Plan.init(geometry(), p) catch unreachable;
            switch (selected) {
                0 => {
                    std.debug.assert(inline_len[1] == p.rows);
                    std.debug.assert(std.mem.eql(i32, inline_values[1][0..p.rows], plan.positions[0..p.rows]));
                },
                1 => {
                    std.debug.assert(inline_values[1][0] == plan.first_pool);
                    std.debug.assert(binds[5].? == 13 * 0x1000000 + plan.first_pool * 256);
                },
                2 => {
                    std.debug.assert(inline_values[3][0] == 1024);
                    std.debug.assert(std.mem.eql(i32, inline_values[4][0..p.rows], plan.complete[0..p.rows]));
                },
                3 => {
                    std.debug.assert(inline_values[1][1] == 1024);
                    std.debug.assert(std.mem.eql(i32, inline_values[3][0..p.rows], plan.ends[0..p.rows]));
                },
                4 => {
                    std.debug.assert(copies == 1 + 2 * p.rows);
                    std.debug.assert(inline_values[2][1] == 4 and inline_values[2][2] == 4096);
                    std.debug.assert(inline_values[5][1] == 2052);
                    std.debug.assert(std.mem.eql(i32, inline_values[6][0..p.rows], plan.keys[0..p.rows]));
                    std.debug.assert(std.mem.eql(i32, inline_values[7][0..p.rows], plan.sparse[0..p.rows]));
                    saw_parts = true;
                },
                5 => std.debug.assert(saw_parts and binds[3].? == 18 * 0x1000000),
                else => unreachable,
            }
            calls += 1;
        }
        pub fn barrier(_: ComputeEncoder) void {
            calls += 1;
        }
    };
    pub const Session = struct {
        queue: Queue = .{},
        fail_copy: ?usize = null,
        pub fn encoder(_: *Session) !ComputeEncoder {
            return .{};
        }
        pub fn copy(s: *Session, to: Buffer, to_off: usize, from: Buffer, from_off: usize, n: usize) !void {
            if (s.fail_copy == copies) return error.PlantedCopy;
            std.debug.assert(selected == 0 and !saw_parts);
            if (copies == 0) {
                std.debug.assert(to.address == 9 * 0x1000000 and to_off == p.start * 2048);
                std.debug.assert(from.address == 8 * 0x1000000 and from_off == 0 and n == p.rows * 2048);
            } else {
                const r = (copies - 1) / 2;
                const value = copies % 2 == 1;
                std.debug.assert(to.address == @as(u64, if (value) 10 else 12) * 0x1000000);
                std.debug.assert(to_off == (p.start + r) * @as(usize, if (value) 2048 else 256));
                std.debug.assert(from.address == 0x1000000);
                std.debug.assert(from_off == r * 14976 * 2 + @as(usize, if (value) (48 + 4) * 256 * 2 else (56 * 256 + 512) * 2));
                std.debug.assert(n == @as(usize, if (value) 2048 else 256));
            }
            copies += 1;
            calls += 1;
        }
    };
};
fn geometry() qsa.Geometry {
    return .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 4, .top = 512 };
}
test "Qwen4 Metal QSA: causal metadata, completed pools and token-major scatter use the actual adapter" {
    const K = family.metal_attention.Kernels(R);
    var k = try K.load(t.allocator, .{}, geometry());
    for ([_]usize{ 0, 7, 2049 }) |start| {
        R.p = .{ .rows = 3, .start = start, .capacity = 4096, .eps = 1e-6 };
        const z = try family.metal_attention.sizes(geometry(), R.p);
        var v: [18]K.View = undefined;
        for (&v, z.bytes, 0..) |*view, n, i| view.* = .{ .buffer = .{ .address = (i + 1) * 0x1000000, .size = n }, .len = n };
        R.copies = 0;
        R.saw_parts = false;
        R.grids = @splat(null);
        var s: R.Session = .{};
        try k.encode(&s, R.p, v);
        try t.expectEqualDeep(R.Size.of(1, 32, 3), R.grids[0].?);
        try t.expectEqualDeep(R.Size.of(24, 3, 16), R.grids[4].?);
        try t.expectEqualDeep(R.Size.of(1, 24, 3), R.grids[5].?);
        const plan = try family.attention_plan.Plan.init(geometry(), R.p);
        try t.expectEqual(plan.pool_count > 0, R.grids[1] != null);
        if (R.grids[1]) |grid| try t.expectEqualDeep(R.Size.of(1, plan.pool_count, 1), grid);
        const before = R.calls;
        var wrong = v;
        wrong[17] = v[6];
        try t.expectError(error.Qwen4AttentionAlias, k.encode(&s, R.p, wrong));
        wrong = v;
        wrong[12].len -= 2;
        try t.expectError(error.Qwen4AttentionBuffer, k.encode(&s, R.p, wrong));
        s.queue.owner = 2;
        try t.expectError(error.Qwen4AttentionContext, k.encode(&s, R.p, v));
        try t.expectEqual(before, R.calls);
        s.queue.owner = 1;
        s.fail_copy = 1;
        R.copies = 0;
        R.saw_parts = false;
        try t.expectError(error.PlantedCopy, k.encode(&s, R.p, v));
        try t.expect(!R.saw_parts);
    }
    k.deinit();
    try t.expectEqual(@as(usize, 0), R.live);
    for (0..6) |i| {
        R.fail_at = i;
        try t.expectError(error.PlantedPipeline, K.load(t.allocator, .{}, geometry()));
        try t.expectEqual(@as(usize, 0), R.live);
    }
    R.fail_at = null;
}
