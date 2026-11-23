//! The 4/32 prompt gathers reading scales and biases direct and through scale_pairs' index, on the same stored bytes.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const qmmw_source = @import("../../core/qmmw_source.zig");
const index_source = @import("prompt_index_source.zig");
const pairs = @import("scale_pairs.zig");

pub const d = 2560;
pub const inner = 640;
pub const experts = 8;
/// The gathers by tile size, then the fused gate/up gathers.
pub const functions = [_][:0]const u8{ "tf_gather_qmmw_nax_32", "tf_gather_qmmw_nax_64", "tf_gather_qmmw_nax_128", "tf_gather_guw_nax_32", "tf_gather_guw_nax_64" };
const tiles = [_]usize{ 32, 64, 128, 32, 64 };

pub const Ref = struct { buffer: mtl.Buffer, size: usize };

/// qmmw_nax.metal at 4/32 before nax.h is inlined; `indexed` as the loader patches it for the gathers.
pub fn text(a: std.mem.Allocator, indexed: bool) ![]const u8 {
    return qmmw_source.source(a, if (indexed) try index_source.patch(a, ks.flashnext_qmmw) else ks.flashnext_qmmw, 4, 32);
}

/// Projection j's outputs and inputs: gate and up [inner, d], down [d, inner].
fn shape(j: usize) [2]usize {
    return if (j < 2) .{ inner, d } else .{ d, inner };
}

pub const Fixture = struct {
    memory: std.heap.ArenaAllocator,
    a: std.mem.Allocator,
    queue: mtl.Queue,
    rows: usize,
    w: [9]Ref, // gate, up and down: weight, scales and biases
    ix: [3][2]Ref, // each projection's index and table
    io: [3]Ref, // input rows [rows, d], activation rows [rows, inner], each expert's first row
    out: [8]Ref, // gate by 32, 64 and 128-row tiles, gate/up by 32 and 64, down by 32, 64 and 128
    pipes: [2][functions.len]mtl.Pipeline, // direct, indexed
    libraries: std.ArrayList(mtl.Library) = .empty,
    pipelines: std.ArrayList(mtl.Pipeline) = .empty,
    buffers: std.ArrayList(Ref) = .empty,
    stored: std.ArrayList([]const u8) = .empty,
    wrong_entry: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, rows: usize, seed: u64, nonfinite: bool) !*Fixture {
        if (rows < experts or rows > 512) return error.PromptIndexFixtureShape;
        const p = try a.create(Fixture);
        p.* = .{ .memory = .init(a), .a = undefined, .queue = queue, .rows = rows, .w = undefined, .ix = undefined, .io = undefined, .out = undefined, .pipes = undefined };
        errdefer p.deinit(a);
        p.a = p.memory.allocator();
        var random: std.Random.DefaultPrng = .init(seed);
        const table = try p.a.create([pairs.most]u32);
        const seen = try p.a.create([pairs.slots]u32);
        for (0..3) |j| {
            const n = experts * shape(j)[0] * (shape(j)[1] / 32);
            const codes = try p.a.alloc(u8, n * 16);
            random.random().bytes(codes);
            const s = try p.a.alloc(u16, n);
            const b = try p.a.alloc(u16, n);
            for (s, b, 0..) |*sv, *bv, i| {
                sv.* = bf16(@as(f32, @floatFromInt(1 + (i * 13 + seed + j) % 97)) / 4096);
                bv.* = bf16(-@as(f32, @floatFromInt((i * 5 + seed) % 31)) / 4096);
            }
            if (nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| {
                s[i * (n / 3)] = v;
            };
            const index = try p.a.alloc(u16, n);
            const count = try pairs.index(s, b, index, table, seen);
            p.w[j * 3] = try p.make(device, codes);
            p.w[j * 3 + 1] = try p.make(device, std.mem.sliceAsBytes(s));
            p.w[j * 3 + 2] = try p.make(device, std.mem.sliceAsBytes(b));
            p.ix[j] = .{ try p.make(device, std.mem.sliceAsBytes(index)), try p.make(device, std.mem.sliceAsBytes(try p.a.dupe(u32, table[0..count]))) };
        }
        const x = try p.a.alloc(u16, rows * d);
        for (x, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt((i * 7 + seed) % 23)) / 64 - 0.125);
        const act = try p.a.alloc(u16, rows * inner);
        for (act, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt((i * 11 + seed) % 19)) / 32 - 0.25);
        // every expert's first row: expert 0 always has rows, one expert has none, the last takes what is left
        const first = try p.a.alloc(i32, experts);
        var at: usize = 0;
        for (first, 0..) |*v, e| {
            v.* = @intCast(at);
            const share = if (e == 3) 0 else 1 + (e * 5 + seed) % (2 * rows / experts);
            at = @min(rows, at + share);
        }
        p.io[0] = try p.make(device, std.mem.sliceAsBytes(x));
        p.io[1] = try p.make(device, std.mem.sliceAsBytes(act));
        p.io[2] = try p.make(device, std.mem.sliceAsBytes(first));
        for (&p.out, 0..) |*o, k| {
            const width: usize = if (k < 5) inner else d;
            o.* = try p.make(device, try p.a.alloc(u8, rows * width * 2));
        }
        for ([_]bool{ false, true }, 0..) |indexed, k| {
            const lib = try mtl.Library.fromSource(device, try frags.sourceMode(device, p.a, try text(p.a, indexed), false), mtl.CompileOptions.mlx());
            try p.libraries.append(p.a, lib);
            for (functions, &p.pipes[k]) |name, *pipe| {
                pipe.* = try mtl.Pipeline.init(device, lib, name, false);
                try p.pipelines.append(p.a, pipe.*);
            }
        }
        return p;
    }

    fn bf16(x: f32) u16 {
        const bits: u32 = @bitCast(x);
        return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }

    /// A guarded device buffer holding `data`, which the fixture keeps to prove the kernels left it alone.
    fn make(p: *Fixture, device: mtl.Device, data: []const u8) !Ref {
        const b = try device.buffer(data.len + 128, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        @memset(b.contents()[0 .. data.len + 128], 0xa5);
        const ref: Ref = .{ .buffer = b, .size = data.len };
        @memcpy(p.bytes(ref), data);
        try p.buffers.append(p.a, ref);
        try p.stored.append(p.a, data);
        return ref;
    }

    pub fn deinit(p: *Fixture, a: std.mem.Allocator) void {
        for (p.buffers.items) |b| b.buffer.deinit();
        for (p.pipelines.items) |pipe| pipe.deinit();
        for (p.libraries.items) |lib| lib.deinit();
        p.memory.deinit();
        a.destroy(p);
    }

    pub fn bytes(_: *const Fixture, ref: Ref) []u8 {
        return ref.buffer.contents()[64..][0..ref.size];
    }

    fn bind(e: mtl.ComputeEncoder, at: usize, ref: Ref) void {
        e.setBuffer(ref.buffer, 64, at);
    }

    /// Projection j's weight, scales and biases as a launch binds them: the index and table when `indexed`.
    fn weights(p: *const Fixture, j: usize, indexed: bool) [3]Ref {
        return if (indexed) .{ p.w[j * 3], p.ix[j][0], p.ix[j][1] } else .{ p.w[j * 3], p.w[j * 3 + 1], p.w[j * 3 + 2] };
    }

    /// Every gather's output in `out`'s order, as Prompt.gather and gatherGU launch them.
    pub fn run(p: *Fixture, indexed: bool) ![]const u8 {
        for (p.out) |ref| @memset(p.bytes(ref), 0);
        if (indexed and p.wrong_entry) p.swapPairs();
        defer if (indexed and p.wrong_entry) p.swapPairs();
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (p.out, 0..) |y, k| {
            const which = if (k < 5) k else k - 5; // the down gathers reuse the three plain gathers
            const fused = k == 3 or k == 4;
            const n: usize = if (k < 5) inner else d;
            const inputs_k: usize = if (k < 5) d else inner;
            const prm = [4]i32{ @intCast(p.rows), @intCast(n), @intCast(inputs_k), experts };
            e.setPipeline(p.pipes[@intFromBool(indexed)][which]);
            bind(e, 0, p.io[if (k < 5) 0 else 1]);
            var at: usize = 1;
            for (p.weights(if (k < 5) 0 else 2, indexed)) |ref| {
                bind(e, at, ref);
                at += 1;
            }
            if (fused) for (p.weights(1, indexed)) |ref| {
                bind(e, at, ref);
                at += 1;
            };
            bind(e, at, p.io[2]);
            e.setBytes(std.mem.asBytes(&prm), at + 1);
            bind(e, at + 2, y);
            e.dispatchThreads(mtl.Size.of(((n + 63) / 64) * 128, p.rows / tiles[which] + experts, 1), mtl.Size.of(128, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.PromptIndexFixtureGpuFailure;
        var all: std.ArrayList(u8) = .empty;
        for (p.out) |ref| try all.appendSlice(p.a, p.bytes(ref));
        return all.items;
    }

    /// The control: the gate's first two table pairs change places.
    fn swapPairs(p: *Fixture) void {
        const table = p.bytes(p.ix[0][1]);
        var pair: [4]u8 = undefined;
        @memcpy(&pair, table[0..4]);
        @memcpy(table[0..4], table[4..8]);
        @memcpy(table[4..8], &pair);
    }

    /// Every input the kernels read, in one slice.
    pub fn inputs(p: *Fixture, a: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        for (p.buffers.items[0 .. p.buffers.items.len - p.out.len]) |ref| try out.appendSlice(a, p.bytes(ref));
    }

    /// Guards whole and every input as stored.
    pub fn intact(p: *Fixture) bool {
        for (p.buffers.items, p.stored.items, 0..) |b, data, i| {
            const all = b.buffer.contents()[0 .. b.size + 128];
            for (all[0..64]) |v| if (v != 0xa5) return false;
            for (all[64 + b.size ..]) |v| if (v != 0xa5) return false;
            if (i + p.out.len < p.buffers.items.len and !std.mem.eql(u8, p.bytes(b), data)) return false;
        }
        return true;
    }
};
