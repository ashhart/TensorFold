//! The grouped expert launches (gx.zig) against the recorded 4/32 expert kernels on the same weights, logits and rows.
const std = @import("std");
const mtl = @import("metal");
const variants = @import("variants.zig");
const names = @import("fsr_contract_source.zig");
const gx = @import("gx.zig");
const pairs = @import("scale_pairs.zig");
const ks = @import("kernel_sources");
const Buf = @import("replay.zig").Buf;

pub const d = 2560;
pub const inner = 640;
pub const experts = 512;
pub const topk = 10;
/// Routed experts the fixture stores; its logits never pick another.
pub const held = 24;
const slots = topk + 1;
const at4 = [_]variants.Const{ .{ .name = "WB", .value = 4 }, .{ .name = "WG", .value = 32 }, .{ .name = "SWB", .value = 4 }, .{ .name = "SWG", .value = 32 } };

pub const Ref = struct { buffer: mtl.Buffer, size: usize };

/// The recorded gate/up or down kernel as the loader patches it before tensor units.
pub fn text(a: std.mem.Allocator, gate: bool) ![]const u8 {
    const file = if (gate) "qa_expert_gateup_fd30a642315017a8.metal" else "qa_expert_down_y_346a3e7c6008e1ac.metal";
    return variants.source(a, try names.recorded(file), &at4, true, false, gate, false);
}

/// fn_gx.metal as the loader compiles it; `indexed` with every scale and bias read through the index.
pub fn groupedText(a: std.mem.Allocator, indexed: bool) ![]const u8 {
    return if (indexed) @import("experts4.zig").indexedGrouped(a, ks.flashnext_gx) else ks.flashnext_gx;
}

/// Groups of 32 weights in projection j: gate, up, shared gate, shared up, down, shared down.
fn groups(j: usize) usize {
    const routed: usize = if (j == 0 or j == 1 or j == 4) held else 1;
    const each: usize = if (j < 4) inner * (d / 32) else d * (inner / 32);
    return routed * each;
}

pub const Fixture = struct {
    memory: std.heap.ArenaAllocator,
    a: std.mem.Allocator,
    queue: mtl.Queue,
    rows: usize,
    w: [18]Ref, // each projection's weight, scales and biases
    ix: [6][2]Ref, // each projection's index and table
    io: [7]Ref, // input rows, logits, row count, then act, picks, weights and y
    direct: [2]mtl.Pipeline, // the recorded gate/up and down
    grouped: [2]gx.Gx, // reading scales and biases direct, and through the index
    libraries: std.ArrayList(mtl.Library) = .empty,
    pipelines: std.ArrayList(mtl.Pipeline) = .empty,
    buffers: std.ArrayList(Ref) = .empty,
    stored: std.ArrayList([]const u8) = .empty,
    list: ?mtl.Buffer = null,
    wrong_scales: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, rows: usize, seed: u64, nonfinite: bool) !*Fixture {
        if (rows == 0 or rows > gx.rmax) return error.GroupedFixtureShape;
        const p = try a.create(Fixture);
        p.* = .{ .memory = .init(a), .a = undefined, .queue = queue, .rows = rows, .w = undefined, .ix = undefined, .io = undefined, .direct = undefined, .grouped = undefined };
        errdefer p.deinit(a);
        p.a = p.memory.allocator();
        var random: std.Random.DefaultPrng = .init(seed);
        const table = try p.a.create([pairs.most]u32);
        const seen = try p.a.create([pairs.slots]u32);
        for (0..6) |j| {
            const n = groups(j);
            const codes = try p.a.alloc(u8, n * 16);
            random.random().bytes(codes);
            const s = try p.a.alloc(u16, n);
            const b = try p.a.alloc(u16, n);
            for (s, b, 0..) |*sv, *bv, i| {
                sv.* = bf16(@as(f32, @floatFromInt(1 + (i + seed + j) % 5)) / 512);
                bv.* = bf16(-@as(f32, @floatFromInt((i + j) % 3)) / 512);
            }
            if (nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| {
                s[i * (n / 3)] = v;
            };
            p.w[j * 3] = try p.make(device, codes);
            p.w[j * 3 + 1] = try p.make(device, std.mem.sliceAsBytes(s));
            p.w[j * 3 + 2] = try p.make(device, std.mem.sliceAsBytes(b));
            const index = try p.a.alloc(u16, n);
            const count = try pairs.index(s, b, index, table, seen);
            p.ix[j] = .{ try p.make(device, std.mem.sliceAsBytes(index)), try p.make(device, std.mem.sliceAsBytes(try p.a.dupe(u32, table[0..count]))) };
        }
        const h = try p.a.alloc(u16, rows * d);
        for (h, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt((i * 7 + seed) % 23)) / 64 - 0.125);
        // six experts every row takes, the rest of a row's picks spread thin: tiles, single pairs and ties
        const lg = try p.a.alloc(f32, rows * (experts + 1));
        for (lg, 0..) |*v, i| {
            const e = i % (experts + 1);
            const row = i / (experts + 1);
            const spread: f32 = @floatFromInt((e * 13 + row * 29 + seed) % 17);
            v.* = if (e >= held) -1.0e30 else if (e < 6) 64 + spread else spread;
        }
        const nr = try p.a.dupe(u32, &.{@intCast(rows)});
        p.io[0] = try p.make(device, std.mem.sliceAsBytes(h));
        p.io[1] = try p.make(device, std.mem.sliceAsBytes(lg));
        p.io[2] = try p.make(device, std.mem.sliceAsBytes(nr));
        const out_sizes = [_]usize{ rows * slots * inner * 2, rows * topk * 4, rows * topk * 4, rows * slots * d * 2 };
        for (out_sizes, 3..) |size, k| p.io[k] = try p.make(device, try p.a.alloc(u8, size));
        for ([_]bool{ true, false }, 0..) |gate, k| {
            const lib = try mtl.Library.fromSource(device, try text(p.a, gate), mtl.CompileOptions.mlx());
            try p.libraries.append(p.a, lib);
            p.direct[k] = try mtl.Pipeline.init(device, lib, try std.fmt.allocPrintSentinel(p.a, "{s}", .{if (gate) names.gate else names.down}, 0), false);
            try p.pipelines.append(p.a, p.direct[k]);
        }
        p.list = try device.buffer(gx.list_bytes, mtl.ResourceOptions.shared);
        for ([_]bool{ false, true }, &p.grouped) |indexed, *g| {
            const lib = try gx.library(p.a, device, indexed);
            try p.libraries.append(p.a, lib);
            for (gx.functions, &g.pipes) |name, *pipe| {
                pipe.* = try mtl.Pipeline.init(device, lib, name, false);
                try p.pipelines.append(p.a, pipe.*);
            }
            g.buf = p.list.?;
            g.min_rows = 1;
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
        if (p.list) |b| b.deinit();
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

    fn buf(ref: Ref) Buf {
        return .{ .b = ref.buffer, .off = 64 };
    }

    /// Act, picks, weights and y after the recorded launches, or gx.zig's when `by_expert` (`indexed`: on the index).
    pub fn run(p: *Fixture, by_expert: bool, indexed: bool) ![]const u8 {
        for (p.io[3..]) |ref| @memset(p.bytes(ref), 0);
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        if (by_expert) {
            var w: [18]Buf = undefined;
            for (p.w, &w) |ref, *out| out.* = buf(ref);
            if (indexed) for (p.ix, 0..) |pair, j| {
                w[j * 3 + 1] = buf(pair[0]);
                w[j * 3 + 2] = buf(pair[1]);
            };
            const slot: usize = if (indexed) 2 else 1; // the control: gate and up scales (indexed: their tables) change places
            if (p.wrong_scales) std.mem.swap(Buf, &w[slot], &w[slot + 3]);
            p.grouped[@intFromBool(indexed)].launch(e, false, buf(p.io[0]), buf(p.io[1]), w, buf(p.io[3]), buf(p.io[4]), buf(p.io[5]), buf(p.io[2]), buf(p.io[6]), p.rows);
        } else {
            e.setPipeline(p.direct[0]);
            bind(e, 0, p.io[0]);
            bind(e, 1, p.io[1]);
            for (p.w[0..12], 2..) |ref, at| bind(e, at, ref);
            for (p.io[3..6], 14..) |ref, at| bind(e, at, ref);
            bind(e, 17, p.io[2]);
            e.dispatchGroups(mtl.Size.of(1, inner / 8, p.rows * slots), mtl.Size.of(64, 1, 1));
            e.setPipeline(p.direct[1]);
            bind(e, 0, p.io[3]);
            bind(e, 1, p.io[4]);
            for (p.w[12..], 2..) |ref, at| bind(e, at, ref);
            bind(e, 8, p.io[2]);
            bind(e, 9, p.io[6]);
            e.dispatchGroups(mtl.Size.of(1, d / 8, (p.rows * slots + 1) / 2), mtl.Size.of(64, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GroupedFixtureGpuFailure;
        return std.mem.concat(p.a, u8, &.{ p.bytes(p.io[3]), p.bytes(p.io[4]), p.bytes(p.io[5]), p.bytes(p.io[6]) });
    }

    /// Every input the kernels read, in one slice.
    pub fn inputs(p: *Fixture, a: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        for (p.buffers.items[0 .. p.buffers.items.len - 4]) |ref| try out.appendSlice(a, p.bytes(ref));
    }

    /// Guards whole and every input as stored.
    pub fn intact(p: *Fixture) bool {
        for (p.buffers.items, p.stored.items, 0..) |b, data, i| {
            const all = b.buffer.contents()[0 .. b.size + 128];
            for (all[0..64]) |v| if (v != 0xa5) return false;
            for (all[64 + b.size ..]) |v| if (v != 0xa5) return false;
            if (i + 4 < p.buffers.items.len and !std.mem.eql(u8, p.bytes(b), data)) return false;
        }
        return true;
    }
};
