//! A two-layer nonzero MoE gate over the real expert shaders and the same cache, reader and shared-map owner.
const std = @import("std");
const mtl = @import("metal");
const f = @import("fsr_fixture_source.zig");
const src = @import("fsr_contract_source.zig");
const arena_mod = @import("../../core/fsr/arena.zig");
const File = @import("../../core/fsr/file.zig").File;
pub const NativeFile = File;
const Ref = struct { buffer: mtl.Buffer, size: usize, off: usize = 64 };
pub const Source = struct {
    bytes: []const u8,
    file: ?File = null,
    pub fn fileSize(s: Source, _: usize) !u64 {
        return if (s.file) |file| file.fileSize(0) else s.bytes.len;
    }
    pub fn readAt(s: Source, _: usize, out: []u8, at: u64) !usize {
        if (s.file) |file| return file.readAt(0, out, at);
        @memcpy(out, s.bytes[@intCast(at)..][0..out.len]);
        return out.len;
    }
};
pub const Fixture = struct {
    memory: std.heap.ArenaAllocator,
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    rows: usize,
    data: []u8,
    initial: []u16,
    sizes: [9]usize,
    full: [f.layers][9]Ref,
    pool: [9]Ref,
    map: Ref,
    strides: Ref,
    cache: arena_mod.Arena(Source),
    router_weights: [f.layers]Ref,
    h: Ref,
    lg: Ref,
    forecast_lg: Ref,
    ids: Ref,
    forecast_ids: Ref,
    act: Ref,
    picks: Ref,
    wts: Ref,
    y: Ref,
    row_count: Ref,
    logits: Ref,
    resident: [2]mtl.Pipeline,
    slotted: [2]mtl.Pipeline,
    glue: [4]mtl.Pipeline,
    libraries: std.ArrayList(mtl.Library) = .empty,
    pipelines: std.ArrayList(mtl.Pipeline) = .empty,
    buffers: std.ArrayList(Ref) = .empty,
    wrong_map: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, rows: usize, bits: usize, seed: u64, nonfinite: bool) !*Fixture {
        if (rows == 0 or rows > 8 or (bits != 4 and bits != 6)) return error.FsrFixtureShape;
        const p = try a.create(Fixture);
        p.* = undefined;
        p.memory = .init(a);
        errdefer {
            p.memory.deinit();
            a.destroy(p);
        }
        p.a = p.memory.allocator();
        p.device = device;
        p.queue = queue;
        p.rows = rows;
        p.buffers = .empty;
        p.libraries = .empty;
        p.pipelines = .empty;
        p.wrong_map = false;
        errdefer {
            for (p.buffers.items) |b| b.buffer.deinit();
            for (p.pipelines.items) |pipe| pipe.deinit();
            for (p.libraries.items) |l| l.deinit();
        }
        p.sizes = .{ f.inner * f.d * bits / 8, f.inner * (f.d / 32) * 2, f.inner * (f.d / 32) * 2, f.inner * f.d * bits / 8, f.inner * (f.d / 32) * 2, f.inner * (f.d / 32) * 2, f.d * f.inner * bits / 8, f.d * (f.inner / 32) * 2, f.d * (f.inner / 32) * 2 };
        var layer_bytes: usize = 0;
        for (p.sizes) |n| layer_bytes += (f.experts + 1) * n;
        p.data = try p.a.alloc(u8, layer_bytes * f.layers);
        var random: std.Random.DefaultPrng = .init(seed);
        random.random().bytes(p.data);
        const descriptors = try p.a.alloc(arena_mod.Expert, f.layers * f.experts);
        var at: usize = 0;
        for (0..f.layers) |layer| for (p.sizes, 0..) |size, part| {
            const span = p.data[at..][0 .. (f.experts + 1) * size];
            if (part % 3 != 0) {
                const words = @as([*]u16, @ptrCast(@alignCast(span.ptr)))[0 .. span.len / 2];
                for (words, 0..) |*word, i| word.* = bf16(if (part % 3 == 1) @as(f32, @floatFromInt(1 + i % 5)) / 512 else -@as(f32, @floatFromInt(i % 3)) / 512);
                if (nonfinite and part % 3 == 1) for (0..f.experts + 1) |expert| {
                    words[expert * size / 2] = switch (expert % 3) {
                        0 => 0x7fc0,
                        1 => 0x7f80,
                        else => 0xff80,
                    };
                };
            }
            p.full[layer][part] = try p.makeBuffer(span.len, span);
            for (0..f.experts) |e| descriptors[layer * f.experts + e][part] = .{ .file = 0, .offset = at + e * size, .length = size };
            at += span.len;
        };
        const capacity = rows * f.topk;
        var slabs: [9][]u8 = undefined;
        for (&p.pool, &slabs, p.sizes) |*buffer, *slab, size| {
            buffer.* = try p.makeBuffer(capacity * size, null);
            slab.* = p.bytes(buffer.*);
        }
        p.map = try p.makeBuffer(f.layers * f.experts * 4, null);
        p.strides = try p.makeBuffer(9 * 4, null);
        const strides = @as([*]u32, @ptrCast(@alignCast(p.bytes(p.strides).ptr)))[0..9];
        for (strides, p.sizes, 0..) |*word, size, i| word.* = @intCast(size / @as(usize, if (i % 3 == 0) 4 else 2));
        p.cache = try arena_mod.Arena(Source).initMapped(p.a, .{ .bytes = p.data }, descriptors, p.sizes, slabs, @as([*]i32, @ptrCast(@alignCast(p.bytes(p.map).ptr)))[0 .. f.layers * f.experts]);
        errdefer p.cache.deinit();
        p.initial = try p.a.alloc(u16, rows * f.d);
        for (p.initial, 0..) |*h, i| h.* = bf16(@as(f32, @floatFromInt((i + seed) % 23)) / 64 - 0.125);
        p.h = try p.makeBuffer(rows * f.d * 2, null);
        p.lg = try p.makeBuffer(rows * (f.experts + 1) * 4, null);
        p.forecast_lg = try p.makeBuffer(p.lg.size, null);
        p.ids = try p.makeBuffer(rows * f.topk * 4, null);
        p.forecast_ids = try p.makeBuffer(p.ids.size, null);
        p.act = try p.makeBuffer(rows * (f.topk + 1) * f.inner * 2, null);
        p.picks = try p.makeBuffer(p.ids.size, null);
        p.wts = try p.makeBuffer(p.ids.size, null);
        p.y = try p.makeBuffer(rows * (f.topk + 1) * f.d * 2, null);
        const nr = [_]u32{@intCast(rows)};
        p.row_count = try p.makeBuffer(4, std.mem.asBytes(&nr));
        p.logits = try p.makeBuffer(rows * f.logits * 2, null);
        for (&p.router_weights, 0..) |*buffer, layer| {
            const host = try p.a.alloc(u16, (f.experts + 1) * f.d);
            for (host, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt((i * 7 + layer * 11 + seed) % 29)) / 64 - 0.1875);
            buffer.* = try p.makeBuffer(host.len * 2, std.mem.sliceAsBytes(host));
        }
        for ([_]bool{ true, false }, 0..) |gate, i| {
            p.resident[i] = try p.pipeline(try f.expert(p.a, gate, bits, false), if (gate) src.gate else src.down);
            p.slotted[i] = try p.pipeline(try f.expert(p.a, gate, bits, true), if (gate) src.gate else src.down);
        }
        const glue = try f.glue(p.a);
        for (&p.glue, [_][]const u8{ "fsr_fixture_router", "fsr_fixture_route", "fsr_fixture_combine", "fsr_fixture_head" }) |*pipe, name|
            pipe.* = try p.pipeline(glue, name);
        return p;
    }
    fn bf16(x: f32) u16 {
        const bits: u32 = @bitCast(x);
        return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }
    fn makeBuffer(p: *Fixture, size: usize, data: ?[]const u8) !Ref {
        const b = try p.device.buffer(size + 128, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        @memset(b.contents()[0 .. size + 128], 0xa5);
        const ref: Ref = .{ .buffer = b, .size = size };
        if (data) |bytes_| @memcpy(p.bytes(ref), bytes_);
        try p.buffers.append(p.a, ref);
        return ref;
    }
    fn pipeline(p: *Fixture, text: []const u8, name: []const u8) !mtl.Pipeline {
        const lib = try mtl.Library.fromSource(p.device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        try p.libraries.append(p.a, lib);
        const pipe = try mtl.Pipeline.init(p.device, lib, try std.fmt.allocPrintSentinel(p.a, "{s}", .{name}, 0), false);
        errdefer pipe.deinit();
        try p.pipelines.append(p.a, pipe);
        return pipe;
    }
    pub fn deinit(p: *Fixture, a: std.mem.Allocator) void {
        p.cache.deinit();
        for (p.pipelines.items) |pipe| pipe.deinit();
        for (p.libraries.items) |lib| lib.deinit();
        for (p.buffers.items) |buffer_| buffer_.buffer.deinit();
        p.memory.deinit();
        a.destroy(p);
    }
    pub fn bytes(_: *const Fixture, ref: Ref) []u8 {
        return ref.buffer.contents()[ref.off..][0..ref.size];
    }
    fn bind(e: mtl.ComputeEncoder, at: usize, ref: Ref) void {
        e.setBuffer(ref.buffer, ref.off, at);
    }
    fn router(p: *Fixture, e: mtl.ComputeEncoder, layer: usize, lg: Ref, ids: Ref) void {
        e.setPipeline(p.glue[0]);
        bind(e, 0, p.h);
        bind(e, 1, p.router_weights[layer]);
        bind(e, 2, lg);
        e.dispatchThreads(mtl.Size.of(f.experts + 1, p.rows, 1), mtl.Size.of(32, 1, 1));
        e.setPipeline(p.glue[1]);
        bind(e, 0, lg);
        bind(e, 1, ids);
        e.dispatchGroups(mtl.Size.of(p.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }
    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.FsrFixtureGpuFailure;
    }
    fn experts(p: *Fixture, e: mtl.ComputeEncoder, layer: usize, streamed: bool) void {
        var weights: [18]Ref = undefined;
        for ([_]usize{ 0, 1, 2, 3, 4, 5 }) |proj| for (0..3) |part| {
            const routed = proj == 0 or proj == 1 or proj == 4;
            const component = (if (proj >= 4) @as(usize, 2) else proj % 2) * 3 + part;
            var ref = if (streamed and routed) p.pool[component] else p.full[layer][component];
            if (!routed) {
                ref.off += f.experts * p.sizes[component];
                ref.size = p.sizes[component];
            }
            weights[proj * 3 + part] = ref;
        };
        e.setPipeline(if (streamed) p.slotted[0] else p.resident[0]);
        bind(e, 0, p.h);
        bind(e, 1, p.lg);
        for (weights[0..12], 2..) |weight, at| bind(e, at, weight);
        bind(e, 14, p.act);
        bind(e, 15, p.picks);
        bind(e, 16, p.wts);
        var map = p.map;
        map.off += layer * f.experts * 4;
        if (streamed) {
            bind(e, 18, map);
            bind(e, 19, p.strides);
        }
        e.dispatchGroups(mtl.Size.of(1, f.inner / 8, p.rows * 3), mtl.Size.of(64, 1, 1));
        e.setPipeline(if (streamed) p.slotted[1] else p.resident[1]);
        bind(e, 0, p.act);
        bind(e, 1, p.picks);
        for (weights[12..], 2..) |weight, at| bind(e, at, weight);
        bind(e, 8, p.row_count);
        bind(e, 9, p.y);
        if (streamed) {
            bind(e, 10, map);
            bind(e, 11, p.strides);
        }
        e.dispatchGroups(mtl.Size.of(1, f.d / 8, (p.rows * 3 + 1) / 2), mtl.Size.of(64, 1, 1));
    }
    pub fn run(p: *Fixture, streamed: bool) ![]const u8 {
        @memcpy(p.bytes(p.h), std.mem.sliceAsBytes(p.initial));
        var observations: std.ArrayList(u8) = .empty;
        for (0..f.layers) |layer| {
            const cb = p.queue.commandBuffer();
            const e = cb.compute(.serial);
            p.router(e, layer, p.lg, p.ids);
            if (streamed and layer + 1 < f.layers) p.router(e, layer + 1, p.forecast_lg, p.forecast_ids);
            try finish(cb, e);
            if (streamed) {
                const ids = @as([*]const u32, @ptrCast(@alignCast(p.bytes(p.ids).ptr)))[0 .. p.rows * f.topk];
                var keys: [16]usize = undefined;
                for (ids, keys[0..ids.len]) |id, *key| {
                    if (id >= f.experts) return error.InvalidExpert;
                    key.* = layer * f.experts + id;
                }
                try p.cache.demand(keys[0..ids.len]);
                if (p.wrong_map) for (keys[0..ids.len]) |key| {
                    p.cache.clock.map[key] = @intCast((@as(usize, @intCast(p.cache.clock.map[key])) + 1) % p.cache.clock.slots.len);
                };
            }
            const compute = p.queue.commandBuffer();
            const enc = compute.compute(.serial);
            p.experts(enc, layer, streamed);
            enc.setPipeline(p.glue[2]);
            bind(enc, 0, p.h);
            bind(enc, 1, p.y);
            bind(enc, 2, p.wts);
            bind(enc, 3, p.lg);
            enc.dispatchThreads(mtl.Size.of(f.d, p.rows, 1), mtl.Size.of(64, 1, 1));
            enc.end();
            compute.commit();
            if (streamed and layer + 1 < f.layers) {
                const predictions = @as([*]const u32, @ptrCast(@alignCast(p.bytes(p.forecast_ids).ptr)))[0 .. p.rows * f.topk];
                var keys: [16]usize = undefined;
                for (predictions, keys[0..predictions.len]) |id, *key| {
                    if (id >= f.experts) return error.InvalidExpert;
                    key.* = (layer + 1) * f.experts + id;
                }
                p.cache.forecast(keys[0..predictions.len]) catch |err| {
                    compute.wait();
                    return err;
                };
            }
            compute.wait();
            if (compute.failure() != null) return error.FsrFixtureGpuFailure;
            if (p.wrong_map) for (p.cache.clock.slots, 0..) |slot, i| {
                if (slot.ready) p.cache.clock.map[slot.key.?] = @intCast(i);
            };
            for ([_]Ref{ p.act, p.picks, p.wts, p.y, p.h }) |ref| try observations.appendSlice(p.a, p.bytes(ref));
        }
        const cb = p.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p.glue[3]);
        bind(e, 0, p.h);
        bind(e, 1, p.logits);
        e.dispatchThreads(mtl.Size.of(f.logits, p.rows, 1), mtl.Size.of(16, 1, 1));
        try finish(cb, e);
        try observations.appendSlice(p.a, p.bytes(p.logits));
        return observations.items;
    }
    pub fn intact(p: *Fixture) bool {
        for (p.buffers.items) |b| {
            const all = b.buffer.contents()[0 .. b.size + 128];
            for (all[0..64]) |v| if (v != 0xa5) return false;
            for (all[64 + b.size ..]) |v| if (v != 0xa5) return false;
        }
        for (p.full, 0..) |layer, li| {
            var at: usize = 0;
            for (p.sizes) |n| at += (f.experts + 1) * n;
            at *= li;
            for (layer) |b| {
                if (!std.mem.eql(u8, p.bytes(b), p.data[at..][0..b.size])) return false;
                at += b.size;
            }
        }
        return true;
    }
};
