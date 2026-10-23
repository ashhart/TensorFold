//! Hash the six real rotary consumers against the precise baseline, including an intentional cache poison control.
const std = @import("std");
const mtl = @import("metal");
const family = @import("tensorfold");
const f = family.flashnext_vision_contract;
const cache = family.rotary_cache;
const rig_ = @import("rig.zig");
const prep = @import("metal_flashnext.zig");
const pool = @import("metal_flashnext_pool.zig");

const Row = struct {
    role: f.Which,
    factor: f64,
    long_positions: bool,
    rows: usize,
    positions: usize,
    baseline: [64]u8,
    cached: [64]u8,
    fallback: [64]u8,
    baseline_source: [64]u8,
    cached_source: [64]u8,
    poison_detected: bool,
};

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

const Prepare = struct {
    pipe: mtl.Pipeline,
    table: rig_.Ref,
    positions: rig_.Ref,
    count: usize,
    reset: bool = true,
    fn run(ptr: *anyopaque, enc: mtl.ComputeEncoder) void {
        const p: *Prepare = @ptrCast(@alignCast(ptr));
        var at: usize = 0;
        while (at < p.count) : (at += cache.max_positions) {
            const n = @min(cache.max_positions, p.count - at);
            cache.encode(enc, p.pipe, .{ .b = p.table.buf, .off = p.table.off }, .{ .b = p.positions.buf, .off = p.positions.off + at * 4 }, n, p.reset and at == 0);
        }
    }
};

fn check(comptime Probe: type, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, which: f.Which, rows: usize, factor: f64, long: bool) !Row {
    var p = try Probe.init(a, device, queue, which);
    defer p.deinit();
    p.factor = factor;
    p.long_positions = long;
    const pooled = comptime Probe == pool.Probe;
    const n = if (pooled) 128 else f.output_width;
    const k: usize = if (pooled) 128 else 256;
    if (!try p.prepare(.{ .rows = rows, .n = n, .k = k }, .random, 17)) return error.BadProbeShape;
    try p.run();
    const baseline = hash(try p.rig.outputBytes());
    const original = f.source(which);
    const changed = try cache.source(a, original, true);
    const compiled = try rig_.pipeline(device, changed, f.symbol(which));
    p.pipe.deinit();
    p.pipe = compiled;
    const table = try p.rig.scratch(cache.bytes);
    const bytes = p.rig.bytes(table, cache.bytes);
    @memset(bytes, 0);
    const prefix = p.rig.bytes(p.base, cache.prefix * 4);
    @memcpy(bytes[0 .. cache.prefix * 4], prefix);
    const prefix_hash = hash(prefix);
    p.base = table;
    var positions: []const i32 = undefined;
    var pos: rig_.Ref = undefined;
    if (comptime pooled) {
        const values = try @import("rotary_cache_positions.zig").pooled(a, p.firstBlock(), rows, if (which == .positioned) p.blocks else null);
        positions = values;
        pos = try p.rig.input(std.mem.sliceAsBytes(values));
    } else {
        positions = p.positions;
        pos = p.pos;
    }
    var before: Prepare = .{ .pipe = pipe, .table = table, .positions = pos, .count = positions.len };
    p.before = .{ .ctx = &before, .call = Prepare.run };
    const inputs = hash(try p.rig.inputBytes());
    try p.run();
    const cached = hash(try p.rig.outputBytes());
    if (!std.mem.eql(u8, &baseline, &cached)) return error.RotaryBitsChanged;
    if (!std.mem.eql(u8, &prefix_hash, &hash(bytes[0 .. cache.prefix * 4]))) return error.RotaryConstantsChanged;
    for (positions) |position| {
        var found = false;
        for (0..cache.slots) |slot| {
            const off = (cache.keys + slot) * 4;
            if (std.mem.readInt(i32, bytes[off..][0..4], .little) == position) found = true;
        }
        if (!found) return error.PositionNotCached;
    }
    for (0..cache.slots) |slot| {
        const off = (cache.keys + slot) * 4;
        if (std.mem.readInt(i32, bytes[off..][0..4], .little) < 0) continue;
        for (0..32) |i| std.mem.writeInt(u32, bytes[(cache.data + slot * 128 + i * 4) * 4 ..][0..4], 0x7fc00000, .little);
    }
    const saved = try a.dupe(u8, bytes);
    before.reset = false;
    try p.run();
    if (!std.mem.eql(u8, saved, bytes)) return error.ExistingPositionRecomputed;
    p.before = null;
    for (cache.data..cache.words) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], 0x7fc00000, .little);
    try p.run();
    if (std.mem.eql(u8, &baseline, &hash(try p.rig.outputBytes()))) return error.CachePoisonUndetected;
    std.mem.writeInt(u32, bytes[cache.prefix * 4 ..][0..4], 0, .little);
    try p.run();
    const fallback = hash(try p.rig.outputBytes());
    if (!std.mem.eql(u8, &baseline, &fallback)) return error.PreciseFallbackChanged;
    if (!try p.rig.intact()) return error.BufferGuardChanged;
    if (!std.mem.eql(u8, &inputs, &hash(try p.rig.inputBytes()))) return error.ReadOnlyInputChanged;
    return .{ .role = which, .factor = factor, .long_positions = long, .rows = rows, .positions = positions.len, .baseline = baseline, .cached = cached, .fallback = fallback, .baseline_source = hash(original), .cached_source = hash(changed), .poison_detected = true };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 or !std.mem.eql(u8, args[1], "--out")) return error.Usage;
    const objc_pool = mtl.objc.Pool.push();
    defer objc_pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const producer = try cache.producer(a);
    const pipe = try rig_.pipeline(device, producer, "tf_rotary_prepare");
    defer pipe.deinit();
    if (pipe.simdWidth() != 32 or pipe.maxThreads() < 128) return error.UnsupportedCachePipeline;
    var rows: std.ArrayList(Row) = .empty;
    for ([_]f.Which{ .target, .mtp, .pooled, .absolute, .mrope, .positioned }) |which| for ([_]usize{ 1, 4, 8, 16 }) |width| for ([_]f64{ 0, 4 }) |factor| for ([_]bool{ false, true }) |long| {
        const case_pool = mtl.objc.Pool.push();
        defer case_pool.pop();
        var scratch = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer scratch.deinit();
        const mem = scratch.allocator();
        const result = if (f.isPool(which)) try check(pool.Probe, mem, device, queue, pipe, which, width, factor, long) else try check(prep.Probe, mem, device, queue, pipe, which, width, factor, long);
        try rows.append(a, result);
    };
    if (rows.items.len != 96) return error.IncompleteRotaryCacheCheck;
    const identity = try @import("rotary_deployment.zig").current(a, init.io, device);
    const host_sources = try std.mem.concat(a, u8, &.{ cache.host_source, @embedFile("rotary_cache_main.zig"), @embedFile("rotary_cache_positions.zig"), @embedFile("metal_flashnext.zig"), @embedFile("metal_flashnext_pool.zig"), @embedFile("rig.zig") });
    const receipt = .{ .complete = true, .identity = identity, .tensor_units = device.tensorUnits(), .host_sources = hash(host_sources), .producer_source = hash(producer), .cache_bytes = cache.bytes, .rows = rows.items };
    const encoded = try std.json.Stringify.valueAlloc(a, receipt, .{ .whitespace = .indent_2 });
    const temporary = try std.fmt.allocPrint(a, "{s}.part-{d}", .{ args[2], @import("sys").c.getpid() });
    const dir = std.Io.Dir.cwd();
    try dir.writeFile(init.io, .{ .sub_path = temporary, .data = encoded, .flags = .{ .exclusive = true } });
    errdefer dir.deleteFile(init.io, temporary) catch {};
    try dir.rename(temporary, dir, args[2], init.io);
}
