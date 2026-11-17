//! Startup-only group inventory. Private boot and device identities are used for grouping and never serialized.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

pub const max_ranks = 8;
pub const Wire = extern struct {
    rank: u32 = 0,
    world: u32 = 0,
    node: [32]u8 = @splat(0),
    device: [16]u8 = @splat(0),
    device_known: u32 = 0,
    cpu: [120]u8 = @splat(0),
    gpu: [120]u8 = @splat(0),
    reserved: u32 = 0,
    memory: u64 = 0,
    gpu_memory: u64 = 0,
};
pub const Gpu = struct { name: []const u8, memory_bytes: ?u64, cores: ?u64 = null };
pub const Hardware = struct {
    cpu_model: ?[]const u8,
    system_memory_bytes: ?u64,
    memory_type: []const u8,
    gpus: []const Gpu,
    gpu_count: usize,
    source: []const u8,
};
const Node = struct { id: usize, ranks: []const u32, hardware: Hardware };

fn plain(s: []const u8) bool {
    if (s.len == 0 or s.len > 120 or !std.ascii.isAlphanumeric(s[0])) return false;
    for (s) |c| if (!std.ascii.isAlphanumeric(c) and std.mem.indexOfScalar(u8, " ._()+-", c) == null) return false;
    return true;
}
fn put(dst: *[120]u8, s: []const u8) void {
    if (plain(s)) @memcpy(dst[0..s.len], s);
}
fn read(a: Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer f.close(io);
    var buf: [1024]u8 = undefined;
    var r = f.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}
fn sysctl(comptime key: [:0]const u8, out: []u8) ?[]const u8 {
    var n = out.len;
    if (std.c.sysctlbyname(key, out.ptr, &n, null, 0) != 0 or n == 0 or n > out.len) return null;
    return std.mem.trimEnd(u8, out[0..n], "\x00");
}

pub fn probe(a: Allocator, io: std.Io, rank: u32, world: u32, gpu: []const u8, gpu_memory: u64, device: ?[16]u8) Wire {
    var w: Wire = .{ .rank = rank, .world = world, .gpu_memory = gpu_memory };
    put(&w.gpu, gpu);
    if (device) |id| {
        w.device = id;
        w.device_known = 1;
    }
    w.memory = std.process.totalSystemMemory() catch 0;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const temp = arena.allocator();
    if (builtin.os.tag == .macos) {
        var buf: [256]u8 = undefined;
        if (sysctl("kern.bootsessionuuid", &buf)) |s| std.crypto.hash.sha2.Sha256.hash(s, &w.node, .{});
        if (sysctl("machdep.cpu.brand_string", &buf)) |s| put(&w.cpu, s);
    } else if (builtin.os.tag == .linux) {
        if (read(temp, io, "/proc/sys/kernel/random/boot_id")) |s| {
            const id = std.mem.trim(u8, s, " \r\n\t");
            if (id.len > 0) std.crypto.hash.sha2.Sha256.hash(id, &w.node, .{});
        }
        if (read(temp, io, "/proc/cpuinfo")) |s| {
            var lines = std.mem.splitScalar(u8, s, '\n');
            while (lines.next()) |line| {
                const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
                const key = std.mem.trim(u8, line[0..colon], " \t");
                if (!std.mem.eql(u8, key, "model name") and !std.mem.eql(u8, key, "Hardware")) continue;
                put(&w.cpu, std.mem.trim(u8, line[colon + 1 ..], " \t"));
                break;
            }
        }
    }
    return w;
}

/// The communicator reports its actual count and rank; every rank contributes once, before any serving thread.
pub fn cuda(a: Allocator, io: std.Io, ctx: anytype, group: anytype, stream: anytype) !?[]const u8 {
    const shape = try group.benchmarkShape();
    if (shape.world == 0 or shape.world > max_ranks) return null;
    var name: [256]u8 = undefined;
    const gpu = ctx.name(&name) catch "";
    const memory = if (ctx.memInfo()) |m| m.total else |_| 0;
    const id: ?[16]u8 = ctx.uuid() catch null;
    const mine = probe(a, io, shape.rank, shape.world, gpu, memory, id);
    var all: [max_ranks]Wire = undefined;
    try group.benchmarkGather(Wire, ctx.d, stream, mine, all[0..shape.world]);
    return assemble(a, all[0..shape.world]);
}

/// The connected endpoint attests two ranks; HELLO=1 peers retain serving compatibility but no inventory.
pub fn metal(a: Allocator, io: std.Io, device: anytype, rd: anytype, offset: usize, hello: usize, ready: u64) !?[]const u8 {
    return metalFor(a, io, device, rd, offset, hello, ready, null);
}

/// A bounded inventory wait for transports whose startup must fail when a connected peer goes silent.
pub fn metalFor(a: Allocator, io: std.Io, device: anytype, rd: anytype, offset: usize, hello: usize, ready: u64, comptime wait_ns: ?u64) !?[]const u8 {
    if (rd.size() != 2 or rd.rank() > 1) return error.BadWorld;
    const win = rd.window();
    if (offset > win.len or @sizeOf(Wire) > win.len - offset or hello > win.len - 8) return error.BadInventoryOffset;
    const rank = rd.rank();
    const memory = std.process.totalSystemMemory() catch 0;
    const mine = probe(a, io, rank, 2, std.mem.span(device.name()), memory, @as([16]u8, @splat(0)));
    try rd.write2Signal(1 - rank, offset, std.mem.asBytes(&mine), &.{}, hello, ready);
    try rd.flush();
    const word: *const u64 = @ptrCast(@alignCast(win[hello..][0..8]));
    const start: std.Io.Timestamp = if (wait_ns != null) .now(io, .awake) else .{ .nanoseconds = 0 };
    while (@atomicLoad(u64, word, .acquire) < ready and @atomicLoad(u64, word, .acquire) != 1) {
        if (wait_ns) |budget| if (start.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds >= budget)
            return error.PairPeerSilent;
        std.atomic.spinLoopHint();
    }
    if (@atomicLoad(u64, word, .acquire) != ready) return null;
    var all: [2]Wire = undefined;
    all[rank] = mine;
    all[1 - rank] = std.mem.bytesToValue(Wire, win[offset..][0..@sizeOf(Wire)]);
    return assemble(a, &all);
}

/// A missing probe, inconsistent cover or conflicting local hardware leaves the group unranked.
pub fn assemble(a: Allocator, all: []const Wire) !?[]const u8 {
    if (all.len == 0 or all.len > max_ranks) return null;
    var nodes: [max_ranks]Node = undefined;
    var nranks: [max_ranks][max_ranks]u32 = undefined;
    var ngpus: [max_ranks][max_ranks]Gpu = undefined;
    var keys: [max_ranks][32]u8 = undefined;
    var devices: [max_ranks][max_ranks][16]u8 = undefined;
    var count: usize = 0;
    for (all, 0..) |*w, rank| {
        if (w.rank != rank or w.world != all.len or std.mem.allEqual(u8, &w.node, 0) or w.device_known != 1 or w.reserved != 0) return null;
        const gpu = std.mem.sliceTo(&w.gpu, 0);
        const cpu = std.mem.sliceTo(&w.cpu, 0);
        if (!plain(gpu) or (cpu.len > 0 and !plain(cpu)) or w.gpu_memory == 0) return null;
        const at = for (keys[0..count], 0..) |key, i| {
            if (std.mem.eql(u8, &key, &w.node)) break i;
        } else blk: {
            const i = count;
            keys[i] = w.node;
            nodes[i] = .{ .id = i, .ranks = nranks[i][0..0], .hardware = .{
                .cpu_model = if (cpu.len > 0) cpu else null,
                .system_memory_bytes = if (w.memory > 0) w.memory else null,
                .memory_type = if (std.mem.startsWith(u8, gpu, "Apple M") or std.mem.eql(u8, gpu, "NVIDIA GB10")) "unified" else "dedicated",
                .gpus = ngpus[i][0..0],
                .gpu_count = 0,
                .source = if (cpu.len > 0 and w.memory > 0) "detected" else "partial",
            } };
            count += 1;
            break :blk i;
        };
        const node = &nodes[at];
        if (node.hardware.system_memory_bytes != (if (w.memory > 0) @as(?u64, w.memory) else null)) return null;
        if (!std.mem.eql(u8, node.hardware.cpu_model orelse "", cpu)) return null;
        const n = node.ranks.len;
        nranks[at][n] = @intCast(rank);
        node.ranks = nranks[at][0 .. n + 1];
        const device_at = for (devices[at][0..node.hardware.gpu_count], 0..) |key, i| {
            if (std.mem.eql(u8, &key, &w.device)) break i;
        } else node.hardware.gpu_count;
        if (device_at < node.hardware.gpu_count) {
            const prior = ngpus[at][device_at];
            if (!std.mem.eql(u8, prior.name, gpu) or prior.memory_bytes != w.gpu_memory) return null;
        } else {
            devices[at][device_at] = w.device;
            ngpus[at][device_at] = .{ .name = gpu, .memory_bytes = w.gpu_memory };
            node.hardware.gpu_count += 1;
            node.hardware.gpus = ngpus[at][0..node.hardware.gpu_count];
        }
    }
    return try std.json.Stringify.valueAlloc(a, .{
        .schema_version = @as(u8, 1),
        .source = "loaded-group",
        .rank_count = all.len,
        .category = if (count == 1) "single-node" else "multi-node",
        .nodes = nodes[0..count],
    }, .{});
}

test {
    _ = @import("benchmark_group_test.zig");
}
