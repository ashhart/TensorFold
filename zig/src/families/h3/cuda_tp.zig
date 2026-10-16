//! The H3 CUDA family across ranks: each rank runs the rows of its own tiles and hands the others its keys and values.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Model = h3.Model;
const head_dim = h3.head_dim;
const slot_tile = h3.slot_tile;

// the rank-to-rank exchange, h3_xchg.c over libibverbs
const Xchg = opaque {};
pub const XchgInfo = [72]u8;
extern fn h3x_create(device: [*:0]const u8, gid_index: c_int, region: *anyopaque, len: u64, world: c_int, rank: c_int, err: [*]u8, err_len: u64) ?*Xchg;
extern fn h3x_destroy(x: ?*Xchg) void;
extern fn h3x_error(x: *Xchg) [*:0]const u8;
extern fn h3x_info_for(x: *Xchg, peer: c_int, out: *XchgInfo) c_int;
extern fn h3x_connect(x: *Xchg, peer: c_int, in: *const XchgInfo) c_int;
extern fn h3x_post(x: *Xchg, peer: c_int, off: [*]const u64, len: [*]const u64, n: c_int, seq: u32) c_int;
extern fn h3x_wait(x: *Xchg, seq: u32, seconds: f64) c_int;

/// A rank of a sequence-parallel world and its registered region.
pub const Tp = struct {
    x: *Xchg,
    /// Pinned and mapped for the device: the GPU packs into it and the NIC reads and writes it.
    region: cuda.HostBuffer,
    dev: u64,
    world: u32,
    rank: u32,
    connected: u32 = 0,
    seq: u32 = 0,
    /// The last forward's seconds in the exchange (packing, posting and waiting), and how many exchanges it made.
    waited: f64 = 0,
    swaps: u32 = 0,
    /// Of `waited`: seconds packing this rank's tiles into the region.
    packed_seconds: f64 = 0,
};

/// Tiles [a, b)'s keys, values, scales (and means) into the region's half `parity` with `pack`, out of it otherwise.
fn ship(self: *Model, parity: usize, a: usize, b: usize, pooled: bool, pack: bool) !void {
    const g = &self.geometry;
    const t = &self.tp.?;
    const n = b - a;
    if (n == 0) return;
    const slots = g.tiles * slot_tile;
    const tile_bytes = slot_tile * head_dim;
    // device address of head 0's first byte, bytes from one head to the next, bytes a head
    const parts = [7][3]u64{
        .{ g.tk[0] + a * tile_bytes, slots * head_dim, n * tile_bytes },
        .{ g.tvt[0] + a * tile_bytes, g.tiles * tile_bytes, n * tile_bytes },
        .{ g.ks[0] + a * 4, g.tiles * 4, n * 4 },
        .{ g.vs[0] + a * 4, g.tiles * 4, n * 4 },
        .{ g.qp[0] + a * head_dim * 4, g.tiles * head_dim * 4, n * head_dim * 4 },
        .{ g.kp[0] + a * head_dim * 4, g.tiles * head_dim * 4, n * head_dim * 4 },
        .{ g.vp[0] + a * head_dim * 4, g.tiles * head_dim * 4, n * head_dim * 4 },
    };
    var at: u64 = t.dev + parity * g.half + a * g.unit;
    for (parts[0..if (pooled) 7 else 4]) |part| {
        var args: cuda.Args = .{};
        args.add(if (pack) at else part[0]);
        args.add(if (pack) part[0] else at);
        args.add(@as(c_longlong, @intCast(if (pack) part[2] else part[1])));
        args.add(@as(c_longlong, @intCast(if (pack) part[1] else part[2])));
        args.add(@as(c_int, @intCast(part[2] / 4)));
        try self.launch("h3_spans", .{ .x = @intCast((part[2] / 4 + 16383) / 16384), .y = @intCast(self.heads) }, 256, 0, &args);
        at += self.heads * part[2];
    }
}

/// One exchange of this rank's tiles for every other rank's; rank r owns tiles [cuts[r], cuts[r + 1]).
pub fn swap(self: *Model, parity: usize, pooled: bool, cuts: *const [9]u32) !void {
    const g = &self.geometry;
    const t = &self.tp.?;
    // the kernels before this finish before the clock starts: their time is not the exchange's
    try self.stream.synchronize();
    const started = h3.seconds();
    try ship(self, parity, g.own0, g.own1, pooled, true);
    try self.stream.synchronize();
    t.packed_seconds += h3.seconds() - started;
    t.seq +%= 1;
    if (std.c.getenv("TF_H3_TP_SOLO") == null) {
        const off = [1]u64{parity * g.half + g.own0 * g.unit};
        const len = [1]u64{(g.own1 - g.own0) * (if (pooled) g.unit else self.heads * (2 * slot_tile * head_dim + 8))};
        for (0..t.world) |peer| {
            if (peer == t.rank) continue;
            if (h3x_post(t.x, @intCast(peer), &off, &len, 1, t.seq) != 0) {
                std.debug.print("tensorfold h3 rank {d}: exchange with rank {d}: {s}\n", .{ t.rank, peer, std.mem.span(h3x_error(t.x)) });
                return error.ExchangeFailed;
            }
        }
        if (h3x_wait(t.x, t.seq, 120.0) != 0) {
            std.debug.print("tensorfold h3 rank {d}: exchange: {s}\n", .{ t.rank, std.mem.span(h3x_error(t.x)) });
            return error.ExchangeFailed;
        }
        for (0..t.world) |source| {
            if (source != t.rank) try ship(self, parity, cuts[source], cuts[source + 1], pooled, false);
        }
    } else {
        // a rank alone for a bench: unpack as many tiles as the others would have sent (this rank's own)
        const others = g.tiles - (g.own1 - g.own0);
        var done: usize = 0;
        while (done < others) : (done += g.own1 - g.own0) try ship(self, parity, g.own0, g.own0 + @min(g.own1 - g.own0, others - done), pooled, false);
    }
    if (self.profile) try self.stream.synchronize();
    const took = h3.seconds() - started;
    t.waited += took;
    t.swaps += 1;
    if (self.profile) self.spent[h3.stages.len - 1] += took;
}

/// Joins a world of `world` ranks as `rank` over RDMA `device` at RoCE v2 GID `gid_index` with a region of that size.
pub fn init(self: *Model, device: [*:0]const u8, gid_index: c_int, world: u32, rank: u32, region_bytes: usize) !void {
    try self.ctx.makeCurrent();
    if (self.tp != null or world < 2 or world > 8 or rank >= world) return error.BadWorld;
    h3.geometry.release(self);
    var region = try cuda.HostBuffer.allocMapped(&self.driver, region_bytes);
    errdefer region.free();
    var err: [256]u8 = @splat(0);
    const x = h3x_create(device, gid_index, region.bytes.ptr, region_bytes, @intCast(world), @intCast(rank), &err, err.len) orelse {
        std.debug.print("tensorfold h3 rank {d}: {s}\n", .{ rank, std.mem.sliceTo(&err, 0) });
        return error.ExchangeFailed;
    };
    self.tp = .{ .x = x, .region = region, .dev = try region.device(), .world = world, .rank = rank };
}

/// Leaves the world: the queue pairs and the region go, and the next forward is one GPU's again.
pub fn close(self: *Model) void {
    self.ctx.makeCurrent() catch return;
    _ = self.driver.api.cuCtxSynchronize();
    h3.geometry.release(self);
    if (self.tp) |*t| {
        h3x_destroy(t.x);
        t.region.free();
        self.tp = null;
    }
}

pub fn info(self: *Model, peer: u32, out: *XchgInfo) !void {
    const t = &(self.tp orelse return error.BadWorld);
    if (h3x_info_for(t.x, @intCast(peer), out) != 0) return error.BadWorld;
}

pub fn connect(self: *Model, peer: u32, in: *const XchgInfo) !void {
    const t = if (self.tp) |*held| held else return error.BadWorld;
    if (h3x_connect(t.x, @intCast(peer), in) != 0) {
        std.debug.print("tensorfold h3 rank {d}: connecting rank {d}: {s}\n", .{ t.rank, peer, std.mem.span(h3x_error(t.x)) });
        return error.ExchangeFailed;
    }
    t.connected += 1;
}

/// `forward` for inputs at host addresses: copied to the device, the stream copied back; no kernel reads host memory.
pub fn forwardHost(self: *Model, in: *const Model.Inputs) !f64 {
    try self.ctx.makeCurrent();
    const rows: usize = in.rows;
    const count = if (in.blocks == 0) self.blocks.len else @min(in.blocks, self.blocks.len);
    const sizes = [7]usize{ rows * self.hidden * 2, count * in.lines * 6 * self.hidden * 4, rows * 4, rows * in.rot * 4, rows * in.rot * 4, rows * 4, @as(usize, in.tiles) * 4 };
    const from = [7]u64{ in.x, in.tables, in.line, in.cos, in.sin, in.slot, in.sizes };
    var copy = in.*;
    inline for (.{ "x", "tables", "line", "cos", "sin", "slot", "sizes" }, 0..) |f, i| {
        if (self.inputs[i] == null or self.inputs[i].?.len < sizes[i]) {
            if (self.inputs[i]) |*old| old.free();
            self.inputs[i] = try cuda.DeviceBuffer.alloc(&self.driver, sizes[i]);
        }
        try self.inputs[i].?.upload(0, @as([*]const u8, @ptrFromInt(from[i]))[0..sizes[i]]);
        @field(copy, f) = self.inputs[i].?.ptr;
    }
    const took = try self.forward(&copy);
    try self.inputs[0].?.download(0, @as([*]u8, @ptrFromInt(in.x))[0..sizes[0]]);
    return took;
}

/// Pinned host memory the GPU maps: returns its host address and stores its device address.
pub fn hostAlloc(self: *Model, bytes: usize, device: *u64) !*anyopaque {
    try self.ctx.makeCurrent();
    const held = try cuda.HostBuffer.allocMapped(&self.driver, bytes);
    device.* = try held.device();
    return held.bytes.ptr;
}

pub fn hostFree(self: *Model, ptr: *anyopaque) void {
    self.ctx.makeCurrent() catch return;
    _ = self.driver.api.cuMemFreeHost(ptr);
}

/// Frees what a world and forwardHost hold.
pub fn free(self: *Model) void {
    for (&self.inputs) |*held| if (held.*) |*b| b.free();
    if (self.tp) |*t| {
        h3x_destroy(t.x);
        t.region.free();
        self.tp = null;
    }
}
