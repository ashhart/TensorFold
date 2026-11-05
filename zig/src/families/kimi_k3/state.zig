//! Per-stream caches: KDA state with its last drafted window logged (kept rows replay next round), MLA latents.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const kernels = @import("kernels.zig");

pub const Limits = struct { slots: u32, log_rows: u32 = 16, max_ctx: u32 = 4096 };

/// A stream's place: tokens committed, and rows of its last window to replay before its next rows.
pub const Stream = struct { pos: u32 = 0, kept: u32 = 0 };

const KdaArena = struct { S: mtl.Buffer, conv: mtl.Buffer, log_a: mtl.Buffer, log_k: mtl.Buffer, log_u: mtl.Buffer, log_x: mtl.Buffer };

pub const State = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    limits: Limits,
    heads: u32,
    kda: []?KdaArena,
    mla: []?mtl.Buffer,
    index_keys: []?mtl.Buffer,
    streams: []Stream,

    /// Caches for `layers` (a subset for tests) and `heads` local KDA heads; every slot starts empty.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, cfg: Config, limits: Limits, layers: []const u32, heads: u32) !State {
        try cfg.checkEnd(0, limits.max_ctx);
        var s = meta: {
            const kda = try gpa.alloc(?KdaArena, cfg.layers);
            errdefer gpa.free(kda);
            const mla = try gpa.alloc(?mtl.Buffer, cfg.layers);
            errdefer gpa.free(mla);
            const index_keys = try gpa.alloc(?mtl.Buffer, cfg.layers);
            errdefer gpa.free(index_keys);
            const streams = try gpa.alloc(Stream, limits.slots);
            break :meta State{ .gpa = gpa, .cfg = cfg, .limits = limits, .heads = heads, .kda = kda, .mla = mla, .index_keys = index_keys, .streams = streams };
        };
        @memset(s.kda, null);
        @memset(s.mla, null);
        @memset(s.index_keys, null);
        @memset(s.streams, .{});
        errdefer s.deinit();
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const n: usize = limits.slots;
        const sz: usize = @as(usize, heads) * cfg.kda_dim;
        const W3: usize = 3 * sz;
        for (layers) |i| switch (cfg.kind(i)) {
            .kda => s.kda[i] = try arena(device, .{ n * sz * cfg.kda_dim * 4, n * (cfg.conv - 1) * W3 * 2, n * limits.log_rows * sz * 4, n * limits.log_rows * sz * 4, n * limits.log_rows * sz * 4, n * limits.log_rows * W3 * 2 }, opts),
            .mla => s.mla[i] = try device.buffer(n * limits.max_ctx * (cfg.kv_lora + cfg.rope) * 2, opts),
        };
        if (cfg.dsa != null) for (layers) |i| {
            if (cfg.kind(i) == .mla) s.index_keys[i] = try device.buffer(n * limits.max_ctx * 128 * 2, opts);
        };
        for (0..limits.slots) |slot| s.reset(@intCast(slot));
        return s;
    }

    fn arena(device: mtl.Device, sizes: [6]usize, opts: usize) !KdaArena {
        var result: KdaArena = undefined;
        var made: usize = 0;
        const fields = @typeInfo(KdaArena).@"struct".field_names;
        errdefer inline for (fields, 0..) |name, n| {
            if (n < made) @field(result, name).deinit();
        };
        inline for (fields, 0..) |name, n| {
            @field(result, name) = try device.buffer(sizes[n], opts);
            made += 1;
        }
        return result;
    }

    /// A slot's caches emptied for a new stream (zero state, zero conv window).
    pub fn reset(s: *State, slot: u32) void {
        s.streams[slot] = .{};
        const sz: usize = @as(usize, s.heads) * s.cfg.kda_dim;
        for (s.kda) |maybe| if (maybe) |k| {
            @memset(k.S.slice(f32, (slot + 1) * sz * s.cfg.kda_dim)[slot * sz * s.cfg.kda_dim ..], 0);
            const conv = (s.cfg.conv - 1) * 3 * sz;
            @memset(k.conv.slice(u16, (slot + 1) * conv)[slot * conv ..], 0);
        };
    }

    pub fn kdaState(s: *const State, layer: u32) kernels.KdaState {
        const k = s.kda[layer].?;
        return .{ .S = k.S.gpuAddress(), .conv = k.conv.gpuAddress(), .log_a = k.log_a.gpuAddress(), .log_k = k.log_k.gpuAddress(), .log_u = k.log_u.gpuAddress(), .log_x = k.log_x.gpuAddress() };
    }

    pub fn mlaCache(s: *const State, layer: u32) u64 {
        return s.mla[layer].?.gpuAddress();
    }

    pub fn indexCache(s: *const State, layer: u32) kernels.Ref {
        return .{ .buf = s.index_keys[layer].? };
    }

    /// After a round: a prompt chunk commits all its rows; a drafted window keeps its first `kept` rows.
    pub fn advance(s: *State, slot: u32, rows: u32, commit: bool, kept: u32) void {
        const st = &s.streams[slot];
        if (commit) {
            st.pos += rows;
            st.kept = 0;
        } else {
            std.debug.assert(kept <= rows);
            st.pos += kept;
            st.kept = kept;
        }
    }

    /// Every cache buffer, for a residency set.
    pub fn buffers(s: *const State, out: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator) !void {
        for (s.kda) |maybe| if (maybe) |k| inline for (@typeInfo(KdaArena).@"struct".field_names) |f| try out.append(gpa, @field(k, f));
        for (s.mla) |maybe| if (maybe) |b| try out.append(gpa, b);
        for (s.index_keys) |maybe| if (maybe) |b| try out.append(gpa, b);
    }

    pub fn deinit(s: *State) void {
        for (s.kda) |maybe| if (maybe) |k| inline for (@typeInfo(KdaArena).@"struct".field_names) |f| @field(k, f).deinit();
        for (s.mla) |maybe| if (maybe) |b| b.deinit();
        for (s.index_keys) |maybe| if (maybe) |b| b.deinit();
        s.gpa.free(s.kda);
        s.gpa.free(s.mla);
        s.gpa.free(s.index_keys);
        s.gpa.free(s.streams);
    }
};

test "Kimi state metadata releases each partially allocated array without opening a GPU" {
    const Probe = struct {
        fn run(a: std.mem.Allocator) !void {
            const c: Config = .{};
            var s = try State.init(a, undefined, c, .{ .slots = 1, .max_ctx = 32 }, &.{}, c.kda_heads);
            defer s.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Probe.run, .{});
}
