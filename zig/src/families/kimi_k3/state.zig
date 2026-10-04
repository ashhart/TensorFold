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
    streams: []Stream,

    /// Caches for `layers` (a subset for tests) and `heads` local KDA heads; every slot starts empty.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, cfg: Config, limits: Limits, layers: []const u32, heads: u32) !State {
        var s = State{ .gpa = gpa, .cfg = cfg, .limits = limits, .heads = heads, .kda = try gpa.alloc(?KdaArena, cfg.layers), .mla = try gpa.alloc(?mtl.Buffer, cfg.layers), .streams = try gpa.alloc(Stream, limits.slots) };
        @memset(s.kda, null);
        @memset(s.mla, null);
        @memset(s.streams, .{});
        errdefer s.deinit();
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const n: usize = limits.slots;
        const sz: usize = @as(usize, heads) * cfg.kda_dim;
        const W3: usize = 3 * sz;
        for (layers) |i| switch (cfg.kind(i)) {
            .kda => s.kda[i] = .{
                .S = try device.buffer(n * sz * cfg.kda_dim * 4, opts),
                .conv = try device.buffer(n * (cfg.conv - 1) * W3 * 2, opts),
                .log_a = try device.buffer(n * limits.log_rows * sz * 4, opts),
                .log_k = try device.buffer(n * limits.log_rows * sz * 4, opts),
                .log_u = try device.buffer(n * limits.log_rows * sz * 4, opts),
                .log_x = try device.buffer(n * limits.log_rows * W3 * 2, opts),
            },
            .mla => s.mla[i] = try device.buffer(n * limits.max_ctx * (cfg.kv_lora + cfg.rope) * 2, opts),
        };
        for (0..limits.slots) |slot| s.reset(@intCast(slot));
        return s;
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
    }

    pub fn deinit(s: *State) void {
        for (s.kda) |maybe| if (maybe) |k| inline for (@typeInfo(KdaArena).@"struct".field_names) |f| @field(k, f).deinit();
        for (s.mla) |maybe| if (maybe) |b| b.deinit();
        s.gpa.free(s.kda);
        s.gpa.free(s.mla);
        s.gpa.free(s.streams);
    }
};
