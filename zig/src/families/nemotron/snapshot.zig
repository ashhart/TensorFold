//! Nemotron's prompt states for the prompt cache: Mamba conv and SSM states, the KV and head KV prefixes.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const Metal = @import("backend.zig").Metal;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// One kept state: `at` prompt tokens of one stream, in one buffer the GPU copies on the engine's queue.
pub const Snap = struct { at: u32, head: bool, buf: mtl.Buffer, bytes: usize };

fn convBytes(c: cfg.Config) usize {
    return (c.conv_kernel - 1) * c.convDim() * 2;
}

fn ssmBytes(c: cfg.Config) usize {
    return c.mamba_heads * c.mamba_head_dim * c.state * 4;
}

/// One head's key (or value) rows for `at` tokens.
fn rowBytes(c: cfg.Config, at: usize) usize {
    return at * c.head_dim * 2;
}

/// The bytes a state at `at` tokens takes (with the head's cache when `head`).
pub fn bytes(c: cfg.Config, at: usize, head: bool) usize {
    const kvs = c.count(.attention) + @intFromBool(head);
    return c.count(.mamba) * (convBytes(c) + ssmBytes(c)) + kvs * 2 * c.kv_heads * rowBytes(c, at);
}

/// Copies between a stream's live rows (Mamba states in `slot`) and `snap`, a dispatch each (`into`: live to snap).
const Copies = struct {
    c: *st.Cache,
    slot: u32,
    snap: *const Snap,
    into: bool,

    fn part(self: @This(), b: *Metal, e: *fwd.Enc, off: *usize, live: mtl.Buffer, live_off: usize, n: usize) void {
        e.alongside(); // every part reads and writes its own rows
        e.pipe(b.m.kernels.get("tf_copy_u32"));
        if (self.into) {
            e.buf(live, live_off, 0);
            e.buf(self.snap.buf, off.*, 1);
        } else {
            e.buf(self.snap.buf, off.*, 0);
            e.buf(live, live_off, 1);
        }
        e.run(.{ n / 4, 1, 1 }, .{ 256, 1, 1 });
        off.* += n;
    }

    pub fn encode(self: @This(), b: *Metal, e: *fwd.Enc) !void {
        const c = b.m.config;
        var off: usize = 0;
        for (0..b.pool.layers) |m| {
            self.part(b, e, &off, b.pool.conv[m], self.slot * convBytes(c), convBytes(c));
            self.part(b, e, &off, b.pool.ssm[m], self.slot * ssmBytes(c), ssmBytes(c));
        }
        const n = rowBytes(c, self.snap.at);
        for (0..self.c.attentions + @intFromBool(self.c.mtp != null)) |i| {
            const kv = if (i < self.c.attentions) self.c.kv[i] else self.c.mtp.?;
            for ([2]mtl.Buffer{ kv.k, kv.v }) |live| for (0..c.kv_heads) |h| self.part(b, e, &off, live, rowBytes(c, h * kv.capacity), n);
        }
        if (off != self.snap.bytes) return error.SnapshotSize;
    }
};

/// The stream's state after `at` prompt tokens (its pass stands there) into a new buffer, copied on the queue.
pub fn save(b: *Metal, cache: *st.Cache, at: u32) !*Snap {
    const head = cache.mtp != null;
    if (at == 0 or cache.len != at or cache.replay != 0 or cache.pending != null or (head and cache.mtp_len != at)) return error.SnapshotOutOfStep;
    const n = bytes(b.m.config, at, head);
    const snap = try b.gpa.create(Snap);
    errdefer b.gpa.destroy(snap);
    snap.* = .{ .at = at, .head = head, .buf = try b.m.device.buffer(n, opts), .bytes = n };
    errdefer snap.buf.deinit();
    try b.submit(.prefill, b.next, Copies{ .c = cache, .slot = cache.slot, .snap = snap, .into = true });
    return snap;
}

/// A fresh stream's cache becomes `snap`'s, in a new state slot, copied on the queue before its first prompt chunk.
pub fn restore(b: *Metal, cache: *st.Cache, snap: *const Snap) !void {
    if (cache.len != 0 or snap.head != (cache.mtp != null)) return error.SnapshotLayout;
    if (snap.at + 1 > cache.kv[0].capacity) return error.ContextFull;
    const slot = try b.pool.take();
    errdefer b.pool.give(slot);
    try b.submit(.prefill, b.next, Copies{ .c = cache, .slot = slot, .snap = snap, .into = false });
    cache.advance(&b.pool, snap.at, slot);
    if (cache.mtp != null) cache.mtp_len = snap.at;
}

/// A state's buffer goes once no command buffer in flight reads it (the queue holds unretained references).
pub fn drop(b: *Metal, snap: *Snap) void {
    b.drain() catch {};
    snap.buf.deinit();
    b.gpa.destroy(snap);
}

/// The prompt cache's Snapshots functions over the Metal backend (`ptr`: the *Metal, `owner`: the stream).
pub const Cached = struct {
    fn metal(ptr: *anyopaque) *Metal {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        const b = metal(ptr);
        return bytes(b.m.config, at, b.head != null);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const b = metal(ptr);
        const c = try b.cacheOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        return @ptrCast(try save(b, c, at));
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        drop(metal(ptr), @ptrCast(@alignCast(saved)));
    }
};

test "a state's bytes: fixed Mamba states, then a key and value row per KV head and attention layer a token" {
    var c: cfg.Config = undefined;
    c.conv_kernel = 4;
    c.mamba_heads = 64;
    c.mamba_head_dim = 64;
    c.groups = 8;
    c.state = 128;
    c.head_dim = 128;
    c.kv_heads = 2;
    c.layers = 3;
    c.kinds[0] = .mamba;
    c.kinds[1] = .moe;
    c.kinds[2] = .attention;
    const fixed = 3 * 6144 * 2 + 64 * 64 * 128 * 4;
    try std.testing.expectEqual(@as(usize, fixed), bytes(c, 0, true));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2 * 128 * 2), bytes(c, 1, true) - bytes(c, 0, true));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 128 * 2), bytes(c, 1, false) - bytes(c, 0, false));
}
