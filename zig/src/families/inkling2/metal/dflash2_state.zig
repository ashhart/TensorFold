//! DFlash 2 Metal scratch arenas and per-stream private K/V ring buffers.
const std = @import("std");
const mtl = @import("metal");
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");

pub const absorb_rows: usize = 256;
pub const max_chunks: usize = (w.window_size + w.block_size + 127) / 128;
pub const max_batch_streams: usize = k.max_batch;
pub const min_buffer_bytes: usize = 16;

pub const scratch_fields = [_][]const u8{
    "ids",  "resid", "x",  "xc",   "dyn", "q_raw", "k_raw",  "v_raw", "q",    "kb",  "vb",  "part",
    "attn", "y",     "yc", "gate", "up",  "act",   "logits", "hid",   "tval", "tid", "seq", "taps",
    "fcx",  "hctx",  "ck", "cv",
};

pub fn scratchSizes(max_streams: usize) [scratch_fields.len]usize {
    const streams = @max(@as(usize, 1), @min(max_streams, max_batch_streams));
    const n = streams * w.block_size;
    const a = absorb_rows;
    const bf = 2;
    return .{
        n * 4,
        n * w.hidden * 4,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.dyn_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.query_width * bf,
        n * w.kv_width * bf,
        n * w.kv_width * bf,
        n * w.q_heads * max_chunks * 132 * 4,
        n * w.query_width * bf,
        n * w.hidden * bf,
        n * w.hidden * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.intermediate * bf,
        n * w.padded_vocab * bf,
        n * w.rank * bf,
        n * w.top_k * 4,
        n * w.top_k * 4,
        n * 4,
        if (streams > 1) a * w.taps_n * w.hidden * bf else 0,
        a * w.hidden * bf,
        a * w.hidden * bf,
        a * w.kv_width * bf,
        a * w.kv_width * bf,
    };
}

pub fn scratchBytes(max_streams: usize) u64 {
    var total: u64 = 0;
    for (scratchSizes(max_streams)) |s| total += @max(s, min_buffer_bytes);
    return total;
}

pub fn streamBytes() u64 {
    return @as(u64, w.layers_n) * 2 * w.ring_size * w.kv_width * 2;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer,
    x: mtl.Buffer,
    xc: mtl.Buffer,
    dyn: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    q: mtl.Buffer,
    kb: mtl.Buffer,
    vb: mtl.Buffer,
    part: mtl.Buffer,
    attn: mtl.Buffer,
    y: mtl.Buffer,
    yc: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    logits: mtl.Buffer,
    hid: mtl.Buffer,
    tval: mtl.Buffer,
    tid: mtl.Buffer,
    seq: mtl.Buffer,
    taps: mtl.Buffer,
    fcx: mtl.Buffer,
    hctx: mtl.Buffer,
    ck: mtl.Buffer,
    cv: mtl.Buffer,

    pub fn init(device: mtl.Device, max_streams: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        const ns = scratchSizes(max_streams);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (scratch_fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (scratch_fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], min_buffer_bytes), mtl.ResourceOptions.shared);
            made += 1;
        }
        return s;
    }

    pub fn bytes(s: *const Scratch) u64 {
        var total: u64 = 0;
        inline for (scratch_fields) |f| total += @field(s, f).length();
        return total;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (scratch_fields) |f| @field(s, f).deinit();
        s.* = undefined;
    }
};

pub const Lane = struct {
    k: [w.layers_n]mtl.Buffer,
    v: [w.layers_n]mtl.Buffer,
    from: u64 = 0,
    end: u64 = 0,
    held: [w.block_size - 1]u32 = @splat(0),
    held_len: u32 = 0,

    pub fn init(device: mtl.Device) !Lane {
        var l: Lane = .{ .k = undefined, .v = undefined };
        const bytes = w.ring_size * w.kv_width * 2;
        var made: usize = 0;
        errdefer for (0..made) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        };
        for (0..w.layers_n) |i| {
            l.k[i] = try device.buffer(bytes, mtl.ResourceOptions.private);
            l.v[i] = device.buffer(bytes, mtl.ResourceOptions.private) catch |err| {
                l.k[i].deinit();
                return err;
            };
            made += 1;
        }
        return l;
    }

    pub fn deinit(l: *Lane) void {
        for (0..w.layers_n) |i| {
            l.k[i].deinit();
            l.v[i].deinit();
        }
        l.* = undefined;
    }
};

pub const AbsorbSeg = struct { lane: *const Lane, src: mtl.Buffer, src_off: usize, pos0: u64, rows: usize };
pub const AbsorbWork = struct { lane: *Lane, prev_end: u64, start: u64, keep_from: u64, end: u64, src: mtl.Buffer, offset: usize };
pub const Block = struct { lane: *const Lane, pending: u32, anchor: u64 };

test "DFlash2 1-stream Scratch clamps zero-sized taps buffer to 16 bytes and matches scratchBytes exactly" {
    const raw1 = scratchSizes(1);
    var raw_sum: u64 = 0;
    var taps_idx: usize = 0;
    for (scratch_fields, 0..) |name, i| {
        if (std.mem.eql(u8, name, "taps")) taps_idx = i;
        raw_sum += raw1[i];
    }
    try std.testing.expectEqual(@as(usize, 0), raw1[taps_idx]);
    try std.testing.expectEqual(raw_sum + min_buffer_bytes, scratchBytes(1));
    if (comptime @import("builtin").os.tag != .macos) return;
    const device = mtl.Device.init() catch return;
    defer device.deinit();
    var s1 = try Scratch.init(device, 1);
    defer s1.deinit();
    try std.testing.expectEqual(@as(usize, min_buffer_bytes), s1.taps.length());
    try std.testing.expectEqual(scratchBytes(1), s1.bytes());
    var s2 = try Scratch.init(device, 2);
    defer s2.deinit();
    try std.testing.expectEqual(scratchSizes(2)[taps_idx], s2.taps.length());
    try std.testing.expectEqual(scratchBytes(2), s2.bytes());
}
