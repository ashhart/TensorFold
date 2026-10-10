//! Native storage flags and contiguous slot layouts refuse unsupported GDN state before a GPU dispatch.
const std = @import("std");
const Config = @import("config.zig").Config;
const DType = @import("checkpoint.zig").DType;

pub const Storage = enum(u32) { bf16 = 0, f32 = 1 };

pub fn storage(dtype: DType) !Storage {
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        else => error.UnsupportedGdnStorage,
    };
}

pub const WeightKinds = struct {
    conv: Storage,
    a_log: Storage,
    dt: Storage,
    norm: Storage,

    pub fn flags(k: WeightKinds) u32 {
        return @backingInt(k.conv) | (@backingInt(k.a_log) << 1) | (@backingInt(k.dt) << 2) | (@backingInt(k.norm) << 3);
    }
};

pub const Params = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    taps: u32,
    slots: u32,
    weight_flags: u32,
    eps: f32,
    /// 0: z, a and b in their own rows; else one fused zba row of this many values (z, then b, then a).
    zba_stride: u32 = 0,
};

pub const Segment = extern struct { first: u32, rows: u32, state_slot: u32, next_slot: u32 };
pub const Keep = extern struct { first: u32, rows: u32, state_slot: u32, next_slot: u32 };

pub const Shape = struct {
    c: Config,
    qkv: usize,
    value: usize,
    state: usize,
    conv: usize,

    pub fn init(c: Config) !Shape {
        if (c.k_heads == 0 or c.v_heads == 0 or c.k_heads > 128 or c.v_heads > 128 or c.v_heads % c.k_heads != 0 or c.dk != c.dv or c.dk < 32 or c.dk > 256 or c.dk % 32 != 0 or c.conv_kernel < 2 or c.conv_kernel > 8) return error.UnsupportedGdnShape;
        return .{ .c = c, .qkv = c.gdnQkvDim(), .value = c.gdnValueDim(), .state = try std.math.mul(usize, c.gdnValueDim(), c.dk), .conv = try std.math.mul(usize, c.gdnQkvDim(), c.conv_kernel - 1) };
    }

    pub fn params(s: Shape, rows: usize, slots: usize, k: WeightKinds) !Params {
        if (rows == 0 or rows > 128 or slots == 0 or slots > 64 or !std.math.isFinite(s.c.eps) or s.c.eps <= 0) return error.BadGdnDispatch;
        return .{ .rows = @intCast(rows), .nk = @intCast(s.c.k_heads), .nv = @intCast(s.c.v_heads), .dk = @intCast(s.c.dk), .dv = @intCast(s.c.dv), .taps = @intCast(s.c.conv_kernel), .slots = @intCast(slots), .weight_flags = k.flags(), .eps = s.c.eps };
    }
};

pub fn countBytes(comptime T: type, elements: usize, copies: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, elements, copies), @sizeOf(T));
}

pub const Bytes = struct {
    qkv: usize,
    query: usize,
    value: usize,
    heads: usize,
    decay: usize,
    state: usize,
    snapshots: usize,
    history: usize,
    tails: usize,
    conv_weight: usize,
    a_log: usize,
    dt: usize,
    norm: usize,

    pub fn init(s: Shape, p: Params, k: WeightKinds) !Bytes {
        var expected = try s.params(p.rows, p.slots, k);
        if (p.zba_stride != 0 and p.zba_stride < s.value + 2 * s.c.v_heads) return error.BadGdnDispatch;
        expected.zba_stride = p.zba_stride;
        if (!std.mem.eql(u8, std.mem.asBytes(&p), std.mem.asBytes(&expected))) return error.BadGdnDispatch;
        return .{
            .qkv = try countBytes(u16, s.qkv, p.rows),
            .query = try countBytes(u16, s.c.k_heads * s.c.dk, p.rows),
            .value = try countBytes(u16, s.value, p.rows),
            .heads = try countBytes(u16, s.c.v_heads, p.rows),
            .decay = try countBytes(f32, s.c.v_heads, p.rows),
            .state = try countBytes(f32, s.state, p.slots),
            .snapshots = try countBytes(f32, s.state, p.rows),
            .history = try countBytes(u16, s.conv, p.slots),
            .tails = try countBytes(u16, s.conv, p.rows),
            .conv_weight = try countBytes(u8, s.qkv * s.c.conv_kernel, byteWidth(k.conv)),
            .a_log = try countBytes(u8, s.c.v_heads, byteWidth(k.a_log)),
            .dt = try countBytes(u8, s.c.v_heads, byteWidth(k.dt)),
            .norm = try countBytes(u8, s.c.dv, byteWidth(k.norm)),
        };
    }
};

fn byteWidth(s: Storage) usize {
    return switch (s) {
        .bf16 => 2,
        .f32 => 4,
    };
}

pub fn checkSpan(total: usize, offset: usize, bytes: usize, required: usize, alignment: usize) !void {
    if ((alignment != 2 and alignment != 4) or offset > total or bytes > total - offset or bytes < required or offset % alignment != 0) return error.BadGdnBuffer;
}
