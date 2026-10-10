//! What the kernel tests share: the device, a launcher with every choice on and one with each off, and helpers.
const std = @import("std");
const Runtime = @import("../runtime.zig").Runtime;
const Context = @import("../context.zig").Context;
const Stream = @import("../stream.zig").Stream;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Arena = @import("../arena.zig").Arena;
const Launcher = @import("../launches.zig").Launcher;
const Ops = @import("../ops/ops.zig").Ops;
const kernels = @import("../kernels.zig");
const caps = @import("../caps.zig");

pub const gpa = std.testing.allocator;

pub const Rng = struct {
    state: u64,

    pub fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    /// Uniform on [-1, 1) in 1/1024 steps.
    pub fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }

    pub fn uniform(r: *Rng) f64 {
        return (@as(f64, @floatFromInt(r.next() >> 11)) + 0.5) / 9007199254740992.0;
    }

    pub fn normal(r: *Rng) f64 {
        const a = r.uniform();
        const b = r.uniform();
        return @sqrt(-2 * @log(a)) * @cos(2 * std.math.pi * b);
    }
};

pub const Rig = struct {
    r: Runtime,
    ctx: Context,
    stream: Stream,
    /// Every kernel choice on, as a run takes them.
    on: Launcher,
    /// Merged decode launches and the chunked DeltaNet off: the kernels they replace.
    off: Launcher,
    arena: Arena,
    /// The activation type: bf16 where the GPU has a bf16 dot2, fp16 on RDNA2.
    bf16: bool,

    pub fn open(arena_bytes: usize) !*Rig {
        const t = try gpa.create(Rig);
        errdefer gpa.destroy(t);
        t.r = try Runtime.open();
        errdefer t.r.close();
        t.ctx = try Context.init(&t.r, 0);
        errdefer t.ctx.deinit();
        t.stream = try Stream.init(&t.r);
        errdefer t.stream.deinit();
        t.on = try Launcher.load(&t.r, .{}, kernels.images);
        errdefer t.on.unload();
        t.off = try Launcher.load(&t.r, .{ .fuse = false, .chunked = false }, kernels.images);
        errdefer t.off.unload();
        t.arena = try Arena.init(&t.r, arena_bytes);
        t.bf16 = caps.Caps.of(kernels.arch).?.act == .bf16;
        return t;
    }

    pub fn close(t: *Rig) void {
        t.arena.deinit();
        t.off.unload();
        t.on.unload();
        t.stream.deinit();
        t.ctx.deinit();
        t.r.close();
        gpa.destroy(t);
    }

    pub fn ops(t: *Rig, l: *const Launcher) Ops {
        return .{ .l = l, .bf16 = t.bf16, .stream = t.stream.handle, .arena = &t.arena };
    }

    pub fn upload(t: *Rig, host: anytype) !DeviceBuffer {
        return DeviceBuffer.fromHost(&t.r, std.mem.sliceAsBytes(host));
    }

    pub fn alloc(t: *Rig, bytes: usize) !DeviceBuffer {
        return DeviceBuffer.alloc(&t.r, bytes);
    }

    /// The activation's 16-bit pattern of `v` and its value.
    pub fn bits(t: *const Rig, v: f32) u16 {
        return if (t.bf16) bf16Bits(v) else f16Bits(v);
    }

    pub fn value(t: *const Rig, b: u16) f64 {
        return if (t.bf16) bf16Value(b) else f16Value(b);
    }

    /// The activation kind as the kernels number it: 1 fp16, 2 bf16.
    pub fn kind(t: *const Rig) c_int {
        return if (t.bf16) 2 else 1;
    }
};

pub fn at(b: DeviceBuffer) u64 {
    return @intFromPtr(b.ptr);
}

pub fn download(comptime T: type, b: DeviceBuffer) ![]T {
    const out = try gpa.alloc(T, b.len / @sizeOf(T));
    errdefer gpa.free(out);
    try b.download(0, std.mem.sliceAsBytes(out));
    return out;
}

pub fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

pub fn f16Value(b: u16) f64 {
    const h: f16 = @bitCast(b);
    return h;
}

/// bf16 by truncation: test inputs are chosen exact in bf16.
pub fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

pub fn bf16Value(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// A device buffer with its address, as the product tests hand it to launches.
pub const Buffer = struct {
    buf: DeviceBuffer,
    ptr: u64,

    pub fn alloc(t: *Rig, bytes: usize) !Buffer {
        const b = try DeviceBuffer.alloc(&t.r, bytes);
        return .{ .buf = b, .ptr = at(b) };
    }

    pub fn fromHost(t: *Rig, host: anytype) !Buffer {
        const b = try t.upload(host);
        return .{ .buf = b, .ptr = at(b) };
    }

    pub fn free(b: *Buffer) void {
        b.buf.free();
    }

    pub fn fill8(b: Buffer, value: u8, _: @TypeOf(null)) !void {
        try b.buf.fill8(value);
    }

    pub fn upload(b: Buffer, offset: usize, bytes: []const u8) !void {
        try b.buf.upload(offset, bytes);
    }

    pub fn download(b: Buffer, offset: usize, out: []u8) !void {
        try b.buf.download(offset, out);
    }
};

/// A failed check with its message.
pub fn expect(ok: bool, comptime fmt: []const u8, args: anytype) !void {
    if (ok) return;
    std.debug.print(fmt ++ "\n", args);
    return error.TestUnexpectedResult;
}
