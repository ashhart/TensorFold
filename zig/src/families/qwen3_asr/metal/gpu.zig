//! Qwen3-ASR's Metal device: one queue, the qwen3_asr metallib, buffers addressed as u64 (slot << 40 | byte offset).
const std = @import("std");
const mtl = @import("metal");

/// Every pipeline of qwen3_asr.metallib, by its kernel name.
pub const P = enum { q3a_gemm, q3a_gemv, q3a_conv_cols, q3a_flatten, q3a_layernorm, q3a_add_rmsnorm, q3a_gelu, q3a_swiglu, q3a_embed, q3a_copy, q3a_qk_rope, q3a_attn };

const n_pipes = @typeInfo(P).@"enum".field_names.len;
const slot_shift = 40;

pub const Ref = struct { buf: mtl.Buffer, off: usize };

pub const Gpu = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipes: [n_pipes]mtl.Pipeline,
    bufs: std.ArrayList(mtl.Buffer) = .empty, // by slot; slot 0 is never used, so address 0 is null
    bytes: u64 = 0, // every buffer's length
    pool: ?mtl.objc.Pool = null,
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,

    pub fn open(gpa: std.mem.Allocator, metallib: []const u8) !*Gpu {
        const g = try gpa.create(Gpu);
        errdefer gpa.destroy(g);
        g.* = .{ .gpa = gpa, .device = try mtl.Device.init(), .queue = undefined, .lib = undefined, .pipes = undefined };
        errdefer g.device.deinit();
        g.queue = try g.device.queue();
        errdefer g.queue.deinit();
        g.lib = try mtl.Library.fromBytes(g.device, metallib);
        errdefer g.lib.deinit();
        var made: usize = 0;
        errdefer for (g.pipes[0..made]) |pipe| pipe.deinit();
        inline for (@typeInfo(P).@"enum".field_names, 0..) |name, i| {
            g.pipes[i] = try mtl.Pipeline.init(g.device, g.lib, name, false);
            made += 1;
        }
        try g.bufs.append(gpa, undefined);
        return g;
    }

    pub fn close(g: *Gpu) void {
        g.flush() catch {};
        for (g.bufs.items[1..]) |b| b.deinit();
        g.bufs.deinit(g.gpa);
        for (&g.pipes) |*p| p.deinit();
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
        g.gpa.destroy(g);
    }

    /// A new zeroed shared buffer of `bytes`, by address.
    pub fn alloc(g: *Gpu, bytes: usize) !u64 {
        const len = std.mem.alignForward(usize, @max(bytes, 256), 256);
        if (len > g.device.maxWorkingSet() -| g.device.allocated()) return error.OutOfMemory;
        const b = try g.device.buffer(len, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        @memset(b.contents()[0..len], 0);
        try g.bufs.append(g.gpa, b);
        g.bytes += len;
        return @as(u64, g.bufs.items.len - 1) << slot_shift;
    }

    pub fn ref(g: *const Gpu, addr: u64) Ref {
        return .{ .buf = g.bufs.items[@intCast(addr >> slot_shift)], .off = @intCast(addr & ((@as(u64, 1) << slot_shift) - 1)) };
    }

    /// The host view of `len` bytes at `addr` (unified memory; flush first when the GPU writes them).
    pub fn host(g: *const Gpu, addr: u64, len: usize) []u8 {
        const r = g.ref(addr);
        std.debug.assert(r.off + len <= r.buf.length());
        return r.buf.contents()[r.off..][0..len];
    }

    fn encoder(g: *Gpu) mtl.ComputeEncoder {
        if (g.enc) |e| return e;
        if (g.pool == null) g.pool = mtl.objc.Pool.push();
        const cb = g.queue.commandBuffer();
        g.cb = cb;
        const e = cb.compute(.serial);
        g.enc = e;
        return e;
    }

    /// One dispatch: buffers bound at 0.., then `args` (an extern struct) at the next index.
    pub fn go(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype) void {
        const e = g.encoder();
        e.setPipeline(g.pipes[@backingInt(p)]);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
    }

    /// Commit what is encoded and wait until the GPU is done.
    pub fn flush(g: *Gpu) !void {
        const e = g.enc orelse return;
        e.end();
        const cb = g.cb.?;
        g.enc = null;
        g.cb = null;
        defer {
            if (g.pool) |p| p.pop();
            g.pool = null;
        }
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("qwen3-asr metal: command buffer failed: {s}", .{msg});
            return error.CommandBufferFailed;
        }
    }

    pub fn put(g: *Gpu, dst: u64, bytes: []const u8) !void {
        try g.flush();
        @memcpy(g.host(dst, bytes.len), bytes);
    }

    pub fn get(g: *Gpu, src: u64, out: []u8) !void {
        try g.flush();
        @memcpy(out, g.host(src, out.len));
    }
};
