//! ZONOS3 Mini's Metal device: one queue, the zonos3 metallib, buffers addressed as u64 (slot << 40 | byte offset).
const std = @import("std");
const mtl = @import("metal");

/// Every pipeline of zonos3.metallib, by its kernel name: the backbone's, the codec's, the encoder's, the teacher's.
pub const P = enum {
    z3_embed,
    z3_rmsnorm,
    z3_attnres,
    z3_linear,
    z3_conv_silu,
    z3_gdn_prep,
    z3_gdn,
    z3_gated_norm,
    z3_attn_prep,
    z3_attention,
    z3_mul_silu,
    z3_route,
    z3_combine,
    z3_softcap,
    zc_linear,
    zc_layernorm,
    zc_attn_prep,
    zc_attention,
    zc_add_scaled,
    zc_gelu,
    zc_patch_up,
    zc_dwconv,
    zc_gather,
    zc_latent,
    zc_spec,
    zc_ola,
    zc_mix,
    zc_resample,
    ze_resample_in,
    ze_conv,
    ze_snake,
    ze_add,
    ze_sub,
    ze_patch_down,
    ze_nearest,
    ze_sub_code,
    zt_swish,
    zt_glu,
    zt_dwconv,
    zt_attention,
};

const n_pipes = @typeInfo(P).@"enum".field_names.len;
const slot_shift = 40;

pub const Ref = struct { buf: mtl.Buffer, off: usize };

pub const Gpu = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipes: [n_pipes]mtl.Pipeline,
    bufs: std.ArrayList(mtl.Buffer), // by slot; slot 0 is never used, so address 0 is null
    counted: ?usize = null, // set: allocations are counted, not made
    held: usize = 0, // bytes of the buffers made
    peak: usize = 0, // the most `held` has been
    pool: ?mtl.objc.Pool = null,
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,

    pub fn open(gpa: std.mem.Allocator, metallib: []const u8) !*Gpu {
        const g = try gpa.create(Gpu);
        errdefer gpa.destroy(g);
        g.* = .{ .gpa = gpa, .device = try mtl.Device.init(), .queue = undefined, .lib = undefined, .pipes = undefined, .bufs = .empty };
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
        errdefer g.bufs.deinit(gpa);
        try g.bufs.append(gpa, undefined);
        return g;
    }

    /// The device with this family's metallib.
    pub fn openFamily(gpa: std.mem.Allocator) !*Gpu {
        return open(gpa, @import("zonos3_metallib").bytes);
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

    /// The buffers from slot `first` on, released (a part's state and workspaces, made after the weights).
    pub fn releaseFrom(g: *Gpu, first: usize) void {
        g.flush() catch {};
        std.debug.assert(first >= 1 and first <= g.bufs.items.len);
        for (g.bufs.items[first..]) |b| {
            g.held -= b.length();
            b.deinit();
        }
        g.bufs.shrinkRetainingCapacity(first);
    }

    pub fn slots(g: *const Gpu) usize {
        return g.bufs.items.len;
    }

    /// A new zeroed shared buffer of `bytes`, by address (counted only, while counting).
    pub fn alloc(g: *Gpu, bytes: usize) !u64 {
        if (bytes > std.math.maxInt(usize) - 255) return error.OutOfMemory;
        const len = std.mem.alignForward(usize, @max(bytes, 256), 256);
        if (g.counted) |*n| {
            n.* += len;
            return 0;
        }
        if (len > g.device.maxWorkingSet() -| g.device.allocated()) return error.OutOfMemory;
        const b = try g.device.buffer(len, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        @memset(b.contents()[0..len], 0);
        try g.bufs.append(g.gpa, b);
        g.held += len;
        g.peak = @max(g.peak, g.held);
        return @as(u64, g.bufs.items.len - 1) << slot_shift;
    }

    /// A new buffer holding `bytes`.
    pub fn upload(g: *Gpu, bytes: []const u8) !u64 {
        const at = try g.alloc(bytes.len);
        @memcpy(g.host(at, bytes.len), bytes);
        return at;
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
        g.goShared(p, groups, threads, bufs, args, 0);
    }

    /// One dispatch with `shared` bytes of threadgroup memory at threadgroup index 0.
    pub fn goShared(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype, shared: usize) void {
        const e = g.encoder();
        if (shared > 0) e.setThreadgroupMemory(std.mem.alignForward(usize, shared, 16), 0);
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
            std.log.err("zonos3 metal: command buffer failed: {s}", .{msg});
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
