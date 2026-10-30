//! LFM3-Omni's Metal device: one queue, buffers as u64 (slot << 40 | offset), one serial encoder until a flush.
const std = @import("std");
const mtl = @import("metal");

/// Every pipeline of lfm3_omni.metallib, by its kernel name.
pub const P = enum {
    om_rmsnorm,
    om_rmsnorm_1k,
    om_head_norm,
    om_layernorm,
    om_rope,
    om_copy_rows,
    om_set_words,
    om_zero_words,
    om_copy_words,
    om_short_conv,
    om_short_conv_rows,
    om_conv_state,
    om_attention_tiles,
    om_attend,
    om_qk_prep,
    om_swiglu,
    om_bias_act,
    om_embed,
    om_embed_codes,
    om_depth_in,
    om_conv2d_s2,
    om_f32_bf16,
    om_bf16_f32,
    om_fsq_decode,
    om_repeat_rows,
    om_polar,
    om_argmax_part,
    om_argmax_final,
    om_rows_1,
    om_rows_4,
    om_rows_8,
    om_rows_12,
    om_rows_16,
    om_gemm_bb,
    om_gemm_bf,
    om_gemm_ff,
    om_depth_pick,
    om_frame_end,
    om_mel_frames,
    om_power,
    om_logmel,
};

const n_pipes = @typeInfo(P).@"enum".field_names.len;

/// The tensor-unit products (omni_nax.metal), built only on a GPU that has the units (M5 and later).
pub const Nax = enum { om_nax_bb_nt, om_nax_bf_nt, om_nax_bf_nn, om_nax_bf_tt };
const n_nax = @typeInfo(Nax).@"enum".field_names.len;
const slot_shift = 40;

pub const Ref = struct { buf: mtl.Buffer, off: usize };

pub const Gpu = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipes: [n_pipes]mtl.Pipeline,
    /// Null without tensor units (the simdgroup-matrix product takes every product then).
    nax: [n_nax]?mtl.Pipeline = @splat(null),
    bufs: std.ArrayList(mtl.Buffer), // by slot; slot 0 is a small dummy, bound where a kernel's optional input is null
    bytes: usize = 0,
    pool: ?mtl.objc.Pool = null,
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,
    pending: ?mtl.CommandBuffer = null, // committed, not yet waited for
    /// Dispatches encoded since the last commit, and in all (for the profile).
    encoded: usize = 0,
    dispatches: usize = 0,
    /// GPU seconds of the command buffers waited for (`flush`), for the profile.
    gpu_s: f64 = 0,

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
        errdefer for (g.nax) |n| if (n) |pipe| pipe.deinit();
        // LFM3_OMNI_NAX=0 keeps every product on the simdgroup-matrix kernels (an A/B of the two)
        const off = if (std.c.getenv("LFM3_OMNI_NAX")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        if (g.device.tensorUnits() and !off) inline for (@typeInfo(Nax).@"enum".field_names, 0..) |name, i| {
            g.nax[i] = mtl.Pipeline.initOptional(g.device, g.lib, name, false) catch null;
        };
        errdefer g.bufs.deinit(gpa);
        const dummy = try g.device.buffer(256, mtl.ResourceOptions.shared);
        try g.bufs.append(gpa, dummy);
        return g;
    }

    pub fn close(g: *Gpu) void {
        g.flush() catch {};
        for (g.bufs.items) |b| b.deinit();
        g.bufs.deinit(g.gpa);
        for (&g.pipes) |*p| p.deinit();
        for (g.nax) |n| if (n) |pipe| pipe.deinit();
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

    /// The host view of `len` bytes at `addr` (unified memory; the GPU's writes are there after a flush).
    pub fn host(g: *const Gpu, addr: u64, len: usize) []u8 {
        const r = g.ref(addr);
        return r.buf.contents()[r.off..][0..len];
    }

    /// The encoder of the open command buffer (made on first use).
    pub fn encoder(g: *Gpu) mtl.ComputeEncoder {
        if (g.enc) |e| return e;
        if (g.pool == null) g.pool = mtl.objc.Pool.push();
        const cb = g.queue.commandBufferUnretained();
        g.cb = cb;
        const e = cb.compute(.serial);
        g.enc = e;
        return e;
    }

    /// One dispatch: buffers bound at 0.. (address 0 the dummy), then `args` (extern struct, scalar or void) after.
    pub fn go(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype) void {
        const e = g.encoder();
        e.setPipeline(g.pipes[@backingInt(p)]);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        if (@TypeOf(args) != void) e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
        g.encoded += 1;
        g.dispatches += 1;
    }

    /// `go` with `tg_bytes` of threadgroup memory at threadgroup index 0.
    pub fn goTg(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype, tg_bytes: usize) void {
        const e = g.encoder();
        e.setPipeline(g.pipes[@backingInt(p)]);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.setThreadgroupMemory(std.mem.alignForward(usize, @max(tg_bytes, 16), 16), 0);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
        g.encoded += 1;
        g.dispatches += 1;
    }

    /// Whether every tensor-unit product built.
    pub fn hasNax(g: *const Gpu) bool {
        for (g.nax) |n| if (n == null) return false;
        return true;
    }

    /// A tensor-unit product's dispatch (as `go`).
    pub fn goNax(g: *Gpu, p: Nax, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype) void {
        const e = g.encoder();
        e.setPipeline(g.nax[@backingInt(p)].?);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
        g.encoded += 1;
        g.dispatches += 1;
    }

    /// One dispatch with a raw byte argument (at most 4 KB) after the buffers, and a scalar after it.
    pub fn goBytes(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, bytes: []const u8, scalar: anytype) void {
        const e = g.encoder();
        e.setPipeline(g.pipes[@backingInt(p)]);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        e.setBytes(bytes, bufs.len);
        e.setBytes(std.mem.asBytes(&scalar), bufs.len + 1);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
        g.encoded += 1;
        g.dispatches += 1;
    }

    /// Commit what is encoded without waiting (the previous commit is waited for first).
    pub fn commit(g: *Gpu) !void {
        const e = g.enc orelse return;
        e.end();
        const cb = g.cb.?.retain();
        cb.commit();
        g.enc = null;
        g.cb = null;
        g.encoded = 0;
        defer {
            if (g.pool) |p| p.pop();
            g.pool = null;
        }
        g.waitPending() catch |err| {
            cb.wait();
            cb.release();
            return err;
        };
        g.pending = cb;
    }

    fn waitPending(g: *Gpu) !void {
        const cb = g.pending orelse return;
        g.pending = null;
        defer cb.release();
        cb.wait();
        g.gpu_s += cb.gpuSeconds();
        if (cb.failure()) |msg| {
            std.log.err("lfm3_omni metal: command buffer failed: {s}", .{msg});
            return error.CommandBufferFailed;
        }
    }

    /// Commit and wait until the GPU is done.
    pub fn flush(g: *Gpu) !void {
        try g.commit();
        try g.waitPending();
    }

    /// Host bytes into a buffer after the GPU is done with it (a flush first).
    pub fn put(g: *Gpu, dst: u64, bytes: []const u8) !void {
        try g.flush();
        @memcpy(g.host(dst, bytes.len), bytes);
    }

    /// Buffer bytes to the host once the GPU's work so far is done.
    pub fn get(g: *Gpu, src: u64, out: []u8) !void {
        try g.flush();
        @memcpy(out, g.host(src, out.len));
    }

    /// Small host bytes (whole 32-bit words, at most 4 KB) written in order with the dispatches, without a flush.
    pub fn set(g: *Gpu, dst: u64, bytes: []const u8) void {
        std.debug.assert(bytes.len % 4 == 0 and bytes.len <= 4096 and bytes.len > 0);
        const n: u32 = @intCast(bytes.len / 4);
        g.goBytes(.om_set_words, .{ (n + 255) / 256, 1, 1 }, 256, &.{dst}, bytes, n);
    }

    /// `bytes` (whole words) zeroed on the GPU, in order with the dispatches.
    pub fn zero(g: *Gpu, dst: u64, bytes: usize) void {
        std.debug.assert(bytes % 4 == 0);
        const n: u32 = @intCast(bytes / 4);
        g.go(.om_zero_words, .{ (n + 255) / 256, 1, 1 }, 256, &.{dst}, n);
    }

    /// `bytes` copied on the GPU, in order with the dispatches around it.
    pub fn copy(g: *Gpu, src: u64, dst: u64, bytes: usize) void {
        std.debug.assert(bytes % 4 == 0);
        const n: u32 = @intCast(bytes / 4);
        g.go(.om_copy_words, .{ (n + 255) / 256, 1, 1 }, 256, &.{ src, dst }, n);
    }
};
