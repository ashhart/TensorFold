//! Drama 3's Metal device: one queue, the drama3 metallib, and buffers addressed as u64 (slot << 40 | byte offset).
const std = @import("std");
const mtl = @import("metal");

/// Every pipeline of drama3.metallib, by its kernel name.
pub const P = enum {
    d3_rows_r1,
    d3_rows_r8,
    d3_rmsnorm,
    d3_add_rmsnorm,
    d3_embed_backbone,
    d3_swiglu,
    d3_moe_combine,
    d3_tower_mix,
    d3_conv_silu,
    d3_gdn,
    d3_gated_norm,
    d3_qk_rope,
    d3_attn,
    d3_router,
    d3_act_quant,
    d3_plan,
    d3_expert_up,
    d3_expert_down,
    d3_argmax,
    d3_sample_main,
    d3_sample_code,
    d3_finish_frame,
    d3_next_input,
    d3_copy_i32,
    c_gemm,
    c_vq_gather,
    c_vq_encode,
    c_rmsnorm,
    c_layernorm,
    c_rope_kv,
    c_attn,
    c_add_gamma,
    c_silu_mul,
    c_gelu,
    c_bias,
    c_snake,
    c_dwconv7,
    c_copy,
};

const n_pipes = @typeInfo(P).@"enum".field_names.len;
const slot_shift = 40;

pub const Ref = struct { buf: mtl.Buffer, off: usize };

pub const Gpu = struct {
    follow: ?mtl.ResidencySet.Follow = null,
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    pipes: [n_pipes]mtl.Pipeline,
    bufs: std.ArrayList(mtl.Buffer), // by slot; slot 0 is never used, so address 0 is null
    owned: std.ArrayList(bool),
    indirect: std.ArrayList(mtl.Buffer), // read through pointer tables: declared on every encoder
    pool: ?mtl.objc.Pool = null,
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,
    pending: ?mtl.CommandBuffer = null, // committed, not yet waited for
    peak: u64 = 0,

    pub fn open(gpa: std.mem.Allocator, metallib: []const u8) !*Gpu {
        const g = try gpa.create(Gpu);
        errdefer gpa.destroy(g);
        g.* = .{ .gpa = gpa, .device = try mtl.Device.init(), .queue = undefined, .lib = undefined, .pipes = undefined, .bufs = .empty, .owned = .empty, .indirect = .empty };
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
        errdefer g.owned.deinit(gpa);
        try g.bufs.append(gpa, undefined);
        try g.owned.append(gpa, false);
        return g;
    }

    pub fn setResident(g: *Gpu, hook: ?mtl.ResidencySet.Follow) void {
        for (g.bufs.items[1..]) |buffer| {
            if (g.follow) |f| f.leave(f.ctx, buffer);
            if (hook) |f| f.join(f.ctx, buffer);
        }
        g.follow = hook;
    }

    pub fn releaseFrom(g: *Gpu, first: usize) void {
        g.flush() catch {};
        std.debug.assert(first >= 1 and first <= g.bufs.items.len);
        for (g.bufs.items[first..], g.owned.items[first..]) |buffer, own| {
            const retained = for (g.bufs.items[1..first]) |kept| {
                if (kept.id == buffer.id) break true;
            } else false;
            if (!retained) if (g.follow) |f| f.leave(f.ctx, buffer);
            if (own) buffer.deinit();
        }
        g.bufs.shrinkRetainingCapacity(first);
        g.owned.shrinkRetainingCapacity(first);
    }

    pub fn close(g: *Gpu) void {
        g.flush() catch {};
        g.setResident(null);
        for (g.bufs.items[1..], g.owned.items[1..]) |b, own| if (own) b.deinit();
        g.bufs.deinit(g.gpa);
        g.owned.deinit(g.gpa);
        g.indirect.deinit(g.gpa);
        for (&g.pipes) |*p| p.deinit();
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
        g.gpa.destroy(g);
    }

    /// A new zeroed shared buffer of `bytes`, by address.
    pub fn alloc(g: *Gpu, bytes: usize) !u64 {
        if (bytes > std.math.maxInt(usize) - 255) return error.OutOfMemory;
        const len = std.mem.alignForward(usize, @max(bytes, 256), 256);
        const cap = g.device.maxWorkingSet();
        if (len > cap -| g.device.allocated()) return error.OutOfMemory;
        const b = try g.device.buffer(len, mtl.ResourceOptions.shared);
        @memset(b.contents()[0..len], 0);
        g.peak = @max(g.peak, g.device.allocated());
        return g.register(b, true) catch |err| {
            b.deinit();
            return err;
        };
    }

    /// An address for a buffer made elsewhere (a mapped checkpoint shard); `own` releases it at close.
    pub fn register(g: *Gpu, b: mtl.Buffer, own: bool) !u64 {
        if (g.device.allocated() > g.device.maxWorkingSet()) return error.OutOfMemory;
        // both lists grow first, so a failed allocation leaves them the same length and the caller owning `b`
        try g.bufs.ensureUnusedCapacity(g.gpa, 1);
        try g.owned.ensureUnusedCapacity(g.gpa, 1);
        g.bufs.appendAssumeCapacity(b);
        g.owned.appendAssumeCapacity(own);
        if (g.follow) |f| f.join(f.ctx, b);
        return @as(u64, g.bufs.items.len - 1) << slot_shift;
    }

    pub fn ref(g: *const Gpu, addr: u64) Ref {
        return .{ .buf = g.bufs.items[@intCast(addr >> slot_shift)], .off = @intCast(addr & ((@as(u64, 1) << slot_shift) - 1)) };
    }

    /// The GPU address of `addr` (for pointer tables the kernels read).
    pub fn gpuAddress(g: *const Gpu, addr: u64) u64 {
        const r = g.ref(addr);
        return r.buf.gpuAddress() + r.off;
    }

    /// The host view of `len` bytes at `addr` (unified memory; flush first when the GPU writes them).
    pub fn host(g: *const Gpu, addr: u64, len: usize) []u8 {
        const r = g.ref(addr);
        return r.buf.contents()[r.off..][0..len];
    }

    /// The encoder of the open command buffer (made on first use).
    pub fn encoder(g: *Gpu) mtl.ComputeEncoder {
        if (g.enc) |e| return e;
        if (g.pool == null) g.pool = mtl.objc.Pool.push();
        const cb = g.queue.commandBuffer();
        g.cb = cb;
        const e = cb.compute(.serial);
        for (g.indirect.items) |b| e.useResource(b, mtl.ResourceUsage.read);
        g.enc = e;
        return e;
    }

    /// One dispatch: buffers bound at 0.., then `args` (an extern struct, or void) at the next index.
    pub fn go(g: *Gpu, p: P, groups: [3]usize, threads: usize, bufs: []const u64, args: anytype) void {
        const e = g.encoder();
        e.setPipeline(g.pipes[@backingInt(p)]);
        for (bufs, 0..) |a, i| {
            const r = g.ref(a);
            e.setBuffer(r.buf, r.off, i);
        }
        if (@TypeOf(args) != void) e.setBytes(std.mem.asBytes(&args), bufs.len);
        e.dispatchGroups(mtl.Size.of(groups[0], groups[1], groups[2]), mtl.Size.of(threads, 1, 1));
    }

    /// Commit what is encoded without waiting (the previous commit is waited for first).
    pub fn commit(g: *Gpu) !void {
        const e = g.enc orelse return;
        e.end();
        const cb = g.cb.?.retain();
        cb.commit();
        g.enc = null;
        g.cb = null;
        defer {
            if (g.pool) |p| p.pop();
            g.pool = null;
        }
        g.waitPending() catch |err| {
            // the new command is already queued: let it finish before anyone can release its buffers
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
        if (cb.failure()) |msg| {
            std.log.err("drama3 metal: command buffer failed: {s}", .{msg});
            return error.CommandBufferFailed;
        }
    }

    /// Commit and wait until the GPU is done.
    pub fn flush(g: *Gpu) !void {
        try g.commit();
        try g.waitPending();
        g.peak = @max(g.peak, g.device.allocated());
    }

    pub fn put(g: *Gpu, dst: u64, bytes: []const u8) !void {
        try g.flush();
        @memcpy(g.host(dst, bytes.len), bytes);
    }

    pub fn get(g: *Gpu, src: u64, out: []u8) !void {
        try g.flush();
        @memcpy(out, g.host(src, out.len));
    }

    /// Copy `bytes` on the GPU, in order with the dispatches around it.
    pub fn copy(g: *Gpu, src: u64, dst: u64, bytes: usize) void {
        std.debug.assert(bytes % 4 == 0);
        const n: u32 = @intCast(bytes / 4);
        g.go(.c_copy, .{ (n + 255) / 256, 1, 1 }, 256, &.{ src, dst }, n);
    }
};
