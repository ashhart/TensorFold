//! Music 3's Metal device: one queue, the music3 metallib, and buffers addressed as u64 (slot << 40 | byte offset).
const std = @import("std");
const mtl = @import("metal");

/// Every pipeline of music3.metallib, by its kernel name.
pub const P = enum {
    m3_embed,
    m3_embed_code,
    m3_frame_embed,
    m3_rmsnorm,
    m3_add,
    m3_swiglu,
    m3_quant,
    m3_dequant,
    m3_gemv_bf16_2,
    m3_gemv_bf16_4,
    m3_gemv_i8_2,
    m3_gemv_i8_4,
    m3_gemv_g8_2,
    m3_gemv_g8_4,
    m3_glu_bf16_2,
    m3_glu_bf16_4,
    m3_glu_i8_2,
    m3_glu_i8_4,
    m3_glu_g8_2,
    m3_glu_g8_4,
    m3_qk_rope,
    m3_attn_part,
    m3_attn_merge,
    m3_depth_kv,
    m3_depth_attn,
    m3_sample,
    m3_frame_done,
    m3_state_part,
    m3_force_codes,
    m3_copy,
    m3_linear_b,
    m3_linear_h,
    m3_linear_glu_h,
    m3_linear_rope_h,
    m3_linear_s,
    m3_split,
    m3_cond_mix,
    m3_im2col3,
    m3_dit_input,
    m3_add_h,
    m3_time_features,
    m3_silu_h,
    m3_linear_h1,
    m3_gather_rows_h,
    m3_put_time,
    m3_layernorm_h,
    m3_bias_h,
    m3_dit_gather,
    m3_euler,
    m3_noise,
    m3_dit_attn,
    m3_snake,
    m3_bias_rows,
    m3_overlap_add,
    m3_zero,
    m3_latent_half,
    m3_tanh,
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
    bufs: std.ArrayList(mtl.Buffer), // by slot; slot 0 is a small zeroed buffer bound for address 0 (an unused input)
    owned: std.ArrayList(bool),
    pool: ?mtl.objc.Pool = null,
    cb: ?mtl.CommandBuffer = null,
    enc: ?mtl.ComputeEncoder = null,
    pending: ?mtl.CommandBuffer = null, // committed, not yet waited for
    peak: u64 = 0,

    pub fn open(gpa: std.mem.Allocator, metallib: []const u8) !*Gpu {
        const g = try gpa.create(Gpu);
        errdefer gpa.destroy(g);
        g.* = .{ .gpa = gpa, .device = try mtl.Device.init(), .queue = undefined, .lib = undefined, .pipes = undefined, .bufs = .empty, .owned = .empty };
        if (!g.device.tensorUnits()) {
            std.log.err("music3 metal: this GPU has no tensor units; the GEMMs need an M5 or later", .{});
            return error.NeedsTensorUnits;
        }
        g.queue = try g.device.queue();
        g.lib = try mtl.Library.fromBytes(g.device, metallib);
        inline for (@typeInfo(P).@"enum".field_names, 0..) |name, i| g.pipes[i] = try mtl.Pipeline.init(g.device, g.lib, name, false);
        const zero = try g.device.buffer(256, mtl.ResourceOptions.shared);
        @memset(zero.contents()[0..256], 0);
        try g.bufs.append(gpa, zero);
        try g.owned.append(gpa, true);
        return g;
    }

    pub fn close(g: *Gpu) void {
        g.flush() catch {};
        for (g.bufs.items, g.owned.items) |b, own| if (own) b.deinit();
        g.bufs.deinit(g.gpa);
        g.owned.deinit(g.gpa);
        for (&g.pipes) |*p| p.deinit();
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
        g.gpa.destroy(g);
    }

    /// A new zeroed shared buffer of `bytes`, by address.
    pub fn alloc(g: *Gpu, bytes: usize) !u64 {
        const len = std.mem.alignForward(usize, @max(bytes, 256), 256);
        const b = try g.device.buffer(len, mtl.ResourceOptions.shared);
        @memset(b.contents()[0..len], 0);
        g.peak = @max(g.peak, g.device.allocated());
        return g.register(b, true) catch |err| {
            b.deinit();
            return err;
        };
    }

    /// Release a buffer from alloc (its slot stays, empty); the GPU must be done with it.
    pub fn free(g: *Gpu, addr: u64) void {
        const slot: usize = @intCast(addr >> slot_shift);
        if (slot == 0 or !g.owned.items[slot]) return;
        g.bufs.items[slot].deinit();
        g.owned.items[slot] = false;
    }

    pub fn register(g: *Gpu, b: mtl.Buffer, own: bool) !u64 {
        try g.bufs.ensureUnusedCapacity(g.gpa, 1);
        try g.owned.ensureUnusedCapacity(g.gpa, 1);
        g.bufs.appendAssumeCapacity(b);
        g.owned.appendAssumeCapacity(own);
        return @as(u64, g.bufs.items.len - 1) << slot_shift;
    }

    pub fn ref(g: *const Gpu, addr: u64) Ref {
        return .{ .buf = g.bufs.items[@intCast(addr >> slot_shift)], .off = @intCast(addr & ((@as(u64, 1) << slot_shift) - 1)) };
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
        g.enc = e;
        return e;
    }

    /// One dispatch: buffers bound at 0.., then `args` (an extern struct, a scalar, or void) at the next index.
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

    /// An elementwise launch: n threads in groups of 256.
    pub fn flat(g: *Gpu, p: P, n: usize, bufs: []const u64, args: anytype) void {
        g.go(p, .{ @max(1, (n + 255) / 256), 1, 1 }, 256, bufs, args);
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
            std.log.err("music3 metal: command buffer failed: {s}", .{msg});
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
        g.flat(.m3_copy, n, &.{ src, dst }, n);
    }
};
