//! Clean-room Metal 4 runtime backend for Apple Silicon (M5 Max+).
//!
//! Owns the MTLDevice, MTL4CommandQueue, MTL4CommandAllocator, recycled MTL4CommandBuffers,
//! MTL4ArgumentTable, constant-data bump ring, MTLResidencySet, and optional MTL4CounterHeap.
const std = @import("std");
const objc = @import("objc.zig");

const Id = objc.Id;

extern "c" fn MTLCreateSystemDefaultDevice() ?Id;

pub const Error = error{
    NoMetalDevice,
    NoMetal4Support,
    BufferAllocFailed,
    ResidencyFailed,
    ShaderCompileFailed,
    FunctionNotFound,
    PipelineFailed,
    GpuTimeout,
};

pub const Size = extern struct {
    width: usize = 1,
    height: usize = 1,
    depth: usize = 1,
};

pub const Range = extern struct {
    location: usize,
    length: usize,
};

// Metal resource & stage constants
const resource_storage_shared: usize = 0;
const resource_hazard_untracked: usize = 1 << 8;
const stage_dispatch: usize = 1 << 27; // MTLStageDispatch
const visibility_device: usize = 1; // MTL4VisibilityOptionDevice

pub const MathPrecision = enum(isize) {
    fast = 0,
    precise = 1,
};

pub const Buffer = struct {
    id: Id,
    gpu_addr: u64,
    ptr: [*]u8,
    len: usize,

    pub fn slice(self: Buffer, comptime T: type, count: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0..count];
    }

    pub fn view(self: Buffer, offset: usize) BufferRef {
        return .{ .buf_id = self.id, .offset = offset, .gpu_addr = self.gpu_addr + offset, .ptr = self.ptr + offset };
    }

    pub fn deinit(self: Buffer) void {
        objc.release(self.id);
    }
};

pub const BufferRef = struct {
    buf_id: Id,
    offset: usize,
    gpu_addr: u64,
    ptr: [*]u8,

    pub fn at(self: BufferRef, byte_offset: usize) BufferRef {
        return .{ .buf_id = self.buf_id, .offset = self.offset + byte_offset, .gpu_addr = self.gpu_addr + byte_offset, .ptr = self.ptr + byte_offset };
    }
};

pub const Library = struct {
    id: Id,

    pub fn deinit(self: Library) void {
        objc.release(self.id);
    }
};

pub const Pipeline = struct {
    id: Id,
    name: [32]u8 = @splat(0),

    pub fn deinit(self: Pipeline) void {
        objc.release(self.id);
    }
};

pub const Fence = struct {
    id: Id,

    pub fn deinit(self: Fence) void {
        objc.release(self.id);
    }
};

pub const Timeline = struct {
    heap: Id,
    device: Id,
    names: [][32]u8,
    count: usize = 0,

    pub fn init(a: std.mem.Allocator, ctx: *const Context, capacity: usize) !Timeline {
        const desc = objc.allocInit("MTL4CounterHeapDescriptor");
        defer objc.release(desc);
        objc.send(void, desc, "setType:", .{@as(isize, 1)}); // timestamp
        objc.send(void, desc, "setCount:", .{capacity});
        var err: ?Id = null;
        const heap = objc.send(?Id, ctx.device, "newCounterHeapWithDescriptor:error:", .{ desc, &err }) orelse return error.NoMetal4Support;
        return .{
            .heap = heap,
            .device = ctx.device,
            .names = try a.alloc([32]u8, capacity),
        };
    }

    pub fn reset(self: *Timeline) void {
        self.count = 0;
    }

    pub fn readTicks(self: *const Timeline, out: []u64) void {
        const range = Range{ .location = 0, .length = self.count };
        const data = objc.send(Id, self.heap, "resolveCounterRange:", .{range});
        const words: [*]align(1) const u64 = @ptrCast(objc.send([*]const u8, data, "bytes", .{}));
        for (out[0..self.count], 0..) |*dst, i| dst.* = words[i];
    }

    pub fn frequencyHz(self: *const Timeline) f64 {
        return @floatFromInt(objc.send(u64, self.device, "queryTimestampFrequency", .{}));
    }

    pub fn deinit(self: *Timeline, a: std.mem.Allocator) void {
        a.free(self.names);
        objc.release(self.heap);
    }
};

pub const Stats = struct {
    command_buffers: u32 = 0,
    dispatches: u32 = 0,
    barriers: u32 = 0,
    fences: u32 = 0,
};

pub const BackendApi = enum {
    metal4_device,
    metal4_none,
    metal3_serial,
    metal3_concurrent,
};

pub const Context = struct {
    pub const max_cmd_buffers = 32;
    pub const constants_ring_bytes = 2 * 1024 * 1024; // 2 MiB constant bump ring

    api: BackendApi = .metal3_serial,
    device: Id,
    queue: ?Id = null,
    m3_queue: Id,
    allocator: ?Id = null,
    cmd_buffers: [max_cmd_buffers]?Id = @splat(null),
    cmd_used: usize = 0,
    last_m3_cb: ?Id = null,
    arg_table: ?Id = null,
    shared_event: ?Id = null,
    residency: ?Id = null,
    constants_buf: ?Buffer = null,
    constants_bump: usize = 0,
    tick: u64 = 0,
    waited_tick: u64 = 0,
    timeline: ?*Timeline = null,
    stats: Stats = .{},
    has_metal4: bool = false,
    has_tensor_ops: bool = false,
    force_simd: bool = false,

    pub fn init(residency_capacity: usize) Error!Context {
        const pool = objc.Pool.push();
        defer pool.pop();

        const device = MTLCreateSystemDefaultDevice() orelse return error.NoMetalDevice;
        const m3_queue = objc.send(?Id, device, "newCommandQueue", .{}) orelse return error.NoMetalDevice;

        var ctx = Context{
            .device = device,
            .m3_queue = m3_queue,
        };

        // Optional MTLResidencySet (Metal 3.2+ / macOS 15+)
        if (objc.responds(device, "newResidencySetWithDescriptor:error:")) {
            if (objc.allocInitOpt("MTLResidencySetDescriptor")) |res_desc| {
                defer objc.release(res_desc);
                objc.send(void, res_desc, "setInitialCapacity:", .{residency_capacity});
                var err: ?Id = null;
                if (objc.send(?Id, device, "newResidencySetWithDescriptor:error:", .{ res_desc, &err })) |res| {
                    ctx.residency = res;
                    if (objc.responds(m3_queue, "addResidencySet:")) {
                        objc.send(void, m3_queue, "addResidencySet:", .{res});
                    }
                }
            }
        }

        // Optional Metal 4 command path (macOS 26+ / M5+); gracefully falls back to Metal 3 on M1-M4 / macOS 14-15!
        if (objc.responds(device, "newMTL4CommandQueue")) {
            const q_opt = objc.send(?Id, device, "newMTL4CommandQueue", .{});
            const a_opt = if (q_opt != null) objc.send(?Id, device, "newCommandAllocator", .{}) else null;
            const ev_opt = if (a_opt != null) objc.send(?Id, device, "newSharedEvent", .{}) else null;
            var tbl_opt: ?Id = null;
            if (ev_opt != null) {
                if (objc.allocInitOpt("MTL4ArgumentTableDescriptor")) |tbl_desc| {
                    defer objc.release(tbl_desc);
                    objc.send(void, tbl_desc, "setMaxBufferBindCount:", .{@as(usize, 31)});
                    var err: ?Id = null;
                    tbl_opt = objc.send(?Id, device, "newArgumentTableWithDescriptor:error:", .{ tbl_desc, &err });
                }
            }

            if (q_opt != null and a_opt != null and ev_opt != null and tbl_opt != null) {
                var all_cbs_ok = true;
                for (&ctx.cmd_buffers) |*cb| {
                    cb.* = objc.send(?Id, device, "newCommandBuffer", .{});
                    if (cb.* == null) {
                        all_cbs_ok = false;
                        break;
                    }
                }
                if (all_cbs_ok) {
                    ctx.queue = q_opt;
                    ctx.allocator = a_opt;
                    ctx.shared_event = ev_opt;
                    ctx.arg_table = tbl_opt;
                    if (ctx.residency) |res| {
                        objc.send(void, q_opt.?, "addResidencySet:", .{res});
                    }
                    ctx.constants_buf = try ctx.allocBuffer(constants_ring_bytes);
                    ctx.has_metal4 = true;
                } else {
                    for (&ctx.cmd_buffers) |*cb| {
                        if (cb.*) |c| objc.release(c);
                        cb.* = null;
                    }
                    objc.release(tbl_opt.?);
                    objc.release(ev_opt.?);
                    objc.release(a_opt.?);
                    objc.release(q_opt.?);
                }
            } else {
                if (tbl_opt) |t| objc.release(t);
                if (ev_opt) |e| objc.release(e);
                if (a_opt) |a| objc.release(a);
                if (q_opt) |q| objc.release(q);
            }
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.last_m3_cb) |cb| objc.release(cb);
        if (self.constants_buf) |cb| cb.deinit();
        for (self.cmd_buffers) |cb_opt| {
            if (cb_opt) |cb| objc.release(cb);
        }
        if (self.residency) |r| objc.release(r);
        if (self.shared_event) |e| objc.release(e);
        if (self.arg_table) |t| objc.release(t);
        if (self.allocator) |a| objc.release(a);
        objc.release(self.m3_queue);
        if (self.queue) |q| objc.release(q);
        objc.release(self.device);
    }

    pub fn isMetal4(self: *const Context) bool {
        return self.has_metal4 and (self.api == .metal4_device or self.api == .metal4_none);
    }

    pub fn deviceName(self: *const Context) []const u8 {
        const s = objc.send(?Id, self.device, "name", .{}) orelse return "Apple GPU";
        const cstr = objc.send(?[*:0]const u8, s, "UTF8String", .{}) orelse return "Apple GPU";
        return std.mem.span(cstr);
    }

    /// Allocate a shared, hazard-untracked Metal buffer and register it in the residency set (if available).
    pub fn allocBuffer(self: *Context, byte_len: usize) Error!Buffer {
        const aligned_len = @max((byte_len + 15) & ~@as(usize, 15), 16);
        const opts = resource_storage_shared | resource_hazard_untracked;
        const id = objc.send(?Id, self.device, "newBufferWithLength:options:", .{ aligned_len, opts }) orelse return error.BufferAllocFailed;
        const gpu_addr = objc.send(u64, id, "gpuAddress", .{});
        const ptr = objc.send([*]u8, id, "contents", .{});
        if (self.residency) |res| {
            objc.send(void, res, "addAllocation:", .{id});
        }
        return .{ .id = id, .gpu_addr = gpu_addr, .ptr = ptr, .len = aligned_len };
    }

    /// Wrap an existing page-aligned (16 KiB) mmap region as a zero-copy unified-memory Metal buffer.
    pub fn allocBufferNoCopy(self: *Context, ptr: [*]u8, byte_len: usize) Error!Buffer {
        const opts = resource_storage_shared | resource_hazard_untracked;
        const id = objc.send(?Id, self.device, "newBufferWithBytesNoCopy:length:options:deallocator:", .{
            @as(*anyopaque, @ptrCast(ptr)),
            byte_len,
            opts,
            @as(?*anyopaque, null),
        }) orelse return error.BufferAllocFailed;
        const gpu_addr = objc.send(u64, id, "gpuAddress", .{});
        if (self.residency) |res| {
            objc.send(void, res, "addAllocation:", .{id});
        }
        return .{ .id = id, .gpu_addr = gpu_addr, .ptr = ptr, .len = byte_len };
    }

    /// Commit residency set so all allocated buffers are bound to command queues (pages fault lazily on GPU use).
    pub fn commitResidency(self: *Context) void {
        const res = self.residency orelse return;
        objc.send(void, res, "commit", .{});
    }

    /// Release a temporary GPU buffer and remove it from the residency set so memory is never leaked.
    pub fn freeBuffer(self: *Context, buf: Buffer) void {
        if (self.residency) |res| {
            if (objc.responds(res, "removeAllocation:")) {
                objc.send(void, res, "removeAllocation:", .{buf.id});
            }
        }
        buf.deinit();
    }

    pub fn createFence(self: *Context) Error!Fence {
        const id = objc.send(?Id, self.device, "newFence", .{}) orelse return error.NoMetal4Support;
        return .{ .id = id };
    }

    /// Compile universal Metal 3.1+ shader source (works on all Apple Silicon M1-M5, macOS 14+).
    pub fn compileSource(self: *const Context, source: []const u8, precision: MathPrecision) Error!Library {
        return self.compileSourceVersion(source, precision, 3, 1, true);
    }

    /// Compile optional Metal 4.1 shader source (M5 hardware tensor_ops); returns error silently on M1-M4.
    pub fn compileSourceM4(self: *const Context, source: []const u8, precision: MathPrecision) Error!Library {
        return self.compileSourceVersion(source, precision, 4, 1, false);
    }

    pub fn compileSourceVersion(self: *const Context, source: []const u8, precision: MathPrecision, major: usize, minor: usize, log_errors: bool) Error!Library {
        const pool = objc.Pool.push();
        defer pool.pop();

        const src = objc.nsString(source);
        defer objc.release(src);
        const opts = objc.allocInit("MTLCompileOptions");
        defer objc.release(opts);

        objc.send(void, opts, "setLanguageVersion:", .{@as(usize, (major << 16) | minor)});
        if (objc.responds(opts, "setMathMode:")) {
            objc.send(void, opts, "setMathMode:", .{@as(isize, 0)}); // safe IEEE math
        }
        if (objc.responds(opts, "setMathFloatingPointFunctions:")) {
            objc.send(void, opts, "setMathFloatingPointFunctions:", .{@intFromEnum(precision)});
        }

        var err: ?Id = null;
        const lib = objc.send(?Id, self.device, "newLibraryWithSource:options:error:", .{ src, opts, &err }) orelse {
            if (log_errors) {
                std.debug.print("Metal compile error (v{d}.{d}):\n{s}\n", .{ major, minor, objc.errorMessage(err) });
            }
            return error.ShaderCompileFailed;
        };
        return .{ .id = lib };
    }

    pub fn makePipeline(self: *const Context, lib: Library, name: []const u8) Error!Pipeline {
        const pool = objc.Pool.push();
        defer pool.pop();

        const fn_name = objc.nsString(name);
        defer objc.release(fn_name);
        const func = objc.send(?Id, lib.id, "newFunctionWithName:", .{fn_name}) orelse {
            std.debug.print("Metal kernel not found: {s}\n", .{name});
            return error.FunctionNotFound;
        };
        defer objc.release(func);

        const desc = objc.allocInit("MTLComputePipelineDescriptor");
        defer objc.release(desc);
        objc.send(void, desc, "setComputeFunction:", .{func});

        var err: ?Id = null;
        const pso = objc.send(?Id, self.device, "newComputePipelineStateWithDescriptor:options:reflection:error:", .{
            desc,
            @as(usize, 0),
            @as(?*anyopaque, null),
            &err,
        }) orelse {
            std.debug.print("Metal pipeline build failed ({s}): {s}\n", .{ name, objc.errorMessage(err) });
            return error.PipelineFailed;
        };
        var out = Pipeline{ .id = pso };
        const n = @min(name.len, out.name.len);
        @memcpy(out.name[0..n], name[0..n]);
        return out;
    }

    pub fn beginCommandBuffer(self: *Context) CommandBuffer {
        self.stats.command_buffers += 1;
        if (!self.isMetal4()) {
            const cb = objc.send(Id, self.m3_queue, "commandBufferWithUnretainedReferences", .{});
            _ = objc.retain(cb);
            if (self.last_m3_cb) |old| objc.release(old);
            self.last_m3_cb = cb;
            return .{ .ctx = self, .id = cb, .is_m4 = false };
        }
        if (self.waited_tick == self.tick) {
            objc.send(void, self.allocator.?, "reset", .{});
            self.cmd_used = 0;
        }
        std.debug.assert(self.cmd_used < max_cmd_buffers);
        const cb = self.cmd_buffers[self.cmd_used].?;
        self.cmd_used += 1;
        objc.send(void, cb, "beginCommandBufferWithAllocator:", .{self.allocator.?});
        return .{ .ctx = self, .id = cb, .is_m4 = true };
    }

    pub fn waitIdle(self: *Context) Error!void {
        if (!self.isMetal4()) {
            if (self.last_m3_cb) |cb| {
                objc.send(void, cb, "waitUntilCompleted", .{});
            }
            return;
        }
        if (self.waited_tick == self.tick) return;
        const ok = objc.send(bool, self.shared_event.?, "waitUntilSignaledValue:timeoutMS:", .{ self.tick, @as(u64, 30_000) });
        if (!ok) return error.GpuTimeout;
        self.waited_tick = self.tick;
    }
};

pub const CommandBuffer = struct {
    ctx: *Context,
    id: Id,
    is_m4: bool = true,

    pub fn beginCompute(self: CommandBuffer) Encoder {
        if (!self.is_m4) {
            const dtype: usize = if (self.ctx.api == .metal3_concurrent) 1 else 0;
            const enc = objc.send(Id, self.id, "computeCommandEncoderWithDispatchType:", .{dtype});
            return .{ .ctx = self.ctx, .id = enc, .is_m4 = false };
        }
        const enc = objc.send(Id, self.id, "computeCommandEncoder", .{});
        objc.send(void, enc, "setArgumentTable:", .{self.ctx.arg_table.?});
        var e = Encoder{ .ctx = self.ctx, .id = enc, .is_m4 = true };
        if (self.ctx.timeline) |tl| {
            if (tl.count == 0) e.writeTimestamp("start");
        }
        return e;
    }

    pub fn commit(self: CommandBuffer) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "commit", .{});
            return;
        }
        objc.send(void, self.id, "endCommandBuffer", .{});
        const list = [1]Id{self.id};
        objc.send(void, self.ctx.queue.?, "commit:count:", .{ @as([*]const Id, &list), @as(usize, 1) });
        self.ctx.tick += 1;
        objc.send(void, self.ctx.queue.?, "signalEvent:value:", .{ self.ctx.shared_event.?, self.ctx.tick });
    }
};

pub const Encoder = struct {
    ctx: *Context,
    id: Id,
    is_m4: bool = true,
    dirty: bool = false,
    last_label: [32]u8 = @splat(0),

    pub fn setPipeline(self: *Encoder, p: Pipeline) void {
        self.last_label = p.name;
        objc.send(void, self.id, "setComputePipelineState:", .{p.id});
    }

    pub fn setBuffer(self: *Encoder, buf: Buffer, offset: usize, index: usize) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "setBuffer:offset:atIndex:", .{ buf.id, offset, index });
            return;
        }
        objc.send(void, self.ctx.arg_table.?, "setAddress:atIndex:", .{ buf.gpu_addr + offset, index });
    }

    pub fn setRef(self: *Encoder, ref: BufferRef, index: usize) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "setBuffer:offset:atIndex:", .{ ref.buf_id, ref.offset, index });
            return;
        }
        objc.send(void, self.ctx.arg_table.?, "setAddress:atIndex:", .{ ref.gpu_addr, index });
    }

    /// Copy small POD constants into the Metal 4 ring buffer and bind its GPU address.
    pub fn setValue(self: *Encoder, value: anytype, index: usize) void {
        const bytes = std.mem.asBytes(&value);
        if (!self.is_m4) {
            objc.send(void, self.id, "setBytes:length:atIndex:", .{ @as([*]const u8, bytes.ptr), bytes.len, index });
            return;
        }
        const cbuf = self.ctx.constants_buf.?;
        if (self.ctx.constants_bump + bytes.len > cbuf.len) {
            self.ctx.constants_bump = 0;
        }
        const off = self.ctx.constants_bump;
        @memcpy(cbuf.ptr[off..][0..bytes.len], bytes);
        objc.send(void, self.ctx.arg_table.?, "setAddress:atIndex:", .{ cbuf.gpu_addr + off, index });
        self.ctx.constants_bump += (bytes.len + 63) & ~@as(usize, 63);
    }

    pub fn dispatchGroups(self: *Encoder, groups: Size, threads: Size) void {
        objc.send(void, self.id, "dispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
        self.ctx.stats.dispatches += 1;
        self.dirty = true;
    }

    pub fn dispatchThreads(self: *Encoder, grid: Size, threads: Size) void {
        objc.send(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
        self.ctx.stats.dispatches += 1;
        self.dirty = true;
    }

    /// Device-coherent dispatch-to-dispatch barrier (elided if nothing dispatched since previous barrier).
    pub fn barrier(self: *Encoder) void {
        if (!self.dirty) return;
        self.dirty = false;
        if (!self.is_m4) {
            if (self.ctx.api == .metal3_concurrent) {
                self.ctx.stats.barriers += 1;
                objc.send(void, self.id, "memoryBarrierWithScope:", .{@as(usize, 1)}); // MTLBarrierScopeBuffers
            }
            return;
        }
        self.ctx.stats.barriers += 1;
        const vis: usize = if (self.ctx.api == .metal4_device) visibility_device else 0;
        objc.send(void, self.id, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{
            stage_dispatch,
            stage_dispatch,
            vis,
        });
        if (self.ctx.timeline != null) self.writeTimestamp(std.mem.sliceTo(&self.last_label, 0));
    }

    pub fn updateFence(self: *Encoder, fence: Fence) void {
        self.ctx.stats.fences += 1;
        if (!self.is_m4) {
            objc.send(void, self.id, "updateFence:", .{fence.id});
            return;
        }
        objc.send(void, self.id, "updateFence:afterEncoderStages:", .{ fence.id, stage_dispatch });
    }

    pub fn waitFence(self: *Encoder, fence: Fence) void {
        if (!self.is_m4) {
            objc.send(void, self.id, "waitForFence:", .{fence.id});
            return;
        }
        objc.send(void, self.id, "waitForFence:beforeEncoderStages:", .{ fence.id, stage_dispatch });
    }

    pub fn writeTimestamp(self: *Encoder, label: []const u8) void {
        if (!self.is_m4) return;
        const tl = self.ctx.timeline orelse return;
        if (tl.count >= tl.names.len) return;
        tl.names[tl.count] = @splat(0);
        const n = @min(label.len, 32);
        @memcpy(tl.names[tl.count][0..n], label[0..n]);
        objc.send(void, self.id, "writeTimestampWithGranularity:intoHeap:atIndex:", .{
            @as(isize, 1),
            tl.heap,
            tl.count,
        });
        tl.count += 1;
    }

    pub fn end(self: *Encoder) void {
        if (self.is_m4 and self.dirty and self.ctx.timeline != null) {
            self.writeTimestamp(std.mem.sliceTo(&self.last_label, 0));
        }
        objc.send(void, self.id, "endEncoding", .{});
    }
};
