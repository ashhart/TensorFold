//! One command buffer and fence on the device's queue, and staged uploads and readbacks through it.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const dispatch = @import("dispatch.zig");
const Device = @import("device.zig").Device;
const Buffer = memory.Buffer;
const Error = runtime.Error;
const check = runtime.check;

pub const Commands = struct {
    dev: *const Device,
    pool: c.VkCommandPool,
    cmd: c.VkCommandBuffer,
    fence: c.VkFence,
    // Batch session: when `session` is set, begin() hands out the already
    // open command buffer and submit() is a no-op, so every op wrapper only
    // records; endBatch performs the one flush/submit/fence cycle. Staging
    // slices are bump-allocated per batch (never offset 0 twice) and every
    // recorded readback lands in its own down slice, read only after the
    // batch's fence.
    session: bool = false,
    lifecycle: BatchLifecycle = .{},
    ledger: BatchLedger = .{},
    batch_staging: Buffer = undefined,

    pub fn init(dev: *const Device) Error!Commands {
        const api = &dev.api;
        const pool_info: c.VkCommandPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = dev.queue_family,
        };
        var pool: c.VkCommandPool = null;
        try check(api.vkCreateCommandPool(dev.handle, &pool_info, null, &pool));
        errdefer api.vkDestroyCommandPool(dev.handle, pool, null);
        const alloc: c.VkCommandBufferAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var cmd: c.VkCommandBuffer = null;
        try check(api.vkAllocateCommandBuffers(dev.handle, &alloc, &cmd));
        const fence_info: c.VkFenceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        var fence: c.VkFence = null;
        try check(api.vkCreateFence(dev.handle, &fence_info, null, &fence));
        return .{ .dev = dev, .pool = pool, .cmd = cmd, .fence = fence };
    }

    /// Frees the pool, its command buffer and the fence; call only after the last submit returned success.
    pub fn deinit(self: *Commands) void {
        self.dev.api.vkDestroyFence(self.dev.handle, self.fence, null);
        self.dev.api.vkDestroyCommandPool(self.dev.handle, self.pool, null);
        self.* = undefined;
    }

    /// The command buffer, reset and begun for one submission. Inside an
    /// open batch this is the batch's already-begun command buffer: no reset,
    /// no second begin, no second packed record.
    pub fn begin(self: *Commands) Error!c.VkCommandBuffer {
        if (self.session) return self.cmd;
        try check(self.dev.api.vkResetCommandBuffer(self.cmd, 0));
        const info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        try check(self.dev.api.vkBeginCommandBuffer(self.cmd, &info));
        return self.cmd;
    }

    /// Finite fence wait; Timeout/InFlight require orExit to avoid freeing resources the GPU may still use.
    /// Inside an open batch this records nothing and submits nothing: the
    /// batch's endBatch performs the single physical submit and fence wait.
    pub fn submit(self: *Commands, timeout_ns: u64) Error!void {
        if (self.session) return;
        try self.submitAndWait(timeout_ns);
    }

    /// The physical submission path: end, one vkQueueSubmit, one finite
    /// fence wait, fence reset. Only ever called with no batch session open.
    fn submitAndWait(self: *Commands, timeout_ns: u64) Error!void {
        const api = &self.dev.api;
        try check(api.vkEndCommandBuffer(self.cmd));
        const info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.cmd,
        };
        const queued = api.vkQueueSubmit(self.dev.queue, 1, &info, self.fence);
        submitted(queued) catch |e| {
            std.log.err("vulkan: vkQueueSubmit returned {d}", .{queued});
            return e;
        };
        const waited_on = api.vkWaitForFences(self.dev.handle, 1, &self.fence, 1, timeout_ns);
        waited(waited_on) catch |e| {
            std.log.err("vulkan: vkWaitForFences returned {d} within {d} ns of a submission", .{ waited_on, timeout_ns });
            return e;
        };
        try check(api.vkResetFences(self.dev.handle, 1, &self.fence));
    }

    // ------------------------------------------------ batch session

    /// Opens a batch: one reset+begin and fresh staging ledgers against `staging`. Nested batches
    /// are refused. The physical reset+begin happens BEFORE the session and
    /// the global batch contexts are considered open, and any begin failure
    /// rolls those contexts back, so recording never rides a stale command
    /// buffer. Batched descriptor banks key on the new epoch, so every
    /// pipeline's snapshot cursor restarts only here — always after the
    /// previous batch's fence has been waited.
    pub fn beginBatch(self: *Commands, staging: Buffer) Error!void {
        if (self.session) return error.Invalid;
        if (staging.dev != self.dev or staging.kind != .staging or staging.mapped == null) return error.Invalid;
        try self.lifecycle.start();
        dispatch.batchBegin();
        errdefer { // begin failed before anything was recorded: undo the global batch contexts
            dispatch.batchEnd();
            self.lifecycle.finish();
        }
        try check(self.dev.api.vkResetCommandBuffer(self.cmd, 0));
        const info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        try check(self.dev.api.vkBeginCommandBuffer(self.cmd, &info));
        self.lifecycle.began(); // recording may start: the begin happened exactly once
        self.session = true;
        self.ledger = .{ .capacity = staging.len };
        self.batch_staging = staging;
    }

    /// Records the one physical submission of the open batch: the staging
    /// flush, one end/submit/fence-wait cycle for the whole token, then the
    /// staging invalidate so the host can read the down slices. The batch
    /// contexts close on EVERY path — a pre-submit flush/end/submit failure
    /// unconditionally closes them, so the next call starts clean; the fence
    /// discipline is unchanged (Timeout/InFlight still demand orExit at the
    /// call site).
    pub fn endBatch(self: *Commands, timeout_ns: u64) Error!void {
        if (!self.session) return error.Invalid;
        self.session = false;
        dispatch.batchEnd();
        defer self.lifecycle.finish();
        try self.batch_staging.flush(); // host writes visible to the copy engine before the submit
        try self.submitAndWait(timeout_ns);
        try self.batch_staging.invalidate(); // device writes visible to the host, after the fence
    }

    /// Abandons an open batch after a pre-submit recording error: nothing
    /// was submitted, so a discard is clean. The next begin resets the
    /// command buffer anyway; the ledger restarts at the next beginBatch.
    /// The engine's poison rule (a failure after the first state write is
    /// never reused without a real reset) is unchanged and caller-owned.
    pub fn abortBatch(self: *Commands) void {
        if (!self.session) return;
        self.session = false;
        dispatch.batchEnd();
        self.lifecycle.finish();
    }

    /// The host side of a recorded down slice: copy `out.len` bytes from the
    /// batch staging at `at` after endBatch's fence. Only a slice a
    /// batchDownload recorded in THIS batch, at exactly its recorded offset
    /// and length, is readable — anything else refuses before the mapping is
    /// touched, so a stale or torn host read is impossible by construction.
    pub fn readDown(self: *Commands, at: u64, out: []u8) Error!void {
        if (self.session) return error.Invalid; // never read staging mid-batch
        for (self.ledger.downs[0..self.ledger.down_n]) |d| {
            if (d.at == at and d.len == out.len) break;
        } else return error.Invalid; // not a recorded down slice of this batch
        const host = try self.batch_staging.bytes();
        try memory.span(host.len, at, out.len);
        @memcpy(out, host[at..][0..out.len]);
    }
};

/// The pure ordering contract of a batch's lifecycle, host-tested: the
/// physical command-buffer begin happens at most once per batch and never
/// zero times before recording, nesting refuses, and every path out of the
/// batch clears the open/begun state so the next batch starts clean.
pub const BatchLifecycle = struct {
    open: bool = false,
    begun: bool = false,
    begins: u32 = 0, // physical reset+begin count, for the mock regression

    pub fn start(self: *BatchLifecycle) Error!void {
        if (self.open) return error.Invalid;
        self.open = true;
        self.begun = false;
    }

    /// Called right after the real reset+begin succeeded.
    pub fn began(self: *BatchLifecycle) void {
        self.begins += 1;
        self.begun = true;
    }

    pub fn finish(self: *BatchLifecycle) void {
        self.open = false;
        self.begun = false;
    }
};

/// The pure ledger behind a batch's staging slices: every upload and every
/// recorded readback gets its own immutable slice, in order, never
/// overlapping, never past the staging buffer. Pure and host-tested.
pub const max_batch_downs: u32 = 4;

pub const BatchSlice = struct { at: u64, len: u64 };

pub const BatchLedger = struct {
    capacity: u64 = 0,
    up: u64 = 0,
    downs: [max_batch_downs]BatchSlice = undefined,
    down_n: u32 = 0,

    /// Reserves the next upload slice at the bump cursor; a range past the
    /// staging buffer refuses before any host write.
    pub fn reserveUp(self: *BatchLedger, len: u64) Error!u64 {
        try memory.span(self.capacity, self.up, len);
        const at = self.up;
        self.up += len;
        return at;
    }

    /// Reserves the next readback slice, distinct from every other slice in
    /// the batch; refuses when the batch needs more than max_batch_downs.
    pub fn reserveDown(self: *BatchLedger, len: u64) Error!u64 {
        if (self.down_n >= max_batch_downs) return error.OutOfRange;
        const at = try self.reserveUp(len);
        self.downs[self.down_n] = .{ .at = at, .len = len };
        self.down_n += 1;
        return at;
    }
};

/// Any failed submit may leave work in flight; terminate conservatively instead of freeing its resources.
fn submitted(result: c.VkResult) Error!void {
    return switch (result) {
        c.VK_SUCCESS => {},
        else => error.InFlight,
    };
}

/// After a submit, only a signalled fence proves the GPU is done with the submission's resources.
fn waited(result: c.VkResult) Error!void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_TIMEOUT => error.Timeout,
        else => error.InFlight,
    };
}

/// Work of unknown completion leaves the GPU owning its resources: exit now (124 timeout, 1 otherwise), free nothing.
pub fn orExit(result: anytype) @TypeOf(result) {
    _ = result catch |e| {
        if (e == error.Timeout) std.process.exit(124);
        if (e == error.InFlight) std.process.exit(1);
        return e;
    };
    return result;
}

/// One global memory dependency: `src_access` by `src_stage` made visible to `dst_access` by `dst_stage`.
pub fn barrier(dev: *const Device, cmd: c.VkCommandBuffer, src_stage: u32, src_access: u32, dst_stage: u32, dst_access: u32) void {
    const b: c.VkMemoryBarrier = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
    };
    dev.api.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &b, 0, null, 0, null);
}

/// One `vkCmdFillBuffer`'s geometry: a nonzero len, both offset and len
/// multiples of 4 (the command's alignment contract), the range inside the
/// buffer. Pure and host-tested.
pub fn fillRange(buf_len: u64, offset: u64, len: u64) Error!void {
    if (len == 0 or len % 4 != 0 or offset % 4 != 0) return error.Misaligned;
    try memory.span(buf_len, offset, len);
}

/// Records a fill of the repeating 4-byte little-endian pattern of `value`
/// into a caller-begun command buffer; no submit here. `dst` must be a device
/// buffer of `dev`; offset and len are byte counts, nonzero multiples of 4
/// inside it.
pub fn recordFill(dev: *const Device, cmd: c.VkCommandBuffer, dst: Buffer, offset: u64, len: u64, value: u32) Error!void {
    if (dst.dev != dev or dst.kind != .device) return error.Invalid;
    try fillRange(dst.len, offset, len);
    dev.api.vkCmdFillBuffer(cmd, dst.handle, offset, len, value);
}

/// Fills `len` bytes of device buffer `dst` at `offset` with the repeating
/// 4-byte little-endian pattern of `value`: one submission, transfer-write
/// barriers before and after, finite fence wait. Timeout/InFlight go to the
/// caller, which must orExit rather than free anything the GPU may still use.
pub fn fill(cmds: *Commands, dst: Buffer, offset: u64, len: u64, value: u32, timeout_ns: u64) Error!void {
    if (dst.dev != cmds.dev or dst.kind != .device) return error.Invalid;
    try fillRange(dst.len, offset, len);
    const cmd = try cmds.begin();
    barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_write);
    cmds.dev.api.vkCmdFillBuffer(cmd, dst.handle, offset, len, value);
    barrier(cmds.dev, cmd, transfer, transfer_write, gpu_stages, gpu_access);
    try cmds.submit(timeout_ns);
}

/// fill(..., value = 0): the real device-side zero every family state uses.
pub fn zero(cmds: *Commands, dst: Buffer, offset: u64, len: u64, timeout_ns: u64) Error!void {
    try fill(cmds, dst, offset, len, 0, timeout_ns);
}

const gpu_stages: u32 = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
const gpu_writes: u32 = c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
const gpu_access: u32 = gpu_writes | c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_TRANSFER_READ_BIT;
const transfer: u32 = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
const transfer_write: u32 = c.VK_ACCESS_TRANSFER_WRITE_BIT;
const transfer_read: u32 = c.VK_ACCESS_TRANSFER_READ_BIT;

/// Both on `dev`, `buf` a device buffer, `staging` a separate staging buffer, and the range inside `buf`.
pub fn operands(dev: *const Device, buf: Buffer, staging: Buffer, offset: u64, n: u64) Error!void {
    if (buf.dev != dev or staging.dev != dev or buf.kind != .device or staging.kind != .staging) return error.Invalid;
    if (buf.handle == staging.handle or staging.mapped == null or staging.len == 0) return error.Invalid;
    try memory.span(buf.len, offset, n);
}

/// `bytes` into `dst` at `offset` through `staging`, a chunk a submission; later dispatches and copies see them.
/// Inside an open batch the bytes go into a fresh immutable staging slice
/// (never offset 0 twice) and only the copy is recorded — the batch's
/// single submit moves them all at once.
pub fn upload(cmds: *Commands, dst: Buffer, offset: u64, staging: Buffer, bytes: []const u8, timeout_ns: u64) Error!void {
    try operands(cmds.dev, dst, staging, offset, bytes.len);
    const host = try staging.bytes();
    if (cmds.session) {
        // batch uploads ride the batch's own staging buffer, whose ledger owns the slices
        if (staging.handle != cmds.batch_staging.handle) return error.Invalid;
        const at: usize = @intCast(try cmds.ledger.reserveUp(bytes.len));
        @memcpy(host[at..][0..bytes.len], bytes);
        const cmd = try cmds.begin();
        barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_write);
        const region: c.VkBufferCopy = .{ .srcOffset = at, .dstOffset = offset, .size = bytes.len };
        cmds.dev.api.vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
        barrier(cmds.dev, cmd, transfer, transfer_write, gpu_stages, gpu_access);
        return;
    }
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(host.len, bytes.len - done);
        @memcpy(host[0..n], bytes[done..][0..n]);
        try staging.flush();
        const cmd = try cmds.begin();
        barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_write);
        const region: c.VkBufferCopy = .{ .srcOffset = 0, .dstOffset = offset + @as(u64, done), .size = n };
        cmds.dev.api.vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
        barrier(cmds.dev, cmd, transfer, transfer_write, gpu_stages, gpu_access);
        try cmds.submit(timeout_ns);
        done += n;
    }
}

/// Records, into the OPEN batch, a copy of `len` bytes of `src` at `src_offset`
/// into a fresh immutable staging slice, with the download's existing barrier
/// to the host stage. No submit here; the host reads the returned offset only
/// after endBatch's fence, through readDown.
pub fn batchDownload(cmds: *Commands, src: Buffer, src_offset: u64, len: u64) Error!u64 {
    if (!cmds.session) return error.Invalid;
    try operands(cmds.dev, src, cmds.batch_staging, src_offset, len);
    const at = try cmds.ledger.reserveDown(len);
    const cmd = try cmds.begin();
    barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_read | transfer_write);
    const region: c.VkBufferCopy = .{ .srcOffset = src_offset, .dstOffset = at, .size = len };
    cmds.dev.api.vkCmdCopyBuffer(cmd, src.handle, cmds.batch_staging.handle, 1, &region);
    barrier(cmds.dev, cmd, transfer, transfer_write, c.VK_PIPELINE_STAGE_HOST_BIT, c.VK_ACCESS_HOST_READ_BIT);
    return at;
}

/// `out.len` bytes of `src` from `offset` through `staging`, after every earlier dispatch or copy that wrote them.
/// Refused inside an open batch: its submit would be a no-op and the copy
/// below would read stale bytes — a batched token reads its recorded down
/// slices only after endBatch's fence, through readDown.
pub fn download(cmds: *Commands, src: Buffer, offset: u64, staging: Buffer, out: []u8, timeout_ns: u64) Error!void {
    if (cmds.session) return error.Invalid;
    try operands(cmds.dev, src, staging, offset, out.len);
    const host = try staging.bytes();
    var done: usize = 0;
    while (done < out.len) {
        const n = @min(host.len, out.len - done);
        const cmd = try cmds.begin();
        barrier(cmds.dev, cmd, gpu_stages, gpu_writes, transfer, transfer_read | transfer_write);
        const region: c.VkBufferCopy = .{ .srcOffset = offset + @as(u64, done), .dstOffset = 0, .size = n };
        cmds.dev.api.vkCmdCopyBuffer(cmd, src.handle, staging.handle, 1, &region);
        barrier(cmds.dev, cmd, transfer, transfer_write, c.VK_PIPELINE_STAGE_HOST_BIT, c.VK_ACCESS_HOST_READ_BIT);
        try cmds.submit(timeout_ns);
        try staging.invalidate();
        @memcpy(out[done..][0..n], host[0..n]);
        done += n;
    }
}

test "only a successful submit and a signalled fence let the caller free a submission's resources" {
    try submitted(c.VK_SUCCESS);
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_OUT_OF_HOST_MEMORY));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_OUT_OF_DEVICE_MEMORY));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_DEVICE_LOST));
    try std.testing.expectError(error.InFlight, submitted(c.VK_ERROR_UNKNOWN));
    try waited(c.VK_SUCCESS);
    try std.testing.expectError(error.Timeout, waited(c.VK_TIMEOUT));
    try std.testing.expectError(error.InFlight, waited(c.VK_ERROR_OUT_OF_HOST_MEMORY));
    try std.testing.expectError(error.InFlight, waited(c.VK_ERROR_DEVICE_LOST));
    try std.testing.expectError(error.InFlight, waited(c.VK_NOT_READY));
}

test "fill geometry refuses torn, empty and out-of-range fills" {
    try fillRange(4096, 0, 4096);
    try fillRange(4096, 12, 1024);
    try std.testing.expectError(error.Misaligned, fillRange(4096, 0, 0)); // empty
    try std.testing.expectError(error.Misaligned, fillRange(4096, 2, 1024));
    try std.testing.expectError(error.Misaligned, fillRange(4096, 0, 6));
    try std.testing.expectError(error.OutOfRange, fillRange(4096, 8, 4096));
    try std.testing.expectError(error.OutOfRange, fillRange(0, 0, 4));
}

test "staged transfers refuse foreign devices, wrong kinds, aliasing and ranges past the buffer" {
    // operand identities only: nothing here is dereferenced or reaches Vulkan
    const dev: *const Device = @ptrFromInt(0x1000);
    const other: *const Device = @ptrFromInt(0x2000);
    const host: [*]u8 = @ptrFromInt(0x3000);
    const buf: Buffer = .{ .dev = dev, .handle = @ptrFromInt(0x10), .memory = @ptrFromInt(0x20), .len = 4096, .kind = .device, .mapped = null, .coherent = false };
    const staging: Buffer = .{ .dev = dev, .handle = @ptrFromInt(0x30), .memory = @ptrFromInt(0x40), .len = 1027, .kind = .staging, .mapped = host, .coherent = true };
    try operands(dev, buf, staging, 0, 4096);
    try operands(dev, buf, staging, 13, 4083);
    try std.testing.expectError(error.OutOfRange, operands(dev, buf, staging, 13, 4084));
    try std.testing.expectError(error.OutOfRange, operands(dev, buf, staging, 4097, 0));
    try std.testing.expectError(error.Invalid, operands(other, buf, staging, 0, 4));
    var foreign = staging;
    foreign.dev = other;
    try std.testing.expectError(error.Invalid, operands(dev, buf, foreign, 0, 4));
    try std.testing.expectError(error.Invalid, operands(dev, staging, staging, 0, 4));
    try std.testing.expectError(error.Invalid, operands(dev, buf, buf, 0, 4));
    var same = staging;
    same.handle = buf.handle;
    try std.testing.expectError(error.Invalid, operands(dev, buf, same, 0, 4));
    var unmapped = staging;
    unmapped.mapped = null;
    try std.testing.expectError(error.Invalid, operands(dev, buf, unmapped, 0, 4));
    var empty = staging;
    empty.len = 0;
    try std.testing.expectError(error.Invalid, operands(dev, buf, empty, 0, 4));
}

test "batch staging slices are disjoint, bounded and never reused within a batch" {
    var ledger: BatchLedger = .{ .capacity = 256 };
    const a = try ledger.reserveUp(4);
    const b = try ledger.reserveUp(8);
    try std.testing.expectEqual(@as(u64, 0), a);
    try std.testing.expectEqual(@as(u64, 4), b); // strictly bump: no overlap, no offset-0 overwrite
    const d1 = try ledger.reserveDown(12);
    const d2 = try ledger.reserveDown(16);
    try std.testing.expectEqual(@as(u64, 12), d1);
    try std.testing.expectEqual(@as(u64, 24), d2); // down slices sit after every up slice
    try std.testing.expectEqual(@as(u64, 40), ledger.up);
    // overflow refuses before any host write could start
    try std.testing.expectError(error.OutOfRange, ledger.reserveUp(217));
    try std.testing.expectError(error.OutOfRange, ledger.reserveDown(217));
    // exactly filling the buffer is legal
    var tight: BatchLedger = .{ .capacity = 16 };
    _ = try tight.reserveUp(8);
    _ = try tight.reserveDown(8);
    try std.testing.expectError(error.OutOfRange, tight.reserveUp(1));
    // the down-slot bound is part of the batch contract
    var small: BatchLedger = .{ .capacity = 1 << 20 };
    for (0..max_batch_downs) |_| _ = try small.reserveDown(1);
    try std.testing.expectError(error.OutOfRange, small.reserveDown(1));
}

test "batch lifecycle: one begin per batch, nesting refused, every exit clears state" {
    var lc: BatchLifecycle = .{};
    try lc.start();
    try std.testing.expectError(error.Invalid, lc.start()); // nested begin refused
    try std.testing.expect(lc.open and !lc.begun); // start alone is not a begin
    lc.began(); // the physical reset+begin happened exactly once
    try std.testing.expect(lc.begun);
    try std.testing.expectEqual(@as(u32, 1), lc.begins);
    lc.finish();
    try std.testing.expect(!lc.open and !lc.begun);
    try lc.start(); // the next batch starts clean and begins again
    lc.began();
    try std.testing.expectEqual(@as(u32, 2), lc.begins);
    lc.finish();
}

test "a synchronous download refuses inside an open batch and readDown takes only recorded slices" {
    var cmds: Commands = .{ .dev = undefined, .pool = null, .cmd = null, .fence = null, .session = true };
    const src: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = 16, .kind = .device, .mapped = null, .coherent = false };
    const staging: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = 16, .kind = .staging, .mapped = null, .coherent = false };
    var out: [12]u8 = undefined;
    // the guard fires before any Vulkan or host work, so the fake handles are never touched
    try std.testing.expectError(error.Invalid, download(&cmds, src, 0, staging, &out, 0));
    // after endBatch (session false) a read is legal only for a recorded down slice
    cmds.session = false;
    try std.testing.expectError(error.Invalid, cmds.readDown(0, &out)); // nothing recorded
    cmds.ledger = .{ .capacity = 64 };
    const at = try cmds.ledger.reserveDown(12);
    try std.testing.expectEqual(@as(u64, 0), at);
    try std.testing.expectError(error.Invalid, cmds.readDown(4, &out)); // wrong offset
    try std.testing.expectError(error.Invalid, cmds.readDown(0, out[0..11])); // wrong length
    // the exact recorded slice passes the ledger gate (then fails on the
    // unmapped fake staging, proving the guard is what let it through)
    cmds.batch_staging = .{ .dev = undefined, .handle = null, .memory = null, .len = 64, .kind = .staging, .mapped = null, .coherent = false };
    try std.testing.expectError(error.NotMapped, cmds.readDown(0, &out));
}
