//! One compute pipeline over set-0 storage buffers and push constants: checked SPIR-V, limits, bindings and dispatch.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("memory.zig").Buffer;
const commands = @import("commands.zig");
const hazard = @import("hazard.zig");

const check = runtime.check;

pub const Error = runtime.Error || error{ BadSpirv, NoLocalSize, TooManyBindings, BadPushSize, BadWorkgroup, BadGroupCount, BadBindings, Aliased, DescriptorSetsBusy };

pub const max_bindings = 10;

/// Dispatches recorded into one submission, so a descriptor set per dispatch; run submits at half of it.
pub const batch_max: u32 = 1024;
const batch_flush: u32 = batch_max / 2;

const Lease = struct { stream: u64 = 0, epoch: u64 = 0 };

/// The shader's set 0: storage buffers 0..bindings-1, its push bytes, and a mask of the bindings it writes.
pub const Layout = struct { bindings: u32, push_bytes: u32, writes: u32 };

/// A device-local byte range for one binding.
pub const Binding = struct { buffer: *const Buffer, offset: u64 = 0, len: u64 };

pub const Pipeline = struct {
    dev: *const Device,
    layout: Layout,
    /// The entry point's workgroup size, read from the module itself (localSize).
    local: [3]u32,
    set_layout: c.VkDescriptorSetLayout = null,
    pipeline_layout: c.VkPipelineLayout = null,
    pipeline: c.VkPipeline = null,
    pool: c.VkDescriptorPool = null,
    /// Pool of descriptor sets leased per dispatch and released only when the owning stream fences or discards.
    sets: [batch_max]c.VkDescriptorSet = @splat(null),
    leases: [batch_max]Lease = @splat(.{}),
    next_slot: usize = 0,
    bound: bool = false,
    staged: [max_bindings]c.VkDescriptorBufferInfo = undefined,
    /// The byte range of each binding of the current bind, for the hazard check of the next run.
    ranges: [max_bindings]hazard.Range = undefined,

    /// Checks the module and the device's limits before creating anything; a failure destroys what was made.
    pub fn init(dev: *const Device, spirv: []align(4) const u8, layout: Layout) Error!Pipeline {
        const local = localSize(spirv) catch |e| {
            const hint = if (e == error.MissingFeature) " (LocalSizeId needs maintenance4: compile for SPIR-V 1.5)" else "";
            std.log.err("vulkan: SPIR-V module of {d} bytes refused: {s}{s}", .{ spirv.len, @errorName(e), hint });
            return e;
        };
        const l = dev.properties.limits;
        fits(l, layout, local) catch |e| {
            const say = "vulkan: {s}: {d} storage buffers, {d} push bytes, workgroup {any}; device: {d}/{d}/{d} " ++
                "buffers, {d} push bytes, workgroup {any} of {d}";
            const asked = .{ @errorName(e), layout.bindings, layout.push_bytes, local };
            const has = .{ l.maxPerStageDescriptorStorageBuffers, l.maxDescriptorSetStorageBuffers, l.maxPerStageResources };
            const room = .{ l.maxPushConstantsSize, l.maxComputeWorkGroupSize, l.maxComputeWorkGroupInvocations };
            std.log.err(say, asked ++ has ++ room);
            return e;
        };
        const api = &dev.api;
        const h = dev.handle;
        const module_info: c.VkShaderModuleCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.len,
            .pCode = @ptrCast(spirv.ptr),
        };
        var module: c.VkShaderModule = null;
        try check(api.vkCreateShaderModule(h, &module_info, null, &module));
        defer api.vkDestroyShaderModule(h, module, null); // a created pipeline no longer needs its module
        var slots: [max_bindings]c.VkDescriptorSetLayoutBinding = undefined;
        for (slots[0..layout.bindings], 0..) |*s, i| s.* = .{
            .binding = @intCast(i),
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
        };
        var p: Pipeline = .{ .dev = dev, .layout = layout, .local = local };
        const set_info: c.VkDescriptorSetLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = layout.bindings,
            .pBindings = &slots,
        };
        try check(api.vkCreateDescriptorSetLayout(h, &set_info, null, &p.set_layout));
        errdefer api.vkDestroyDescriptorSetLayout(h, p.set_layout, null);
        const range: c.VkPushConstantRange = .{ .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT, .size = layout.push_bytes };
        const layout_info: c.VkPipelineLayoutCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &p.set_layout,
            .pushConstantRangeCount = @intFromBool(layout.push_bytes > 0),
            .pPushConstantRanges = &range,
        };
        try check(api.vkCreatePipelineLayout(h, &layout_info, null, &p.pipeline_layout));
        errdefer api.vkDestroyPipelineLayout(h, p.pipeline_layout, null);
        const pipe_info: c.VkComputePipelineCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
            .layout = p.pipeline_layout,
            .basePipelineIndex = -1,
        };
        check(api.vkCreateComputePipelines(h, null, 1, &pipe_info, null, &p.pipeline)) catch |e| {
            if (p.pipeline != null) api.vkDestroyPipeline(h, p.pipeline, null); // a driver may leave a partial handle
            return e;
        };
        errdefer api.vkDestroyPipeline(h, p.pipeline, null);
        const size: c.VkDescriptorPoolSize = .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = layout.bindings * batch_max };
        const pool_info: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = batch_max,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        try check(api.vkCreateDescriptorPool(h, &pool_info, null, &p.pool));
        errdefer api.vkDestroyDescriptorPool(h, p.pool, null);
        const set_layouts: [batch_max]c.VkDescriptorSetLayout = @splat(p.set_layout);
        const alloc_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = p.pool,
            .descriptorSetCount = batch_max,
            .pSetLayouts = &set_layouts,
        };
        try check(api.vkAllocateDescriptorSets(h, &alloc_info, &p.sets));
        return p;
    }

    /// Call only after the last submission that used this pipeline returned success.
    pub fn deinit(p: *Pipeline) void {
        const api = &p.dev.api;
        const h = p.dev.handle;
        api.vkDestroyDescriptorPool(h, p.pool, null);
        api.vkDestroyPipeline(h, p.pipeline, null);
        api.vkDestroyPipelineLayout(h, p.pipeline_layout, null);
        api.vkDestroyDescriptorSetLayout(h, p.set_layout, null);
    }

    /// Validates and stages device-local bindings; actual descriptor sets are written only at fence-checked checkout.
    pub fn bind(p: *Pipeline, bindings: []const Binding) Error!void {
        if (bindings.len != p.layout.bindings) return error.BadBindings;
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        for (bindings, 0..) |b, i| {
            if (b.buffer.kind != .device or b.buffer.dev != p.dev) return error.BadBindings;
            if (b.len % 4 != 0) return error.Misaligned; // 32-bit element arrays: a ragged range drops its last element
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
        }
        try disjoint(bindings, p.layout.writes);
        for (bindings, 0..) |b, i| {
            p.staged[i] = infos[i];
            p.ranges[i] = .{ .buffer = @intFromPtr(b.buffer.handle), .start = b.offset, .end = b.offset + b.len };
        }
        p.bound = true;
    }

    fn findFreeSlot(p: *const Pipeline) ?usize {
        for (0..batch_max) |off| {
            const idx = (p.next_slot + off) % batch_max;
            const l = p.leases[idx];
            if (commands.isLeaseFree(l.stream, l.epoch)) return idx;
        }
        return null;
    }

    fn checkoutSet(p: *Pipeline, cmds: *commands.Commands, timeout_ns: ?u64) Error!struct { set: c.VkDescriptorSet, slot: usize } {
        const sid = try cmds.streamId();
        var slot = p.findFreeSlot();
        if (slot == null) {
            if (timeout_ns) |t| if (cmds.recording) {
                var owns_any = false;
                for (p.leases) |l| if (l.stream == sid) {
                    owns_any = true;
                    break;
                };
                if (owns_any) {
                    try cmds.submit(t);
                    slot = p.findFreeSlot();
                }
            };
        }
        const idx = slot orelse return error.DescriptorSetsBusy;
        var writes: [max_bindings]c.VkWriteDescriptorSet = undefined;
        for (0..p.layout.bindings) |i| {
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p.sets[idx],
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &p.staged[i],
            };
        }
        p.dev.api.vkUpdateDescriptorSets(p.dev.handle, p.layout.bindings, &writes, 0, null);
        p.leases[idx] = .{ .stream = sid, .epoch = cmds.epoch };
        p.next_slot = (idx + 1) % batch_max;
        return .{ .set = p.sets[idx], .slot = idx };
    }

    fn validateDispatch(p: *const Pipeline, push: []const u8, groups: [3]u32) Error!void {
        if (!p.bound) return error.BadBindings;
        if (push.len != p.layout.push_bytes) return error.BadPushSize;
        const most = p.dev.properties.limits.maxComputeWorkGroupCount;
        counts(most, groups) catch |e| {
            std.log.err("vulkan: dispatch {any} groups refused, device allows {any}", .{ groups, most });
            return e;
        };
    }

    /// Records one dispatch and a barrier after it into an already-open command buffer on `cmds`.
    pub fn record(p: *Pipeline, cmds: *commands.Commands, push: []const u8, groups: [3]u32) Error!void {
        try p.validateDispatch(push, groups);
        if (cmds.poison) |err| return err;
        if (!cmds.recording) return error.CommandBufferInvalid;
        const leased = try p.checkoutSet(cmds, null);
        p.dispatchSet(cmds.cmd, leased.set, push, groups);
        commands.dispatchBarrier(p.dev, cmds.cmd);
    }

    fn dispatchSet(p: *const Pipeline, cmd: c.VkCommandBuffer, set: c.VkDescriptorSet, push: []const u8, groups: [3]u32) void {
        const api = &p.dev.api;
        api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &set, 0, null);
        if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
        api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
    }

    /// Splits bound ranges into reads and writes; an unbound pipeline's ranges are undefined, so refuse first.
    fn hazardSplit(p: *const Pipeline, reads: *[max_bindings]hazard.Range, writes: *[max_bindings]hazard.Range) Error!struct { reads: []hazard.Range, writes: []hazard.Range } {
        if (!p.bound) return error.BadBindings;
        var n_reads: usize = 0;
        var n_writes: usize = 0;
        for (p.ranges[0..p.layout.bindings], 0..) |r, i| {
            if (p.layout.writes >> @intCast(i) & 1 != 0) {
                writes[n_writes] = r;
                n_writes += 1;
            } else {
                reads[n_reads] = r;
                n_reads += 1;
            }
        }
        return .{ .reads = reads[0..n_reads], .writes = writes[0..n_writes] };
    }

    /// Records one dispatch, after a barrier only on a conflict; submits at half a batch or on descriptor exhaustion.
    pub fn run(p: *Pipeline, cmds: *commands.Commands, push: []const u8, groups: [3]u32, timeout_ns: u64) Error!void {
        try p.validateDispatch(push, groups);
        var reads: [max_bindings]hazard.Range = undefined;
        var writes: [max_bindings]hazard.Range = undefined;
        const split = try p.hazardSplit(&reads, &writes);
        const leased = try p.checkoutSet(cmds, timeout_ns);
        const cmd = cmds.begin() catch |e| {
            p.leases[leased.slot] = .{};
            return e;
        };
        if (cmds.hazards.conflicts(split.reads, split.writes)) {
            commands.dispatchBarrier(p.dev, cmd);
            cmds.hazards.clear();
        }
        p.dispatchSet(cmd, leased.set, push, groups);
        cmds.hazards.add(split.reads, split.writes);
        cmds.pending += 1;
        if (cmds.pending >= batch_flush) try cmds.submit(timeout_ns);
    }
};

const op = struct {
    const entry_point = 15;
    const execution_mode = 16;
    const constant = 43;
    const constant_composite = 44;
    const spec_constant = 50;
    const spec_constant_composite = 51;
    const decorate = 71;
    const execution_mode_id = 331;
};

/// Validates SPIR-V and reads main's workgroup size; refuses LocalSizeId because maintenance4 is not enabled.
pub fn localSize(code: []align(4) const u8) Error![3]u32 {
    if (code.len < 20 or code.len % 4 != 0) return error.BadSpirv;
    const w = std.mem.bytesAsSlice(u32, code);
    const minor = w[1] >> 8 & 0xff;
    if (w[0] != 0x07230203 or w[1] >> 16 != 1 or minor > 6 or w[3] == 0 or w[4] != 0) return error.BadSpirv;
    var entry: ?u32 = null;
    var literal: ?[3]u32 = null;
    var by_id = false;
    var builtin: ?u32 = null;
    var composite: ?[3]u32 = null;
    var i: usize = 5;
    while (i < w.len) : (i += w[i] >> 16) {
        const in = try instruction(w, i);
        const code_op = in[0] & 0xffff;
        const ours = in.len == 6 and entry != null and in[1] == entry.?;
        switch (code_op) {
            op.entry_point => if (in.len >= 5 and in[1] == 5 and std.mem.startsWith(u8, std.mem.sliceAsBytes(in[3..]), "main\x00")) {
                entry = in[2];
            },
            op.execution_mode => if (ours and in[2] == 17) {
                literal = in[3..6].*;
            },
            op.execution_mode_id => if (ours and in[2] == 38) {
                by_id = true;
            },
            op.decorate => if (in.len == 4 and in[2] == 11 and in[3] == 25) {
                builtin = in[1];
            },
            op.constant_composite, op.spec_constant_composite => if (in.len == 6 and builtin != null and in[2] == builtin.?) {
                composite = in[3..6].*;
            },
            else => {},
        }
    }
    if (entry == null) return error.NoLocalSize;
    if (by_id) return error.MissingFeature;
    if (builtin == null) return literal orelse error.NoLocalSize;
    const ids = composite orelse return error.NoLocalSize;
    var size: [3]?u32 = .{ null, null, null };
    i = 5;
    while (i < w.len) : (i += w[i] >> 16) {
        const in = try instruction(w, i);
        const scalar = in.len == 4 and (in[0] & 0xffff == op.constant or in[0] & 0xffff == op.spec_constant);
        if (scalar) for (ids, &size) |id, *s| if (in[2] == id) {
            s.* = in[3];
        };
    }
    var out: [3]u32 = undefined;
    for (size, &out) |s, *o| o.* = s orelse return error.NoLocalSize;
    return out;
}

fn instruction(w: []const u32, i: usize) Error![]const u32 {
    const n = w[i] >> 16;
    if (n == 0 or n > w.len - i) return error.BadSpirv;
    return w[i..][0..n];
}

fn fits(l: c.VkPhysicalDeviceLimits, layout: Layout, local: [3]u32) Error!void {
    const buffers = @min(l.maxPerStageDescriptorStorageBuffers, l.maxDescriptorSetStorageBuffers, l.maxPerStageResources);
    if (layout.bindings == 0 or layout.bindings > max_bindings or layout.bindings > buffers) return error.TooManyBindings;
    if (layout.writes >> @intCast(layout.bindings) != 0) return error.BadBindings;
    if (layout.push_bytes % 4 != 0 or layout.push_bytes > l.maxPushConstantsSize) return error.BadPushSize;
    var threads: u64 = 1;
    for (local, l.maxComputeWorkGroupSize) |n, most| {
        if (n == 0 or n > most) return error.BadWorkgroup;
        threads *= n; // threads was at most the u32 invocation limit, so this product fits u64
        if (threads > l.maxComputeWorkGroupInvocations) return error.BadWorkgroup;
    }
}

fn counts(most: [3]u32, groups: [3]u32) Error!void {
    for (groups, most) |g, m| if (g == 0 or g > m) return error.BadGroupCount;
}

/// A written range may not overlap any other binding's range in the same buffer.
fn disjoint(bindings: []const Binding, writes: u32) Error!void {
    for (bindings, 0..) |o, i| if (writes >> @intCast(i) & 1 != 0) for (bindings, 0..) |b, j| {
        if (i != j and o.buffer.handle == b.buffer.handle and o.offset < b.offset + b.len and b.offset < o.offset + o.len) return error.Aliased;
    };
}

const header = [_]u32{ 0x07230203, 0x00010600, 0, 30, 0 };
const main_entry = [_]u32{ 5 << 16 | 15, 5, 7, 0x6e69616d, 0 };
const local_256 = [_]u32{ 6 << 16 | 16, 7, 17, 256, 1, 1 };
const entry_words = header ++ main_entry ++ local_256;
const constants = [_]u32{ 4 << 16 | 43, 2, 10, 128, 4 << 16 | 43, 2, 11, 1 };

test "localSize reads the compute entry's size in each SPIR-V form and refuses malformed modules" {
    const bytes = std.mem.sliceAsBytes;
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, try localSize(bytes(&entry_words)));
    const by_id = header ++ main_entry ++ [_]u32{ 6 << 16 | 331, 7, 38, 10, 11, 11 } ++ constants;
    try std.testing.expectError(error.MissingFeature, localSize(bytes(&by_id)));
    const decorated = entry_words ++ [_]u32{ 4 << 16 | 71, 20, 11, 25 } ++ constants ++ [_]u32{ 6 << 16 | 44, 3, 20, 10, 11, 11 };
    try std.testing.expectEqual([3]u32{ 128, 1, 1 }, try localSize(bytes(&decorated)));
    try std.testing.expectError(error.NoLocalSize, localSize(bytes(decorated[0 .. decorated.len - 6])));
    try std.testing.expectError(error.BadSpirv, localSize(bytes(entry_words[0..15])));
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&(entry_words ++ [_]u32{0}))));
    var bad = entry_words;
    bad[0] = 0x03022307;
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&bad)));
    bad = entry_words;
    bad[11] = 8; // LocalSize of another function
    try std.testing.expectError(error.NoLocalSize, localSize(bytes(&bad)));
    bad = entry_words;
    bad[1] = 0x00010700;
    try std.testing.expectError(error.BadSpirv, localSize(bytes(&bad)));
}

test "device limits refuse bindings, push bytes, workgroups and group counts past them" {
    var l = std.mem.zeroes(c.VkPhysicalDeviceLimits);
    l.maxPerStageDescriptorStorageBuffers = 4;
    l.maxDescriptorSetStorageBuffers = 8;
    l.maxPerStageResources = 8;
    l.maxPushConstantsSize = 128;
    l.maxComputeWorkGroupSize = .{ 256, 1, 1 };
    l.maxComputeWorkGroupInvocations = 256;
    const qmm: Layout = .{ .bindings = 5, .push_bytes = 120, .writes = 1 << 4 };
    try std.testing.expectError(error.TooManyBindings, fits(l, qmm, .{ 256, 1, 1 }));
    l.maxPerStageDescriptorStorageBuffers = 8;
    try fits(l, qmm, .{ 256, 1, 1 });
    try std.testing.expectError(error.BadPushSize, fits(l, .{ .bindings = 5, .push_bytes = 132, .writes = 1 << 4 }, .{ 256, 1, 1 }));
    try std.testing.expectError(error.BadBindings, fits(l, .{ .bindings = 5, .push_bytes = 120, .writes = 1 << 5 }, .{ 256, 1, 1 }));
    try std.testing.expectError(error.BadWorkgroup, fits(l, qmm, .{ 512, 1, 1 }));
    l.maxComputeWorkGroupInvocations = 128;
    try std.testing.expectError(error.BadWorkgroup, fits(l, qmm, .{ 256, 1, 1 }));
    // restore the limit the negative case reduced, before the positive cases
    l.maxComputeWorkGroupInvocations = 256;
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 0, 1, 1 }));
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 65536, 1, 1 }));
    try counts(.{ 65535, 65535, 65535 }, .{ 2, 1, 1 });
    // The 10-binding GDN prefill/decode closures fit; 11 do not; fewer offered still refuses (real capacity).
    l.maxPerStageDescriptorStorageBuffers = 10;
    l.maxDescriptorSetStorageBuffers = 10;
    l.maxPerStageResources = 10;
    try fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 });
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings + 1, .push_bytes = 120, .writes = 1 << 10 }, .{ 256, 1, 1 }));
    l.maxPerStageDescriptorStorageBuffers = 8;
    try std.testing.expectError(error.TooManyBindings, fits(l, .{ .bindings = max_bindings, .push_bytes = 120, .writes = 1 << 9 }, .{ 256, 1, 1 }));
}

test "hazardSplit refuses an unbound pipeline before reading its ranges" {
    var p: Pipeline = .{
        .dev = @ptrFromInt(0x5000),
        .layout = .{ .bindings = 3, .push_bytes = 0, .writes = 0b010 },
        .local = .{ 32, 1, 1 },
    };
    var reads: [max_bindings]hazard.Range = undefined;
    var writes: [max_bindings]hazard.Range = undefined;
    try std.testing.expectError(error.BadBindings, p.hazardSplit(&reads, &writes));
    p.bound = true;
    p.ranges[0] = .{ .buffer = 0x10, .start = 0, .end = 64 };
    p.ranges[1] = .{ .buffer = 0x20, .start = 0, .end = 128 };
    p.ranges[2] = .{ .buffer = 0x30, .start = 0, .end = 256 };
    const split = try p.hazardSplit(&reads, &writes);
    try std.testing.expectEqual(2, split.reads.len);
    try std.testing.expectEqual(1, split.writes.len);
    try std.testing.expectEqual(0x20, split.writes[0].buffer);
}

test "fence-tied descriptor checkout handles >1024 dispatches, interleaved streams and poisoned submits" {
    const Mock = struct {
        var updates: usize = 0;
        var submits: usize = 0;
        var wait_result: c.VkResult = c.VK_SUCCESS;
        fn okResetCmd(_: c.VkCommandBuffer, _: c.VkCommandBufferResetFlags) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okBegin(_: c.VkCommandBuffer, _: [*c]const c.VkCommandBufferBeginInfo) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okEnd(_: c.VkCommandBuffer) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn okSubmit(_: c.VkQueue, _: c_uint, _: [*c]const c.VkSubmitInfo, _: c.VkFence) callconv(.c) c.VkResult {
            submits += 1;
            return c.VK_SUCCESS;
        }
        fn okWait(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence, _: c.VkBool32, _: c_ulonglong) callconv(.c) c.VkResult {
            return wait_result;
        }
        fn okResetFence(_: c.VkDevice, _: c_uint, _: [*c]const c.VkFence) callconv(.c) c.VkResult {
            return c.VK_SUCCESS;
        }
        fn noopUpdate(_: c.VkDevice, _: c_uint, _: [*c]const c.VkWriteDescriptorSet, _: c_uint, _: [*c]const c.VkCopyDescriptorSet) callconv(.c) void {
            updates += 1;
        }
        fn noopBindPipe(_: c.VkCommandBuffer, _: c.VkPipelineBindPoint, _: c.VkPipeline) callconv(.c) void {}
        fn noopBindDesc(_: c.VkCommandBuffer, _: c.VkPipelineBindPoint, _: c.VkPipelineLayout, _: c_uint, _: c_uint, _: [*c]const c.VkDescriptorSet, _: c_uint, _: [*c]const c_uint) callconv(.c) void {}
        fn noopDispatch(_: c.VkCommandBuffer, _: c_uint, _: c_uint, _: c_uint) callconv(.c) void {}
        fn noopBarrier(_: c.VkCommandBuffer, _: c.VkPipelineStageFlags, _: c.VkPipelineStageFlags, _: c.VkDependencyFlags, _: c_uint, _: [*c]const c.VkMemoryBarrier, _: c_uint, _: [*c]const c.VkBufferMemoryBarrier, _: c_uint, _: [*c]const c.VkImageMemoryBarrier) callconv(.c) void {}
    };
    var dev: Device = undefined;
    dev.handle = @ptrFromInt(0x1000);
    dev.queue = @ptrFromInt(0x2000);
    dev.properties = std.mem.zeroes(c.VkPhysicalDeviceProperties);
    dev.properties.limits.maxComputeWorkGroupCount = .{ 65535, 65535, 65535 };
    dev.properties.limits.minStorageBufferOffsetAlignment = 16;
    dev.properties.limits.maxStorageBufferRange = 1 << 20;
    dev.api.vkResetCommandBuffer = Mock.okResetCmd;
    dev.api.vkBeginCommandBuffer = Mock.okBegin;
    dev.api.vkEndCommandBuffer = Mock.okEnd;
    dev.api.vkQueueSubmit = Mock.okSubmit;
    dev.api.vkWaitForFences = Mock.okWait;
    dev.api.vkResetFences = Mock.okResetFence;
    dev.api.vkUpdateDescriptorSets = Mock.noopUpdate;
    dev.api.vkCmdBindPipeline = Mock.noopBindPipe;
    dev.api.vkCmdBindDescriptorSets = Mock.noopBindDesc;
    dev.api.vkCmdDispatch = Mock.noopDispatch;
    dev.api.vkCmdPipelineBarrier = Mock.noopBarrier;
    var p: Pipeline = .{ .dev = &dev, .layout = .{ .bindings = 2, .push_bytes = 0, .writes = 0b10 }, .local = .{ 64, 1, 1 } };
    for (&p.sets, 0..) |*s, idx| s.* = @ptrFromInt(0x10000 + idx * 8);
    const in_buf: Buffer = .{ .dev = &dev, .handle = @ptrFromInt(0x10), .memory = null, .len = 256, .kind = .device, .mapped = null, .coherent = false };
    const out_buf: Buffer = .{ .dev = &dev, .handle = @ptrFromInt(0x20), .memory = null, .len = 256, .kind = .device, .mapped = null, .coherent = false };
    // Multiple binds before dispatch never mutate live descriptor sets.
    const upd0 = Mock.updates;
    try p.bind(&.{ .{ .buffer = &in_buf, .len = 64 }, .{ .buffer = &out_buf, .len = 64 } });
    try p.bind(&.{ .{ .buffer = &in_buf, .len = 64 }, .{ .buffer = &out_buf, .len = 64 } });
    try std.testing.expectEqual(upd0, Mock.updates);
    // >1024 dispatches wrap safely across fenced submits.
    var ca: commands.Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x3000), .fence = @ptrFromInt(0x4000) };
    var cb: commands.Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x3100), .fence = @ptrFromInt(0x4100) };
    var cc: commands.Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x3200), .fence = @ptrFromInt(0x4200) };
    defer {
        ca.discard();
        cb.discard();
        cc.discard();
    }
    for (0..batch_max * 4 + 32) |_| try p.run(&ca, &.{}, .{ 1, 1, 1 }, 1000);
    try ca.submit(1000);
    try std.testing.expect(Mock.submits >= 8);
    // Interleaved streams: ca/cb hold batch_flush-1 sets each; cc uses 2 and flushes itself without stealing.
    for (0..batch_flush - 1) |_| try p.run(&ca, &.{}, .{ 1, 1, 1 }, 1000);
    for (0..batch_flush - 1) |_| try p.run(&cb, &.{}, .{ 1, 1, 1 }, 1000);
    try p.run(&cc, &.{}, .{ 1, 1, 1 }, 1000);
    try p.run(&cc, &.{}, .{ 1, 1, 1 }, 1000);
    const sub_before = Mock.submits;
    try p.run(&cc, &.{}, .{ 1, 1, 1 }, 1000);
    try std.testing.expectEqual(sub_before + 1, Mock.submits);
    // When other streams hold all batch_max sets, a stream with no sets of its own gets DescriptorSetsBusy.
    var cd: commands.Commands = .{ .dev = &dev, .pool = undefined, .cmd = @ptrFromInt(0x3300), .fence = @ptrFromInt(0x4300) };
    try p.run(&cc, &.{}, .{ 1, 1, 1 }, 1000);
    try std.testing.expectError(error.DescriptorSetsBusy, p.run(&cd, &.{}, .{ 1, 1, 1 }, 1000));
    try ca.submit(1000);
    try cb.submit(1000);
    try cc.submit(1000);
    // A timed-out submit keeps its leased sets locked (not reused while GPU completion is unknown).
    try p.run(&cd, &.{}, .{ 1, 1, 1 }, 1000);
    const leased_sid = try cd.streamId();
    const leased_ep = cd.epoch;
    Mock.wait_result = c.VK_TIMEOUT;
    defer Mock.wait_result = c.VK_SUCCESS;
    try std.testing.expectError(error.Timeout, cd.submit(1000));
    try std.testing.expect(!commands.isLeaseFree(leased_sid, leased_ep));
}
