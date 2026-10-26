//! One compute pipeline over set-0 storage buffers and push constants: checked SPIR-V, limits, bindings and dispatch.
const std = @import("std");
const c = @import("vk");
const runtime = @import("runtime.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("memory.zig").Buffer;
const commands = @import("commands.zig");

const check = runtime.check;

pub const Error = runtime.Error || error{ BadSpirv, NoLocalSize, TooManyBindings, BadPushSize, BadWorkgroup, BadGroupCount, BadBindings, Aliased };

pub const max_bindings = 8;

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
    set: c.VkDescriptorSet = null,
    bound: bool = false,

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
        const size: c.VkDescriptorPoolSize = .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = layout.bindings };
        const pool_info: c.VkDescriptorPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &size,
        };
        try check(api.vkCreateDescriptorPool(h, &pool_info, null, &p.pool));
        errdefer api.vkDestroyDescriptorPool(h, p.pool, null);
        const alloc_info: c.VkDescriptorSetAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = p.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &p.set_layout,
        };
        try check(api.vkAllocateDescriptorSets(h, &alloc_info, &p.set));
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

    /// Updates checked device-local bindings; call only when no submission using this descriptor set is pending.
    pub fn bind(p: *Pipeline, bindings: []const Binding) Error!void {
        if (bindings.len != p.layout.bindings) return error.BadBindings;
        var infos: [max_bindings]c.VkDescriptorBufferInfo = undefined;
        var writes: [max_bindings]c.VkWriteDescriptorSet = undefined;
        for (bindings, 0..) |b, i| {
            if (b.buffer.kind != .device or b.buffer.dev != p.dev) return error.BadBindings;
            if (b.len % 4 != 0) return error.Misaligned; // 32-bit element arrays: a ragged range drops its last element
            infos[i] = try b.buffer.descriptor(b.offset, b.len);
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p.set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        try disjoint(bindings, p.layout.writes);
        p.dev.api.vkUpdateDescriptorSets(p.dev.handle, p.layout.bindings, &writes, 0, null);
        p.bound = true;
    }

    /// Records one dispatch, then a barrier making its writes visible to a later dispatch or transfer.
    pub fn record(p: *const Pipeline, cmd: c.VkCommandBuffer, push: []const u8, groups: [3]u32) Error!void {
        if (!p.bound) return error.BadBindings;
        if (push.len != p.layout.push_bytes) return error.BadPushSize;
        const most = p.dev.properties.limits.maxComputeWorkGroupCount;
        counts(most, groups) catch |e| {
            std.log.err("vulkan: dispatch {any} groups refused, device allows {any}", .{ groups, most });
            return e;
        };
        const api = &p.dev.api;
        api.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        api.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline_layout, 0, 1, &p.set, 0, null);
        if (push.len > 0) api.vkCmdPushConstants(cmd, p.pipeline_layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, p.layout.push_bytes, push.ptr);
        api.vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
        const reads = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT | c.VK_ACCESS_TRANSFER_READ_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;
        const stages = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        commands.barrier(p.dev, cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_ACCESS_SHADER_WRITE_BIT, stages, reads);
    }

    /// One dispatch in its own submission; wrap it in commands.orExit, which ends the process on error.Timeout.
    pub fn run(p: *const Pipeline, cmds: *commands.Commands, push: []const u8, groups: [3]u32, timeout_ns: u64) Error!void {
        try p.record(try cmds.begin(), push, groups);
        try cmds.submit(timeout_ns);
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
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 0, 1, 1 }));
    try std.testing.expectError(error.BadGroupCount, counts(.{ 65535, 65535, 65535 }, .{ 65536, 1, 1 }));
    try counts(.{ 65535, 65535, 65535 }, .{ 2, 1, 1 });
}
