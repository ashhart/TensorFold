// Greedy: max finite, +0/-0 ties to lowest index, any NaN/Inf refuses; argreduce = mlx-omarchy copies (max, one row)
const std = @import("std");
const c = @import("vk");
const commands = @import("commands.zig");
const dispatch = @import("dispatch.zig");
const memory = @import("memory.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const Device = @import("device.zig").Device;
const Commands = commands.Commands;
const Buffer = memory.Buffer;
const View = tensor.View;

pub const Error = dispatch.Error || error{NonFinite};

// finite_logits scans; argreduce_chunk and argreduce_combine reduce; chunk/combine use -DUSE_BF16; scan has no defines.
pub const Shaders = struct {
    finite_logits: []align(4) const u8,
    argreduce_chunk: []align(4) const u8,
    argreduce_combine: []align(4) const u8,
};

// Each workgroup covers 256 lanes x 16 prefetched values, matching the argreduce chunk kernel.
const block_elements: u64 = 4096;
/// The combine kernel folds at most 256 pairs in its single workgroup.
const max_chunks: u64 = 256;

// 12-byte readback stores verdict, first bad index, token; sentinels prevent unwritten indices passing as token 0.
pub const Result = extern struct {
    finite_flag: u32,
    first_bad: u32,
    index: u32,
};

comptime {
    if (@sizeOf(Result) != 12) @compileError("greedy readback drifted");
    if (@offsetOf(params.Params, "flags") + 4 != 60) @compileError("argreduce push prefix drifted");
}

/// The argreduce kernels' push block ends at `flags`: the standard 15-word prefix.
const greedy_push_bytes: u32 = 60;

/// finite_logits.comp reads only the row length.
const FiniteParams = extern struct { count: u32 = 0 };

// Pure plan: geometry only (Invalid empty, TooLarge huge); odd bf16 counts fail View word-padding at the caller
pub const Plan = struct { count: u32, chunk_len: u32, chunks: u32 };

pub fn greedyPlan(count: u64) Error!Plan {
    if (count == 0) return error.Invalid;
    const n = std.math.cast(u32, count) orelse return error.TooLarge;
    const chunks = std.math.cast(u32, (count + block_elements - 1) / block_elements) orelse return error.TooLarge;
    if (chunks > max_chunks) return error.TooLarge; // combine folds one row in a single workgroup
    return .{ .count = n, .chunk_len = @intCast(block_elements), .chunks = chunks };
}

// Decode returns NonFinite on dirty flags and VulkanFailed on missing/bad indices; otherwise it returns the token.
pub fn decode(count: u32, out: Result) Error!u32 {
    if (out.finite_flag != 0) return error.NonFinite;
    if (out.index >= count) return error.VulkanFailed;
    return out.index;
}

// The host widens bf16 exactly, rejects non-finites, and keeps the first max; ascending order resolves +0/-0 ties.
pub fn greedyRef(logits: []const f32) Error!u32 {
    if (logits.len == 0) return error.Invalid;
    var best = -std.math.inf(f32);
    var index: u32 = 0;
    for (logits, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonFinite;
        if (v > best) {
            best = v;
            index = std.math.cast(u32, i) orelse return error.TooLarge;
        }
    }
    return index;
}

pub const Greedy = struct {
    dev: *const Device,
    plan: Plan,
    finite_logits: dispatch.Pipeline,
    chunk: dispatch.Pipeline,
    combine: dispatch.Pipeline,
    /// Per-row (value, index) pairs the chunk kernel writes, one per chunk.
    partials: Buffer,
    /// The three result words: finite flag, first bad index, selected token.
    results: Buffer,

    // Builds pipelines and scratch; the device must support storage16 and int16 for bf16.
    pub fn init(dev: *const Device, shaders: Shaders, count: u64) Error!Greedy {
        if (!dev.features.storage16 or !dev.features.int16) return error.MissingFeature;
        const plan = try greedyPlan(count);
        var partials = try Buffer.init(dev, .device, @as(u64, plan.chunks) * 8);
        errdefer partials.deinit();
        var results = try Buffer.init(dev, .device, @sizeOf(Result));
        errdefer results.deinit();
        var g: Greedy = .{
            .dev = dev,
            .plan = plan,
            .finite_logits = undefined,
            .chunk = undefined,
            .combine = undefined,
            .partials = partials,
            .results = results,
        };
        g.finite_logits = try dispatch.Pipeline.init(dev, shaders.finite_logits, .{ .bindings = 2, .push_bytes = @sizeOf(FiniteParams), .writes = 1 << 1 });
        errdefer g.finite_logits.deinit();
        g.chunk = try dispatch.Pipeline.init(dev, shaders.argreduce_chunk, .{ .bindings = 2, .push_bytes = greedy_push_bytes, .writes = 1 << 1 });
        errdefer g.chunk.deinit();
        g.combine = try dispatch.Pipeline.init(dev, shaders.argreduce_combine, .{ .bindings = 2, .push_bytes = greedy_push_bytes, .writes = 1 << 1 });
        return g;
    }

    /// Destroys the pipelines and the scratch; call only after the last submission using them returned success.
    pub fn deinit(g: *Greedy) void {
        g.combine.deinit();
        g.chunk.deinit();
        g.finite_logits.deinit();
        g.partials.deinit();
        g.results.deinit();
    }

    // One draw resets, scans, reduces, combines, then fences readback; wrap in commands.orExit for Timeout/InFlight.
    pub fn select(g: *Greedy, cmds: *Commands, staging: Buffer, logits: View, timeout_ns: u64) Error!u32 {
        if (logits.kind != .bf16 or logits.elements != g.plan.count) return error.Invalid;
        try g.finite_logits.bind(&.{ logits.bind(), .{ .buffer = &g.results, .len = 8 } });
        try g.chunk.bind(&.{ logits.bind(), .{ .buffer = &g.partials, .len = g.partials.len } });
        try g.combine.bind(&.{ .{ .buffer = &g.partials, .len = g.partials.len }, .{ .buffer = &g.results, .len = g.results.len } });
        const cmd = try cmds.begin();
        // Each call resets flag=0 and bad/index=0xffffffff; a barrier orders fills before compute.
        try commands.recordFill(g.dev, cmd, g.results, 0, 4, 0);
        try commands.recordFill(g.dev, cmd, g.results, 4, 8, std.math.maxInt(u32));
        commands.barrier(g.dev, cmd, c.VK_PIPELINE_STAGE_TRANSFER_BIT, c.VK_ACCESS_TRANSFER_WRITE_BIT, fill_stages, fill_access);
        var fp: FiniteParams = .{ .count = g.plan.count };
        try g.finite_logits.record(cmd, std.mem.asBytes(&fp), .{ g.plan.chunks, 1, 1 });
        var p: params.Params = .{
            .operation = 1, // max
            .reduce_size = g.plan.count,
            .matrix_m = g.plan.chunk_len,
            .matrix_n = g.plan.chunks,
        };
        try g.chunk.record(cmd, std.mem.asBytes(&p)[0..greedy_push_bytes], .{ g.plan.chunks, 1, 1 });
        var o: params.Params = .{
            .operation = 1, // max
            .matrix_n = g.plan.chunks,
            .output_offset = 2, // Result.index: word 2, after flag and first_bad
        };
        try g.combine.record(cmd, std.mem.asBytes(&o)[0..greedy_push_bytes], .{ 1, 1, 1 });
        try cmds.submit(timeout_ns);
        var out: Result = undefined;
        try commands.download(cmds, g.results, 0, staging, std.mem.asBytes(&out), timeout_ns);
        return decode(g.plan.count, out) catch |e| {
            if (e == error.NonFinite) std.log.err("greedy: non-finite bf16 logit at index {d} of {d}; row refused", .{ out.first_bad, g.plan.count });
            return e;
        };
    }
};

const fill_stages: u32 = c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | c.VK_PIPELINE_STAGE_TRANSFER_BIT;
const fill_access: u32 = c.VK_ACCESS_SHADER_READ_BIT | c.VK_ACCESS_SHADER_WRITE_BIT |
    c.VK_ACCESS_TRANSFER_READ_BIT | c.VK_ACCESS_TRANSFER_WRITE_BIT;

/// f32 exactly as the kernels widen a bf16 halfword.
fn widen(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const testing = std.testing;

test "greedyPlan derives 4096-wide chunks and refuses empty, oversized and over-partitioned rows" {
    const plan = try greedyPlan(248320);
    try testing.expectEqual(@as(u32, 248320), plan.count);
    try testing.expectEqual(@as(u32, 4096), plan.chunk_len);
    try testing.expectEqual(@as(u32, 61), plan.chunks);
    try testing.expectEqual(@as(u32, 2), (try greedyPlan(4097)).chunks);
    try testing.expectError(error.Invalid, greedyPlan(0));
    try testing.expectError(error.TooLarge, greedyPlan(1 << 32));
    try testing.expectError(error.TooLarge, greedyPlan(max_chunks * block_elements + 1));
}

test "greedyRef applies the draw rule: max wins, exact ties keep the lowest index, any non-finite refuses" {
    try testing.expectEqual(@as(u32, 2), try greedyRef(&.{ -1.5, -1.5, -0.5, -2 })); // all negative
    try testing.expectEqual(@as(u32, 0), try greedyRef(&.{ 0.5, 0.5 }));
    try testing.expectEqual(@as(u32, 0), try greedyRef(&.{ 0.0, -0.0, -1.0 })); // +0/-0 compare equal: first of the tie
    try testing.expectEqual(@as(u32, 0), try greedyRef(&.{ -0.0, 0.0, -1.0 })); // whichever sign comes first
    try testing.expectEqual(@as(u32, 1), try greedyRef(&.{ widen(0x3f80), widen(0x4000), widen(0x4000), widen(0x4000) }));
    try testing.expectError(error.NonFinite, greedyRef(&.{ 1.0, std.math.nan(f32) }));
    try testing.expectError(error.NonFinite, greedyRef(&.{ 1.0, std.math.inf(f32) }));
    try testing.expectError(error.NonFinite, greedyRef(&.{ -std.math.inf(f32), -1.0 })); // losing -inf still refuses
    try testing.expectError(error.Invalid, greedyRef(&.{}));
}

test "bf16 logits ride word-padded views: an odd count has no view and is refused before any dispatch" {
    const buf: Buffer = .{ .dev = undefined, .handle = null, .memory = null, .len = 248322 * 2, .kind = .device, .mapped = null, .coherent = false };
    try testing.expectError(error.Misaligned, View.init(&buf, .bf16, 0, 1));
    // Both rows fit the fake buffer; the odd row fails View word-padding rather than exceeding the buffer.
    try testing.expectError(error.Misaligned, View.init(&buf, .bf16, 0, 248321));
    _ = try View.init(&buf, .bf16, 0, 248320);
}

test "decode refuses a dirty flag and an unwritten or out-of-row index and passes a real token" {
    const clean: Result = .{ .finite_flag = 0, .first_bad = std.math.maxInt(u32), .index = 262 };
    try testing.expectEqual(@as(u32, 262), try decode(248320, clean));
    try testing.expectError(error.NonFinite, decode(248320, .{ .finite_flag = 1, .first_bad = 7, .index = 0 }));
    try testing.expectError(error.VulkanFailed, decode(248320, .{ .finite_flag = 0, .first_bad = std.math.maxInt(u32), .index = std.math.maxInt(u32) }));
    try testing.expectError(error.VulkanFailed, decode(248320, .{ .finite_flag = 0, .first_bad = std.math.maxInt(u32), .index = 248320 }));
}
