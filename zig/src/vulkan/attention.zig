// Qwen3.5-2B uses half-split RoPE: head_dim 256, rotary_dim 64, theta 1e7; 8 query heads, 2 KV heads; scale 1/16.
// fast_rope: -DUSE_BF16 -DASH_QWEN35 (no mid bf16 rounding); sdpa: -DBF16_IO -DSDPA_DIM=256 -DSDPA_ROWS; gate: none
// zig/src/families/qwen3_5/forward.zig attention(); zig/kernels/metal/qwen3_5/{layout,attn_partial,attn_merge}.metal.
const std = @import("std");
const dispatch = @import("dispatch.zig");
const memory = @import("memory.zig");
const ops_mod = @import("ops.zig");
const params = @import("stage1_params");
const tensor = @import("tensor.zig");
const commands = @import("commands.zig");
const Device = @import("device.zig").Device;
const Commands = commands.Commands;
const Buffer = memory.Buffer;
// dispatch.Error adds ContextTooLong; long prefixes refuse, geometry never falls through, and there is no CPU fallback.
pub const Error = dispatch.Error || error{ContextTooLong};
const View = tensor.View;
const Ops = ops_mod.Ops;

pub const query_heads: u64 = 8;
pub const kv_heads: u64 = 2;
pub const head_dim: u64 = 256;
// Metal rotates i and i+32 for i < 32 (rotary_dim 64) and passes through d >= 64.
pub const rotary_pairs: u64 = 32;
pub const theta: f32 = 1e7;
/// head_dim^-1/2 as the oracle passes it: e.setValue(1.0 / 16.0, 3).
pub const scale: f32 = 1.0 / 16.0;
// bf16 stages kStreamKeys f32 slots; long prefixes refuse, not truncate; p1/p2 is not wired.
pub const max_visible_keys: u64 = 7168;
/// s_qrow (256) + s_stream (7168) + s_tree (256) f32 words.
const sdpa_shared_bytes: u64 = 30720;

// The compiled shader images Attention runs; every field is a real SPIR-V image of the named variant.
pub const Shaders = struct {
    rope_bf16: []align(4) const u8,
    sdpa_rows_bf16: []align(4) const u8,
    gate_bf16: []align(4) const u8,
};

/// attention_gate.comp's own one-word push block.
const GateParams = extern struct { words: u32 = 0 };
const gate_push_bytes: u32 = @sizeOf(GateParams);

const gate_row_elems: u64 = query_heads * head_dim; // 2048
const gate_row_words: u64 = gate_row_elems / 2;

comptime {
    if (gate_push_bytes != 4) @compileError("attention_gate push block must stay one word");
}

pub const Attention = struct {
    dev: *const Device,
    /// Mutable: cacheKV rebinds the shared Ops copy_u16 pipeline (bind needs *Pipeline).
    ops: *Ops,
    rope_bf16: dispatch.Pipeline,
    sdpa_bf16: dispatch.Pipeline,
    gate_bf16: dispatch.Pipeline,
    // The i32 base position is read by each rope dispatch; setBasePosition writes it between operations.
    offsets: Buffer,

    // Builds three pipelines; checks bf16 features, shared memory, and Pipeline.init’s 1024-thread workgroup limit.
    pub fn init(dev: *const Device, shaders: Shaders, ops: *Ops) Error!Attention {
        if (!dev.features.storage16 or !dev.features.int16) return error.MissingFeature;
        if (dev.properties.limits.maxComputeSharedMemorySize < sdpa_shared_bytes) return error.MissingFeature;
        var a: Attention = .{
            .dev = dev,
            .ops = ops,
            .rope_bf16 = undefined,
            .sdpa_bf16 = undefined,
            .gate_bf16 = undefined,
            .offsets = try Buffer.init(dev, .device, 4),
        };
        errdefer a.offsets.deinit();
        // fast_rope reads input, output and the offsets slot; output is binding 1.
        a.rope_bf16 = try dispatch.Pipeline.init(dev, shaders.rope_bf16, .{ .bindings = 3, .push_bytes = params.push_bytes, .writes = 1 << 1 });
        errdefer a.rope_bf16.deinit();
        a.sdpa_bf16 = try dispatch.Pipeline.init(dev, shaders.sdpa_rows_bf16, .{ .bindings = 4, .push_bytes = params.push_bytes, .writes = 1 << 3 });
        errdefer a.sdpa_bf16.deinit();
        a.gate_bf16 = try dispatch.Pipeline.init(dev, shaders.gate_bf16, .{ .bindings = 3, .push_bytes = gate_push_bytes, .writes = 1 << 2 });
        return a;
    }

    // Deinit after successful completion only; Timeout/InFlight may leave GPU-owned resources, so use commands.orExit.
    pub fn deinit(a: *Attention) void {
        a.rope_bf16.deinit();
        a.sdpa_bf16.deinit();
        a.gate_bf16.deinit();
        a.offsets.deinit();
    }

    // Uploads the segment base (cache length); rope reads it, and queue order requires upload to finish first.
    pub fn setBasePosition(a: *Attention, cmds: *Commands, staging: Buffer, position: i64, timeout_ns: u64) Error!void {
        if (position < 0 or position > std.math.maxInt(i32)) return error.Invalid;
        const bytes = std.mem.toBytes(@as(i32, @intCast(position)));
        try commands.upload(cmds, a.offsets, 0, staging, &bytes, timeout_ns);
    }

    // Rotates normalized q (rows,8,256) into a fresh bf16 window at position base+row.
    pub fn ropeQueries(a: *Attention, cmds: *Commands, out: View, q: View, rows: u64, timeout_ns: u64) Error!void {
        const plan = try ropePlan(out.elements, q.elements, rows, query_heads);
        if (out.overlaps(q)) return error.Aliased;
        try a.rope_bf16.bind(&.{ q.bind(), out.bind(), a.offsetBinding() });
        try a.ropeRun(cmds, plan, queryStrides(), 0, timeout_ns);
    }

    // K rotates into cache; V uses bit-exact Ops.copy_u16; barriers order reads. Per-op storage roundings match oracle.
    pub fn cacheKV(a: *Attention, cmds: *Commands, cache_keys: View, cache_values: View, knorm: View, v: View, rows: u64, base: u64, capacity: u64, timeout_ns: u64) Error!void {
        if (cache_keys.kind != .bf16 or cache_values.kind != .bf16) return error.Invalid;
        const cache_elems = kv_heads *| capacity *| head_dim;
        if (cache_keys.elements != cache_elems or cache_values.elements != cache_elems) return error.Invalid;
        const end = std.math.add(u64, base, rows) catch return error.TooLarge;
        if (end > capacity) return error.OutOfRange;
        if (cache_keys.overlaps(knorm) or cache_values.overlaps(v)) return error.Aliased;
        const plan = try ropePlan(knorm.elements, knorm.elements, rows, kv_heads);
        // K: passthrough tail copies d >= 64; rotation covers d < 64.
        try a.rope_bf16.bind(&.{ knorm.bind(), cache_keys.bind(), a.offsetBinding() });
        try a.ropeRun(cmds, plan, .{
            .in_head = head_dim,
            .in_time = kv_heads * head_dim,
            .out_head = capacity * head_dim, // cache head stride
            .out_time = head_dim, // cache time stride
        }, base * head_dim, timeout_ns);
        // V: copy_general USE_U16, dims 3, bit-exact uint16 transport.
        const copy = try vCopyPlan(rows, capacity, v.elements);
        try a.ops.copy_u16.bind(&.{ v.bind(), a.ops.dummyBinding(), cache_values.bind(), a.ops.dummyBinding() });
        var p: params.Params = .{
            .count = copy.elements,
            .dims = 3,
            .shape = .{ @intCast(head_dim), @intCast(kv_heads), try ops_mod.cast32(rows), 0 },
            .in_strides = .{ 1, @intCast(head_dim), @intCast(kv_heads * head_dim), 0 },
            .out_strides = .{ 1, try ops_mod.cast32(capacity * head_dim), @intCast(head_dim), 0 },
        };
        p.output_offset = try ops_mod.cast32(base * head_dim);
        try a.ops.copy_u16.run(cmds, std.mem.asBytes(&p), try groups(copy.elements, a.ops.copy_u16.local[0], a.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }

    // SDPA scales q once, streams f32 scores; Metal scales per-score in 128-key chunks. No bit-exact/performance claim.
    pub fn attend(a: *Attention, cmds: *Commands, y: View, queries: View, cache_keys: View, cache_values: View, prefix: u64, rows: u64, capacity: u64, timeout_ns: u64) Error!void {
        const plan = try attendPlan(prefix, rows, capacity, queries.elements, y.elements, cache_keys.elements, cache_values.elements);
        if (y.overlaps(queries) or y.overlaps(cache_keys) or y.overlaps(cache_values)) return error.Aliased;
        try a.sdpa_bf16.bind(&.{ queries.bind(), cache_keys.bind(), cache_values.bind(), y.bind() });
        var p: params.Params = .{
            .matrix_m = @intCast(query_heads), // repeats = 8/2 = 4
            .matrix_n = @intCast(kv_heads),
            .matrix_k = plan.total_keys,
            .flags = 0,
            .alpha = scale,
            .dims = plan.rows, // q_len
            .shape = .{ @intCast(head_dim), try ops_mod.cast32(capacity * head_dim), @intCast(head_dim), 1 },
            .in_strides = .{ 1, try ops_mod.cast32(capacity * head_dim), @intCast(head_dim), 1 },
            .out_strides = .{ @intCast(query_heads * head_dim), 0, 0, 0 },
        };
        const groups_x = std.math.cast(u32, plan.rows) orelse return error.TooLarge;
        try a.sdpa_bf16.run(cmds, std.mem.asBytes(&p), .{ @intCast(query_heads), groups_x, 1 }, timeout_ns);
    }

    // Gate is bf16(ctx*bf16(sigmoid(gate))); qraw second halves hold gate. Shapes: (rows,2048), qraw (rows,4096).
    pub fn gateOutput(a: *Attention, cmds: *Commands, out: View, ctx: View, qraw: View, rows: u64, timeout_ns: u64) Error!void {
        const plan = try gatePlan(out.elements, ctx.elements, qraw.elements, rows);
        if (out.overlaps(ctx) or out.overlaps(qraw)) return error.Aliased;
        try a.gate_bf16.bind(&.{ ctx.bind(), qraw.bind(), out.bind() });
        const p: GateParams = .{ .words = plan.words };
        try a.gate_bf16.run(cmds, std.mem.asBytes(&p), try groups(plan.words, a.gate_bf16.local[0], a.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }

    fn offsetBinding(a: *const Attention) dispatch.Binding {
        return .{ .buffer = &a.offsets, .len = 4 };
    }

    // out_base offsets cache; trig uses omarchy_trig.h wrappers, not metal::cos/sin; exp/pow/cos may differ in ulps.
    fn ropeRun(a: *Attention, cmds: *Commands, plan: RopePlan, strides: RopeStrides, out_base: u64, timeout_ns: u64) Error!void {
        // Passthrough inv_freq is exp(-i*beta), with beta=ln(1e7)/32, yielding 1e7^(-i/32) up to exp/pow ulps.
        var p: params.Params = .{
            .count = plan.words,
            .flags = rope_flags,
            .alpha = 1.0, // position = 1 * (t + base)
            .beta = @log(theta) / @as(f32, rotary_pairs),
            .dims = @intCast(rotary_pairs), // half_dims
            .shape = .{ plan.heads, plan.rows, @intCast(head_dim), 0 }, // heads, time, rotated-axis extent
            .in_strides = .{ @intCast(strides.in_head), try ops_mod.cast32(strides.in_time), 1, 0 },
            .out_strides = .{ try ops_mod.cast32(strides.out_head), try ops_mod.cast32(strides.out_time), 1, 0 },
        };
        p.output_offset = try ops_mod.cast32(out_base);
        try a.rope_bf16.run(cmds, std.mem.asBytes(&p), try groups(plan.words, a.rope_bf16.local[0], a.dev.properties.limits.maxComputeWorkGroupCount[0]), timeout_ns);
    }
};

/// ropeRun's push flags: bit0 forward, bit3 passthrough tail (fast_rope.comp).
const rope_flags: u32 = 1 | 8;

const RopeStrides = struct { in_head: u64, in_time: u64, out_head: u64, out_time: u64 };

/// (rows, heads, 256) bf16, contiguous on both sides.
fn queryStrides() RopeStrides {
    return .{
        .in_head = head_dim,
        .in_time = query_heads * head_dim,
        .out_head = head_dim,
        .out_time = query_heads * head_dim,
    };
}

fn groups(count: u64, local: u32, max_x: u32) Error![3]u32 {
    const n = std.math.cast(u32, (count + local - 1) / local) orelse return error.TooLarge;
    return .{ @min(n, max_x), 1, 1 };
}

// Rope geometry: one passthrough+rotation dispatch over packed bf16 words; pure and host-tested.
pub const RopePlan = struct { rows: u32, heads: u32, words: u32 };

pub fn ropePlan(out_elements: u64, in_elements: u64, rows: u64, heads: u64) Error!RopePlan {
    if (rows == 0 or rows > std.math.maxInt(u32)) return error.Invalid;
    const row_elems = heads *| head_dim;
    const total = rows *| row_elems;
    if (in_elements != total or out_elements != total) return error.Invalid;
    if (heads != query_heads and heads != kv_heads) return error.Invalid;
    return .{
        .rows = @intCast(rows),
        .heads = @intCast(heads),
        .words = try ops_mod.cast32(total / 2), // one word per bf16 pair
    };
}

/// V scatter geometry (rows of 2*256 elements into a capacity cache).
pub const VCopyPlan = struct { elements: u32 };

pub fn vCopyPlan(rows: u64, capacity: u64, v_elements: u64) Error!VCopyPlan {
    if (rows == 0 or capacity == 0 or rows > capacity) return error.Invalid;
    const row_elems = kv_heads * head_dim;
    const total = rows *| row_elems;
    if (v_elements != total) return error.Invalid;
    if (capacity *| row_elems > std.math.maxInt(u32)) return error.TooLarge; // output offsets stay u32
    return .{ .elements = try ops_mod.cast32(total) };
}

// Causal SDPA row r sees prefix+r+1 keys, including its own; later keys are unread. Pure and host-tested.
pub const AttendPlan = struct { rows: u32, total_keys: u32 };

pub fn attendPlan(prefix: u64, rows: u64, capacity: u64, q_elements: u64, y_elements: u64, k_elements: u64, v_elements: u64) Error!AttendPlan {
    if (rows == 0 or rows > std.math.maxInt(u32)) return error.Invalid;
    const total = std.math.add(u64, prefix, rows) catch return error.TooLarge;
    if (capacity == 0 or total > capacity) return error.OutOfRange;
    if (total > max_visible_keys) return error.ContextTooLong;
    const row_elems = query_heads * head_dim;
    if (q_elements != rows *| row_elems or y_elements != rows *| row_elems) return error.Invalid;
    const cache_elems = kv_heads *| capacity *| head_dim;
    if (k_elements != cache_elems or v_elements != cache_elems) return error.Invalid;
    if (capacity *| head_dim > std.math.maxInt(u32)) return error.TooLarge;
    return .{ .rows = @intCast(rows), .total_keys = try ops_mod.cast32(total) };
}

/// Gate geometry; pure and host-tested.
pub const GatePlan = struct { rows: u32, words: u32 };

pub fn gatePlan(out_elements: u64, ctx_elements: u64, qraw_elements: u64, rows: u64) Error!GatePlan {
    if (rows == 0 or rows > std.math.maxInt(u32)) return error.Invalid;
    if (ctx_elements != rows *| gate_row_elems or out_elements != rows *| gate_row_elems) return error.Invalid;
    if (qraw_elements != rows *| (2 * gate_row_elems)) return error.Invalid;
    return .{
        .rows = @intCast(rows),
        .words = try ops_mod.cast32(rows * gate_row_words),
    };
}

const testing = std.testing;

test "ropePlan derives the word count and refuses torn or foreign shapes" {
    const plan = try ropePlan(4 * 8 * 256, 4 * 8 * 256, 4, query_heads);
    try testing.expectEqual(@as(u32, 4 * 8 * 128), plan.words);
    const kv = try ropePlan(3 * 2 * 256, 3 * 2 * 256, 3, kv_heads);
    try testing.expectEqual(@as(u32, 3 * 2 * 128), kv.words);
    try testing.expectError(error.Invalid, ropePlan(4 * 8 * 256 + 1, 4 * 8 * 256, 4, query_heads));
    try testing.expectError(error.Invalid, ropePlan(4 * 8 * 256, 4 * 8 * 256, 4, 4)); // not the model's head count
    try testing.expectError(error.Invalid, ropePlan(0, 0, 0, query_heads));
    try testing.expectError(error.TooLarge, ropePlan(1 << 35, 1 << 35, 1 << 24, query_heads)); // words past u32
}

test "attendPlan keeps the causal prefix and refuses long or torn contexts" {
    // rows > 1: each row's visible count is prefix + row + 1.
    const plan = try attendPlan(100, 3, 128, 3 * 8 * 256, 3 * 8 * 256, 2 * 128 * 256, 2 * 128 * 256);
    try testing.expectEqual(@as(u32, 103), plan.total_keys);
    try testing.expectEqual(@as(u32, 3), plan.rows);
    try testing.expectError(error.OutOfRange, attendPlan(126, 3, 128, 3 * 2048, 3 * 2048, 2 * 128 * 256, 2 * 128 * 256));
    try testing.expectError(error.ContextTooLong, attendPlan(max_visible_keys, 1, max_visible_keys + 1, 2048, 2048, 2 * (max_visible_keys + 1) * 256, 2 * (max_visible_keys + 1) * 256));
    try testing.expectError(error.Invalid, attendPlan(0, 3, 128, 2 * 2048, 3 * 2048, 2 * 128 * 256, 2 * 128 * 256));
    try testing.expectError(error.Invalid, attendPlan(0, 3, 128, 3 * 2048, 3 * 2048, 2 * 127 * 256, 2 * 128 * 256));
    try testing.expectError(error.TooLarge, attendPlan(0, 1, 1 << 25, 2048, 2048, 2 * (1 << 25) * 256, 2 * (1 << 25) * 256));
}

test "vCopyPlan refuses rows past capacity and caches past u32 offsets" {
    const plan = try vCopyPlan(4, 16, 4 * 2 * 256);
    try testing.expectEqual(@as(u32, 4 * 2 * 256), plan.elements);
    try testing.expectError(error.Invalid, vCopyPlan(17, 16, 17 * 2 * 256));
    try testing.expectError(error.Invalid, vCopyPlan(4, 16, 5 * 2 * 256));
    try testing.expectError(error.TooLarge, vCopyPlan(1, 1 << 24, 2 * 256));
}

test "gatePlan derives the word count and refuses torn rows" {
    const plan = try gatePlan(4 * 2048, 4 * 2048, 4 * 4096, 4);
    try testing.expectEqual(@as(u32, 4 * 1024), plan.words);
    try testing.expectError(error.Invalid, gatePlan(4 * 2048, 4 * 2048 + 1, 4 * 4096, 4));
    try testing.expectError(error.Invalid, gatePlan(4 * 2048, 4 * 2048, 4 * 4095, 4));
    try testing.expectError(error.Invalid, gatePlan(0, 0, 0, 0));
}

test "rope push flags match the masks the embedded rope shader actually tests" {
    const src = @import("vulkan_comp_sources").fast_rope;
    try testing.expectEqual(@as(u32, 1 | 8), rope_flags);
    try testing.expect(std.mem.indexOf(u8, src, "(params.flags & 1u)") != null);
    try testing.expect(std.mem.indexOf(u8, src, "(params.flags & 8u)") != null);
}
