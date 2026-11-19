//! Hy4-flash's compressed sparse indexer on Metal, after the reference's CompressedIndexer: every pass caches its
//! rows' normalized token keys and re-pools the four-slot groups those rows touch (a group is read only once it
//! closed, so a pass that wrote part of one leaves nothing stale), and from 2,048 keys on each row attends its 511
//! best-scoring closed groups plus its own group's causal part. Slots are positions here.
const std = @import("std");
const mtl = @import("metal");
const source = @import("kernel_sources");
const dense = @import("metal_dense.zig");
const mla_gpu = @import("mla_metal.zig");
const rope = @import("rope.zig");
const weights = @import("model_weights.zig");
const recall = @import("index_recall.zig");

pub const Ref = dense.Ref;
pub const heads = recall.heads;
pub const width = recall.width;
/// group_topk: 511 closed groups and the query's own.
pub const budget_groups: u32 = 512;
/// A row's recalled key list: 512 groups of four slots.
pub const keys_a_row: u32 = budget_groups * 4;

pub const Weights = struct {
    token_keys: dense.Linear,
    token_norm: mtl.Buffer,
    query: dense.Linear,
    head_weights: dense.Linear,
    values: mtl.Buffer,
    gates: mtl.Buffer,
    bias: mtl.Buffer,
    norm: mtl.Buffer,

    /// The projections take the row kernels: the tensor units' packings exist only for the layers' own shapes.
    pub fn upload(b: *dense.Buffers, w: *const weights.Index, tensors: bool) !Weights {
        _ = tensors;
        return .{
            .token_keys = try b.linear(w.token_keys, false),
            .token_norm = try b.upload(w.token_norm.bytes),
            .query = try b.linear(w.query, false),
            .head_weights = try b.linear(w.head_weights, false),
            .values = try b.upload(w.group_values.bytes),
            .gates = try b.upload(w.group_gates.bytes),
            .bias = try b.upload(w.group_bias.bytes),
            .norm = try b.upload(w.group_norm.bytes),
        };
    }
};

pub const Kernels = struct {
    keys: mtl.Pipeline,
    groups: mtl.Pipeline,
    score: mtl.Pipeline,
    select: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Kernels {
        const build = try mtl.Library.fromSource(device, source.hy4_flash_index_build, mtl.CompileOptions.mlx());
        defer build.deinit();
        const rec = try mtl.Library.fromSource(device, source.hy4_flash_index_recall, mtl.CompileOptions.mlx());
        defer rec.deinit();
        var keys = try mtl.Pipeline.init(device, build, "tf_hy4_index_keys", false);
        errdefer keys.deinit();
        var groups = try mtl.Pipeline.init(device, build, "tf_hy4_index_groups", false);
        errdefer groups.deinit();
        var score = try mtl.Pipeline.init(device, rec, "tf_hy4_group_scores", false);
        errdefer score.deinit();
        return .{ .keys = keys, .groups = groups, .score = score, .select = try mtl.Pipeline.init(device, rec, "tf_hy4_recall_groups", false) };
    }

    pub fn deinit(self: *Kernels) void {
        self.keys.deinit();
        self.groups.deinit();
        self.score.deinit();
        self.select.deinit();
    }
};

/// A pass's indexer rows: each row's projections, and the groups it touches with their rotary positions.
pub const Scratch = struct {
    own: dense.Buffers,
    capacity: u32,
    keys: Ref,
    query: Ref,
    head_weights: Ref,
    recall_rows: mtl.Buffer,
    group_positions: mtl.Buffer,
    group_table: Ref,
    first: u32 = 0,
    rows: u32 = 0,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, rows: u32) !Scratch {
        var b: dense.Buffers = .{ .a = a, .d = device };
        errdefer b.clear();
        const groups = rows / 4 + 2;
        var out: Scratch = .{
            .own = undefined,
            .capacity = rows,
            .keys = try b.rows(rows, width),
            .query = try b.rows(rows, heads * width),
            .head_weights = try b.rows(rows, heads),
            .recall_rows = try b.alloc(@as(usize, rows) * @sizeOf(recall.Row)),
            .group_positions = try b.alloc(@as(usize, groups) * 3 * @sizeOf(i32)),
            .group_table = try b.rows(groups, 2 * rope.pairs),
        };
        out.own = b;
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.own.clear();
        self.* = undefined;
    }

    /// The pass's rows sit at positions first.. (consecutive); call before encoding, after the previous pass completed.
    pub fn stage(self: *Scratch, first: u32, rows: u32) !void {
        if (rows == 0 or rows > self.capacity) return error.BadHy4IndexRows;
        self.first = first;
        self.rows = rows;
        const r = self.recall_rows.slice(recall.Row, rows);
        for (r, 0..) |*row, i| row.* = .{ .group_base = 0, .token = first + @as(u32, @intCast(i)) };
        const g0 = first / 4;
        const count = (first + rows - 1) / 4 - g0 + 1;
        const pos = self.group_positions.slice(i32, count * 3);
        for (0..count) |g| pos[g * 3 ..][0..3].* = @splat(@intCast((g0 + g) * 4));
    }

    pub fn groupRange(self: *const Scratch) [2]u32 {
        const g0 = self.first / 4;
        return .{ g0, (self.first + self.rows - 1) / 4 - g0 + 1 };
    }

    /// Whether any row's attention takes recalled lists: a row at 2,048 or past has more closed groups than it keeps.
    pub fn selects(self: *const Scratch) bool {
        return self.first + self.rows - 1 >= keys_a_row;
    }
};

/// A sparse layer's indexer state for one stream, and the pass's shared score rows and this layer's key lists.
pub const Cache = struct {
    token_keys: Ref, // [capacity][128] BF16
    group_keys: Ref, // [capacity / 4][128] BF16, RoPE'd at each group's first slot
    capacity: u32, // slots (a multiple of 4)
    scores: Ref, // [rows][score_groups] fp32, shared by every layer of a pass
    score_groups: u32,
    tokens: Ref, // [rows][2048] i32: each row's recalled slots, this layer's (the drafter reads the anchor's)
    counts: Ref, // [rows] i32

    pub fn tokenBytes(capacity: u32) usize {
        return @as(usize, capacity) * width * 2;
    }

    pub fn groupBytes(capacity: u32) usize {
        return @as(usize, capacity / 4) * width * 2;
    }
};

/// What attendSeam needs to run a sparse layer's indexer.
pub const Pass = struct {
    kernels: *const Kernels,
    scratch: *const Scratch,
    cache: Cache,
};

const KeysArgs = extern struct { rows: u32, capacity: u32, epsilon: f32 };
const GroupArgs = extern struct { first: u32, count: u32, capacity_groups: u32, epsilon: f32 };
const RecallShape = extern struct { rows: u32, score_groups: u32, group_budget: u32 };

/// The indexer for the staged rows: `normalized` is the layer's input (the reference's hidden_states), `q_resid`
/// the normalized q_a rows. Leaves the key lists in `c.tokens` and `c.counts` when the scratch selects. Ends
/// without a trailing barrier.
pub fn run(p: Pass, common: *const dense.Kernels, project: mla_gpu.Projections, e: mtl.ComputeEncoder, w: *const Weights, normalized: Ref, q_resid: Ref, write_slots: Ref, positions: Ref, table: Ref, r: rope.Parameters, epsilon: f32) !void {
    encode(p, common, project, e, w, normalized, q_resid, write_slots, positions, table, r, epsilon) catch |err| {
        std.log.err("hy4 indexer: {s}", .{@errorName(err)});
        return err;
    };
}

fn encode(p: Pass, common: *const dense.Kernels, project: mla_gpu.Projections, e: mtl.ComputeEncoder, w: *const Weights, normalized: Ref, q_resid: Ref, write_slots: Ref, positions: Ref, table: Ref, r: rope.Parameters, epsilon: f32) !void {
    const k = p.kernels;
    const s = p.scratch;
    const c = p.cache;
    const rows = s.rows;
    if (rows == 0) return error.BadHy4IndexRows;
    const range = s.groupRange();
    if ((range[0] + range[1]) * 4 > c.capacity) return error.Hy4IndexPastCapacity;
    // the projections, and both rotary tables (positions only)
    try common.linear(e, w.token_keys, normalized, s.keys, rows);
    try common.linear(e, w.head_weights, normalized, s.head_weights, rows);
    try common.linear(e, w.query, q_resid, s.query, rows);
    project.rotaryTable(e, positions, table, rows, r);
    project.rotaryTable(e, .{ .buffer = s.group_positions }, s.group_table, range[1], r);
    e.barrier();
    // each row's normalized key at its slot; the query heads' trailing 64 channels rotated in place
    e.setPipeline(k.keys);
    e.setBuffer(s.keys.buffer, s.keys.offset, 0);
    e.setBuffer(w.token_norm, 0, 1);
    e.setBuffer(write_slots.buffer, write_slots.offset, 2);
    e.setBuffer(c.token_keys.buffer, c.token_keys.offset, 3);
    e.setValue(KeysArgs{ .rows = rows, .capacity = c.capacity, .epsilon = epsilon }, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(width, 1, 1));
    const q_rot: Ref = .{ .buffer = s.query.buffer, .offset = s.query.offset + (width - 2 * rope.pairs) * 2 };
    try project.rotate(e, q_rot, table, q_rot, .{ .rows = rows, .heads = heads, .input_stride = width, .output_stride = width });
    e.barrier();
    // the touched groups pooled from the cached keys
    e.setPipeline(k.groups);
    e.setBuffer(c.token_keys.buffer, c.token_keys.offset, 0);
    e.setBuffer(w.values, 0, 1);
    e.setBuffer(w.gates, 0, 2);
    e.setBuffer(w.bias, 0, 3);
    e.setBuffer(w.norm, 0, 4);
    e.setBuffer(c.group_keys.buffer, c.group_keys.offset, 5);
    e.setValue(GroupArgs{ .first = range[0], .count = range[1], .capacity_groups = c.capacity / 4, .epsilon = epsilon }, 6);
    e.dispatchGroups(mtl.Size.of(range[1], 1, 1), mtl.Size.of(width, 1, 1));
    e.barrier();
    const g_rot: Ref = .{ .buffer = c.group_keys.buffer, .offset = c.group_keys.offset + (@as(usize, range[0]) * width + width - 2 * rope.pairs) * 2 };
    try project.rotate(e, g_rot, s.group_table, g_rot, .{ .rows = range[1], .heads = 1, .input_stride = width, .output_stride = width });
    if (!s.selects()) return;
    e.barrier();
    const last = s.first + rows - 1;
    const shape: RecallShape = .{ .rows = rows, .score_groups = c.score_groups, .group_budget = budget_groups };
    if (last / 4 > c.score_groups) return error.Hy4IndexPastCapacity;
    e.setPipeline(k.score);
    e.setBuffer(s.query.buffer, s.query.offset, 0);
    e.setBuffer(c.group_keys.buffer, c.group_keys.offset, 1);
    e.setBuffer(s.head_weights.buffer, s.head_weights.offset, 2);
    e.setBuffer(s.recall_rows, 0, 3);
    e.setBuffer(c.scores.buffer, c.scores.offset, 4);
    e.setValue(shape, 5);
    e.setValue(recall.head_scale, 6);
    e.dispatchGroups(mtl.Size.of((last / 4 + 3) / 4, rows, 1), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(k.select);
    e.setBuffer(c.scores.buffer, c.scores.offset, 0);
    e.setBuffer(s.recall_rows, 0, 1);
    e.setBuffer(c.tokens.buffer, c.tokens.offset, 2);
    e.setBuffer(c.counts.buffer, c.counts.offset, 3);
    e.setValue(shape, 4);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
}
