//! Original Inkling 2 Mini checkpoint, GPU pipelines and per-forward scratch.
const std = @import("std");
const Io = std.Io;
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");
const state = @import("metal_state.zig");
const wts = @import("metal_weights.zig");
const ops_mod = @import("metal_ops.zig");
const ex = @import("metal_experts.zig");
const fwd = @import("metal_forward.zig");
const plan_mod = @import("plan.zig");
const loop_mod = @import("metal_loop.zig");
const Mapped = @import("metal_mapped.zig").Mapped;

pub const Options = struct {
    /// Positions a stream created by `newState` holds.
    capacity: u32 = 4096,
    /// Concurrent streams the host plans (informational for admission; states are created by the caller).
    streams: u32 = 1,
    /// The most rows of one forward over all items (a shared decode window or a prompt chunk).
    rows: u32 = 16,
    /// Draft head requested: allocates the tap buffer (needs mtp_config).
    drafts: bool = false,
    /// Image rows requested: forward accepts Item.images only when set.
    vision: bool = false,
};

/// Per-forward activations for `rows` rows: row-major, stage by stage in the order the graph writes them.
pub const Scratch = struct {
    ids: mtl.Buffer, // u32 [rows]
    hid0: mtl.Buffer, // bf16 [rows, H]: the stream entering a layer, and leaving it (ping-pong, see `hid`)
    hid1: mtl.Buffer,
    blocks: mtl.Buffer, // bf16 [nblocks][rows][H]: finished attention-residual blocks
    xin: mtl.Buffer, // bf16 [3][rows][H]: attention inputs (q, k, v for a global layer)
    xm: mtl.Buffer, // bf16 [rows, H]: the MLP input
    q: mtl.Buffer,
    kraw: mtl.Buffer,
    vraw: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    rel: mtl.Buffer,
    gate: mtl.Buffer,
    sink: mtl.Buffer,
    dynk: mtl.Buffer,
    dynv: mtl.Buffer,
    att: mtl.Buffer,
    o: mtl.Buffer, // bf16 [rows, H]: the attention output projection
    a: mtl.Buffer, // bf16 [rows, H]: after the post-attention convolution
    h32: mtl.Buffer, // f32 [rows, H]: the running partial block
    rids: mtl.Buffer, // u32 [rows, topk]
    rwts: mtl.Buffer, // f32 [rows, topk]
    moe: mtl.Buffer, // bf16 [rows, H]
    c: mtl.Buffer, // bf16 [rows, H]: after the post-MLP convolution
    xf: mtl.Buffer, // bf16 [rows, H]: the final normed hidden states
    xl: mtl.Buffer, // bf16 [rows, H]: the rows that get logits
    ylogit: mtl.Buffer, // bf16 [rows, unpadded vocab]: the head's rows before the softcap
    bank: mtl.Buffer, // BF16 blocks from the previous pass
    bank_h: mtl.Buffer, // BF16 tail of the previous pass
    msg: mtl.Buffer, // four FP32 messages per row
    mixf: mtl.Buffer, // up to three FP32 attention-residual mixes per row
    down: mtl.Buffer, // BF16 controller projections

    const names = std.meta.fieldNames(Scratch);

    /// The hidden stream entering layer `layer` (even layers read hid0 and write hid1).
    pub fn hid(s: *const Scratch, layer: u32) mtl.Buffer {
        return if (layer % 2 == 0) s.hid0 else s.hid1;
    }

    /// Bytes of each field in field order.
    fn sizes(t: *const config.Text, rows: u32) [names.len]usize {
        const r: usize = rows;
        const h: usize = t.hidden;
        const nb: usize = (t.layers + t.block - 1) / t.block;
        const qw = @max(@as(usize, t.local.q) * t.local.dim, @as(usize, t.global.q) * t.global.dim);
        const kvw = @max(@as(usize, t.local.kv) * t.local.dim, @as(usize, t.global.kv) * t.global.dim);
        const relw = @max(@as(usize, t.local.q), t.global.q) * t.d_rel;
        const qh = @max(t.local.q, t.global.q);
        const taps = @as(usize, @max(t.local.kv, t.global.kv)) * config.conv_taps;
        const loop = t.loop.?;
        return .{
            r * 4,      r * h * 2,   r * h * 2,        nb * r * h * 2, 3 * r * h * 2, r * h * 2,
            r * qw * 2, r * kvw * 2, r * kvw * 2,      r * kvw * 2,    r * kvw * 2,   r * relw * 2,
            r * qw * 2, r * qh * 2,  r * taps * 2,     r * taps * 2,   r * qw * 2,    r * h * 2,
            r * h * 2,  r * h * 4,   r * t.topk * 4,   r * t.topk * 4, r * h * 2,     r * h * 2,
            r * h * 2,  r * h * 2,   r * t.unpadded_vocab * 2,
            ((loop.end - loop.start) / t.block - 1) * r * h * 2, r * h * 2,
            4 * r * h * 4, 3 * r * h * 4, 3 * r * loop.rank * 2,
        };
    }

    /// Only the token ids cross the host boundary here.
    fn shared(comptime name: []const u8) bool {
        return std.mem.eql(u8, name, "ids");
    }

    fn init(device: mtl.Device, t: *const config.Text, rows: u32) !Scratch {
        const lengths = sizes(t, rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer s.release(made);
        inline for (names, 0..) |name, i| {
            const opt = if (shared(name)) mtl.ResourceOptions.shared else mtl.ResourceOptions.private;
            @field(s, name) = try device.buffer(@max(lengths[i], 16), opt);
            made += 1;
        }
        return s;
    }

    fn release(s: *Scratch, upto: usize) void {
        inline for (names, 0..) |name, i| if (i < upto) @field(s.*, name).deinit();
    }

    pub fn bytes(t: *const config.Text, rows: u32) u64 {
        var total: u64 = 0;
        for (sizes(t, rows)) |n| total += @max(n, 16);
        return total;
    }

    pub fn deinit(s: *Scratch) void {
        s.release(names.len);
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: config.Config,
    options: Options,
    mapped: Mapped,
    checkpoint: ck.Checkpoint,
    weights: wts.Weights,
    ops: ops_mod.Ops,
    plan: plan_mod.Plan,
    loop_ops: loop_mod.Ops,
    store: ex.Store,
    scratch: Scratch,
    /// f32 [rows, unpadded_vocab]: the softcapped logits of each item's last `logit_rows` rows, in item order.
    logits: mtl.Buffer,
    /// bf16 [rows, taps_n, hidden]: the draft taps of every row of a tap-enabled item (null without drafts).
    taps: ?mtl.Buffer,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn load(gpa: std.mem.Allocator, io: Io, dir: []const u8, options: Options) !*Model {
        if (options.rows == 0) return error.InvalidOptions;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.options = options;
        m.config = try readConfig(gpa, io, dir);
        const t = &m.config.text;
        const loop = t.loop orelse return error.NotLoopConfig;
        const blocks = (t.layers + t.block - 1) / t.block;
        if (blocks + 1 > loop_mod.max_sources or loop.rank > loop_mod.max_rank or t.hidden > loop_mod.max_hidden)
            return error.UnsupportedLoopGeometry;
        if (options.capacity == 0 or options.capacity > t.max_context) return error.InvalidOptions;
        _ = try std.math.mul(u32, options.rows, t.hidden);
        if (options.drafts and m.config.mtp == null) return error.NoDraftHead;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/model-00001-of-00001.safetensors", .{dir}, 0);
        defer gpa.free(path);
        m.mapped = try Mapped.open(path);
        errdefer m.mapped.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        try m.mapped.adopt(&m.checkpoint, try m.mapped.buffer(m.device));
        var precision: [config.max_layers]plan_mod.Precision = undefined;
        for (precision[0..t.layers], 0..) |*p, i| {
            var buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.w13_weight", .{i});
            p.* = switch ((try m.checkpoint.get(name)).dtype) {
                .u8 => .nvfp4,
                .f8_e4m3 => .mxfp8,
                else => return error.UnexpectedExpertFormat,
            };
        }
        m.plan = try plan_mod.build(t, precision[0..t.layers]);
        m.weights = try wts.load(gpa, &m.checkpoint, &m.config);
        errdefer m.weights.deinit(gpa);
        m.ops = try ops_mod.Ops.load(m.device, .{ .scratch_bytes = opsScratch(t, options) });
        errdefer m.ops.deinit();
        m.loop_ops = try loop_mod.Ops.load(m.device);
        errdefer m.loop_ops.deinit();
        m.store = try ex.Store.load(gpa, m.device, &m.checkpoint, &m.config, options.rows);
        errdefer m.store.deinit();
        m.scratch = try Scratch.init(m.device, t, options.rows);
        errdefer m.scratch.deinit();
        m.logits = try m.device.buffer(@as(usize, options.rows) * t.unpadded_vocab * 4, mtl.ResourceOptions.shared);
        errdefer m.logits.deinit();
        m.taps = null;
        if (options.drafts) m.taps = try m.device.buffer(@as(usize, options.rows) * m.config.mtp.?.taps_n * t.hidden * 2, mtl.ResourceOptions.shared);
        m.follow = null;
        m.mapped.warm(8);
        return m;
    }

    /// The shared forward over several streams' rows (metal_forward.zig), and the exact state operations.
    pub const forward = fwd.forward;
    pub const keep = fwd.keep;
    pub const reset = fwd.reset;
    pub const save = fwd.save;
    pub const restore = fwd.restore;
    pub const logitBase = fwd.logitBase;
    pub const tapBase = fwd.tapBase;

    pub fn deinit(m: *Model) void {
        m.setResident(null);
        if (m.taps) |b| b.deinit();
        m.logits.deinit();
        m.scratch.deinit();
        m.store.deinit();
        m.loop_ops.deinit();
        m.ops.deinit();
        m.weights.deinit(m.gpa);
        m.checkpoint.deinit();
        m.mapped.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }

    /// Shared mapped weights plus graph and expert scratch, excluding per-stream state.
    pub fn fixedBytes(m: *const Model) u64 {
        var total = m.weightBytes() + Scratch.bytes(&m.config.text, m.options.rows) + m.ops.scratch.length() + m.store.scratchBytes() + m.logits.length();
        if (m.taps) |b| total += b.length();
        return total;
    }

    /// Resident mapped checkpoint bytes (dense, routed, and vision weights share one buffer).
    pub fn weightBytes(m: *const Model) u64 {
        return m.checkpoint.residentBytes();
    }

    pub fn geometry(m: *const Model) state.Geometry {
        return .{ .capacity = m.options.capacity, .rows = m.options.rows };
    }

    /// A stream of Options.capacity positions, empty.
    pub fn newState(m: *const Model) !state.State {
        return state.State.init(m.gpa, m.device, &m.config, m.geometry());
    }

    /// Enroll every buffer the model owns with a residency hook (or release them); streams enroll their own.
    pub fn setResident(m: *Model, hook: ?mtl.ResidencySet.Follow) void {
        if (m.follow) |f| m.apply(f, false);
        m.follow = hook;
        if (hook) |f| m.apply(f, true);
        m.store.setResident(hook);
    }

    fn apply(m: *Model, f: mtl.ResidencySet.Follow, joining: bool) void {
        for (m.checkpoint.shards.items) |sh| mtl.resident.fields.one(sh.buffer, f, joining);
        inline for (Scratch.names) |name| mtl.resident.fields.one(@field(m.scratch, name), f, joining);
        mtl.resident.fields.one(m.ops.scratch, f, joining);
        mtl.resident.fields.one(m.logits, f, joining);
        if (m.taps) |b| mtl.resident.fields.one(b, f, joining);
    }
};

fn opsScratch(t: *const config.Text, o: Options) usize {
    const global_chunks = (@as(usize, o.capacity) + ops_mod.attn_chunk - 1) / ops_mod.attn_chunk;
    const local_chunks = (@as(usize, @min(o.capacity, t.window)) + ops_mod.attn_chunk - 1) / ops_mod.attn_chunk;
    const head_chunks = @max(if (global_chunks > 1) t.global.q * global_chunks else 0, if (local_chunks > 1) t.local.q * local_chunks else 0);
    const one = head_chunks * ops_mod.attn_partial_floats * @sizeOf(f32);
    const ceiling = (ops_mod.Options{}).scratch_bytes;
    const attention = @max(one, @min(ceiling, one * o.rows));
    const router = @as(usize, o.rows) * t.experts * @sizeOf(f32);
    return @max(@max(attention, router), 16);
}

fn readConfig(gpa: std.mem.Allocator, io: Io, dir: []const u8) !config.Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const json = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(json);
    var why: config.Why = .{};
    return config.parse(gpa, json, &why) catch |err| {
        std.log.err("inkling2 config: {s}", .{why.text()});
        return err;
    };
}
