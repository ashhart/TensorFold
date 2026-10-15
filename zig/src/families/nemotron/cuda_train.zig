//! Learning in the weights on CUDA (train.zig on Metal): sequences side by side forward, their losses, and back.
const std = @import("std");
const train_mode = @import("train_mode.zig");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const ops = @import("cuda_train_ops.zig");
const sites_mod = @import("cuda_sites.zig");
const lesson_route = @import("lesson_route.zig");
const back = @import("cuda_train_back.zig");
const tfw = @import("cuda_train_fwd.zig");
const lay = @import("cuda_train_layout.zig");
const prefix = @import("cuda_train_prefix.zig");
const Count = @import("pocket_clock.zig").Count;
const Source = @import("cuda_source.zig").Source;
const taps = @import("cuda_train_tap.zig");

/// Rows a sequence learns from at most: a question and its answer (attention's backward reads at most 256).
pub const max_rows = 256;

/// Rows a pass holds at most, its sequences side by side.
pub const pack_rows = 1024;

/// Sequences a pass holds at most.
pub const max_seqs = 32;

pub const Mode = train_mode.Mode;
pub const Result = train_mode.Result;
pub const Nudge = train_mode.Nudge;
pub const Seq = train_mode.Seq;
pub const Span = train_mode.Span;
pub const Answer = train_mode.Answer;
pub const Turn = lesson_route.Turn;
const reads = train_mode.reads;

/// Where the candidate projections start in the mapped host block (after targets, weights, stats and scales).
const proj_at = pack_rows * 20;

/// The trainer's own sequence: cache rows enough for a step, one attention chunk.
const seq_len = state.chunk_keys;

/// A matrix [n, k] as the checkpoint holds it, on the GPU: 4-bit words, scales and biases, or bf16 (a lesson's fold).
pub const Mat = struct { w: u64, s: u64 = 0, b: u64 = 0, bf16: bool = false, n: usize, k: usize };

/// What each layer's backward reads of the weights, in the checkpoint's own layout.
pub const Raw = union(enum) {
    mamba: struct { in_proj: Mat, out_proj: Mat },
    attention: struct { q: Mat, k: Mat, v: Mat, o: Mat },
    moe: struct { up: Mat, down: Mat, fc1: [3]u64, fc2: [3]u64 },
};

/// Where each layer keeps what its backward reads of its forward (bf16 unless named), at pack_rows rows.
const Kept = struct {
    x: u64 = 0, // the normed input (attention, MoE)
    proj: u64 = 0, // Mamba: the in-projection's output
    ya: u64 = 0, // Mamba, attention: the output projection's input
    upr: u64 = 0, // MoE: the shared expert's whole activation [rows, F]
    act: u64 = 0, // MoE: every pair's up activation [rows * slots, W]
    ys: u64 = 0, // MoE: every pair's down output before its weight [rows * slots, D]
    ids: u64 = 0, // MoE: i32 [rows * slots]
    wts: u64 = 0, // MoE: f32 [rows * slots]
};

/// Host memory the GPU reads and writes too (mapped), sliced as the pocket reads Metal's shared buffers.
pub const Shared = struct {
    m: sites_mod.Mapped,

    pub fn slice(s: *const Shared, comptime T: type, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(s.m.host.bytes.ptr)))[0..n];
    }

    pub fn dev(s: *const Shared) u64 {
        return s.m.dev;
    }
};

pub const Trainer = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    c: Config,
    sites: sites_mod.Sites,
    seq: state.Seq,
    buf: state.Buffers, // the engine's scratch shape, our own, bound to `seq`
    raw: []Raw,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty, // the raw weights and the LM head transposed
    head_t: u64 = 0, // bf16 [V, D]: the LM head dequantized, for the loss's backward product
    mem: lay.Arena,
    kept: []Kept,
    saved: u64 = 0, // bf16 [layers + 1, pack_rows, D]: the residual entering each layer, then the last one
    logits: u64 = 0, // bf16 [pack_rows, V]: the logits, then their gradient
    host: sites_mod.Mapped, // the answer rows' targets u32, weights f32, stats [2]f32; the rows' scales f32; then proj
    g: u64 = 0,
    dx: u64 = 0,
    gs: u64 = 0,
    dhn: u64 = 0, // bf16 [pack_rows, D]: the final-normed answer rows, the head's input
    dxa: u64 = 0,
    xb: u64 = 0, // bf16 [max_rows, widest]: a product's left side
    wt: u64 = 0, // bf16 [widest n, k]: a product's dequantized matrix
    d_upr: u64 = 0,
    du: u64 = 0,
    dyp: u64 = 0,
    dxp: u64 = 0,
    d_y1r: u64 = 0,
    du1: u64 = 0,
    y1s: u64 = 0,
    tiles: u64 = 0, // the experts' backward tiles: (expert, first pair) each
    order: u64 = 0,
    inv: u64 = 0,
    offsets: u64 = 0,
    lg: u64 = 0,
    dyn: u64 = 0,
    ytot: u64 = 0,
    dytot: u64 = 0,
    dt: u64 = 0,
    ddt: u64 = 0,
    dact: u64 = 0,
    dproj: u64 = 0,
    ckpt: u64 = 0,
    dbp: u64 = 0,
    dcp: u64 = 0,
    conv_out: u64 = 0,
    act_m: u64 = 0,
    qkv: u64 = 0,
    d_o: u64 = 0,
    dq: u64 = 0,
    dk: u64 = 0,
    dv: u64 = 0,
    probs: u64 = 0,
    dscores: u64 = 0,
    nudge: train_mode.Nudge = .{}, // where a one-sequence target step adds `delta`
    delta: Shared = undefined, // f32 [max_seqs, 3, layers, D]: each sequence's nudges after each layer it names
    push: Shared = undefined, // f32 [max_seqs, 3, layers, D]: the loss's gradient at each nudged row after each of them
    inputs: Shared = undefined, // bf16 [layers, pack_rows, D]: each MoE layer's normed input, as a keys or target pass read it
    residuals: Shared = undefined, // bf16 [layers + 1, pack_rows, D]: the residuals a target pass saved, for the host
    sketched: usize = 0, // rows sketched since the sketches were cleared, each row's place in them
    route: lesson_route.Route = .{}, // this step's: which rows carry the lessons
    from: [@import("config.zig").max_layers]u32 = @splat(std.math.maxInt(u32)), // each layer's open pocket neurons, from here on
    spans: [max_seqs]Span = undefined, // where the last pass's sequences sat
    answers: [max_seqs]Answer = undefined, // and their answer rows, packed for the head
    answered: usize = 0, // the head's rows
    seqs: usize = 0,
    pack: usize = pack_rows, // rows a pass may hold here
    prefixes: prefix.Store = undefined, // sequences' prompt prefixes, run once and their states kept
    count: Count = .{}, // passes run and the rows they held
    tap: ?taps.Tap = null, // made the first time a prompt in a conversation is read (tapRows)

    /// The model's shape, as the pocket and the probe read it on either backend.
    pub fn config(t: *const Trainer) Config {
        return t.c;
    }

    /// The device bytes `init` takes for config `c` at most: what a server keeps out of its streams' room with --slide.
    pub fn memory(c: Config) usize {
        var raw: [@import("config.zig").max_layers]Raw = undefined;
        var kept: [@import("config.zig").max_layers]Kept = undefined;
        const quad = struct {
            fn f(n: usize, k: usize) usize {
                return n * k / 2 + n * k / 16 + 2 * 256; // words, then bf16 scales and biases a 64-group each
            }
        }.f;
        const shape = struct {
            fn f(n: usize, k: usize) Mat {
                return .{ .w = 0, .n = n, .k = k };
            }
        }.f;
        const shared = c.shared_width + c.pocket;
        var copies: usize = quad(c.vocab, c.hidden) + c.vocab * c.hidden * 2; // the LM head, and dequantized
        for (0..c.layers) |l| {
            raw[l] = switch (c.kinds[l]) {
                .mamba => .{ .mamba = .{ .in_proj = shape(c.projDim(), c.hidden), .out_proj = shape(c.hidden, c.inner()) } },
                .attention => .{ .attention = .{ .q = shape(c.heads * c.head_dim, c.hidden), .k = shape(c.kv_heads * c.head_dim, c.hidden), .v = shape(c.kv_heads * c.head_dim, c.hidden), .o = shape(c.hidden, c.heads * c.head_dim) } },
                .moe => .{ .moe = .{ .up = shape(shared, c.hidden), .down = shape(c.hidden, shared), .fc1 = @splat(0), .fc2 = @splat(0) } },
            };
            copies += switch (raw[l]) {
                .mamba => |m| quad(m.in_proj.n, m.in_proj.k) + quad(m.out_proj.n, m.out_proj.k),
                .attention => |a| quad(a.q.n, a.q.k) + quad(a.k.n, a.k.k) + quad(a.v.n, a.v.k) + quad(a.o.n, a.o.k),
                .moe => |m| quad(m.up.n, m.up.k) + quad(m.down.n, m.down.k) + 2 * c.experts * quad(c.expert_width, c.hidden),
            };
        }
        var t: Trainer = undefined;
        t.c = c;
        t.raw = raw[0..c.layers];
        t.kept = kept[0..c.layers];
        const mapped = proj_at + c.layers * max_rows * sites_mod.candidates * 4 + 2 * max_seqs * 3 * c.layers * c.hidden * 4 + (2 * c.layers + 1) * pack_rows * c.hidden * 2;
        const prefixes = prefix.max_slots * prefix.Store.init(undefined, c, max_rows).bytes();
        return copies + lay.layout(&t, true) + mapped + prefixes + state.Seq.bytes(c, seq_len, &.{}) + (1 << 30); // a GiB for the sites and the scratch
    }

    /// Every buffer a step needs and the raw weights its backward reads; `living`, saved blocks the trainer takes over.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8, living: *?sites_mod.Sites) !*Trainer {
        const c = e.c;
        if (c.mamba_head_dim != 64 or c.state != 128 or c.top_k > 16 or c.experts > 256) return error.UnsupportedShape;
        const d = e.ctx.d;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .e = e, .c = c, .sites = undefined, .seq = undefined, .buf = undefined, .raw = undefined, .mem = undefined, .kept = undefined, .host = undefined, .prefixes = prefix.Store.init(d, c, max_rows) };
        const saved = living.* != null;
        t.sites = if (living.*) |s| s else try sites_mod.Sites.init(gpa, d, c, state.prefill_rows, 0);
        living.* = null;
        errdefer if (saved) {
            living.* = t.sites;
        } else t.sites.deinit();
        t.seq = try state.Seq.init(d, c, seq_len, &.{}, e.stream, null);
        errdefer t.seq.deinit();
        t.buf = try e.b.sibling(d, c, e.nch);
        errdefer t.buf.deinit();
        inline for (state.seq_fields, 0..) |name, i| @field(t.buf, name) = t.seq.ptr[i];
        t.host = try sites_mod.Mapped.init(d, proj_at + c.layers * max_rows * sites_mod.candidates * 4);
        errdefer t.host.deinit();
        t.delta = .{ .m = try sites_mod.Mapped.init(d, max_seqs * 3 * c.layers * c.hidden * 4) };
        errdefer t.delta.m.deinit();
        t.push = .{ .m = try sites_mod.Mapped.init(d, max_seqs * 3 * c.layers * c.hidden * 4) };
        errdefer t.push.m.deinit();
        t.inputs = .{ .m = try sites_mod.Mapped.init(d, c.layers * pack_rows * c.hidden * 2) };
        errdefer t.inputs.m.deinit();
        t.residuals = .{ .m = try sites_mod.Mapped.init(d, (c.layers + 1) * pack_rows * c.hidden * 2) };
        errdefer t.residuals.m.deinit();
        t.kept = try gpa.alloc(Kept, c.layers);
        errdefer gpa.free(t.kept);
        t.raw = try gpa.alloc(Raw, c.layers);
        errdefer gpa.free(t.raw);
        errdefer for (t.owned.items) |*b| b.free();
        const t0 = std.Io.Clock.awake.now(io);
        try t.loadRaw(io, dir);
        try e.stream.synchronize();
        const t1 = std.Io.Clock.awake.now(io);
        var raw: usize = 0;
        for (t.owned.items) |b| raw += b.len;
        t.mem = .{ .buf = try cuda.DeviceBuffer.alloc(d, lay.layout(t, true)) };
        errdefer t.mem.buf.free();
        _ = lay.layout(t, false);
        try e.stream.synchronize();
        std.log.info("slide: the trainer's raw weights ({d:.1} GB) uploaded in {d:.1} s, its buffers ({d:.1} GB) in {d:.1} s", .{ @as(f64, @floatFromInt(raw)) / 1e9, seconds(t0, t1), @as(f64, @floatFromInt(t.mem.buf.len)) / 1e9, seconds(t1, std.Io.Clock.awake.now(io)) });
        return t;
    }

    fn seconds(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
        return @as(f64, @floatFromInt(b.toNanoseconds() - a.toNanoseconds())) / 1e9;
    }

    pub fn deinit(t: *Trainer) void {
        const gpa = t.gpa;
        if (t.tap) |*x| x.deinit();
        t.prefixes.deinit(gpa);
        t.sites.attach(&t.e.w, false);
        t.sites.deinit();
        t.mem.buf.free();
        for (t.owned.items) |*b| b.free();
        t.owned.deinit(gpa);
        gpa.free(t.raw);
        gpa.free(t.kept);
        t.host.deinit();
        inline for (.{ "delta", "push", "inputs", "residuals" }) |f| @field(t, f).m.deinit();
        t.buf.deinit();
        t.seq.deinit();
        gpa.destroy(t);
    }

    pub fn ops_(t: *const Trainer) ops.Ops {
        return .{ .k = &t.e.k.train, .s = t.e.stream };
    }

    /// The raw weights the backward's products read, and the LM head dequantized once.
    fn loadRaw(t: *Trainer, io: std.Io, dir: []const u8) !void {
        const c = t.c;
        var ck = try core.Checkpoint.openModel(t.gpa, io, dir);
        defer ck.close();
        // direct reads through the engine loader's slots: copies from the mapping crawl on GB10
        var src = try Source.init(t.gpa, t.e.ops());
        defer src.deinit();
        try src.add(&ck);
        var name: [192]u8 = undefined;
        for (0..c.layers) |i| {
            const pre = try std.fmt.bufPrint(&name, "backbone.layers.{d}.mixer.", .{i});
            var buf: [256]u8 = undefined;
            const at = struct {
                fn f(b: *[256]u8, p: []const u8, leaf: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "{s}{s}", .{ p, leaf });
                }
            }.f;
            t.raw[i] = switch (c.kinds[i]) {
                .mamba => .{ .mamba = .{ .in_proj = try t.mat(&ck, &src, try at(&buf, pre, "in_proj")), .out_proj = try t.mat(&ck, &src, try at(&buf, pre, "out_proj")) } },
                .attention => .{ .attention = .{ .q = try t.mat(&ck, &src, try at(&buf, pre, "q_proj")), .k = try t.mat(&ck, &src, try at(&buf, pre, "k_proj")), .v = try t.mat(&ck, &src, try at(&buf, pre, "v_proj")), .o = try t.mat(&ck, &src, try at(&buf, pre, "o_proj")) } },
                .moe => .{ .moe = .{ .up = try t.mat(&ck, &src, try at(&buf, pre, "shared_experts.up_proj")), .down = try t.mat(&ck, &src, try at(&buf, pre, "shared_experts.down_proj")), .fc1 = try t.four(&ck, &src, try at(&buf, pre, "switch_mlp.fc1")), .fc2 = try t.four(&ck, &src, try at(&buf, pre, "switch_mlp.fc2")) } },
            };
        }
        const head = try t.mat(&ck, &src, "lm_head");
        try src.flush();
        if (head.bf16) {
            t.head_t = head.w;
        } else {
            t.head_t = try t.device(head.n * head.k * 2);
            try t.ops_().dequant(.{ head.w, head.s, head.b }, t.head_t, head.n * head.k);
        }
    }

    fn device(t: *Trainer, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(t.e.ctx.d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try t.owned.append(t.gpa, b);
        return b.ptr;
    }

    fn upload(t: *Trainer, src: *Source, tensor: core.checkpoint.Tensor) !u64 {
        const ptr = try t.device(tensor.bytes.len);
        try src.upload(ptr, tensor.bytes);
        return ptr;
    }

    /// `base`.weight [n, k] as the checkpoint holds it: 4-bit with its scales and biases, or bf16.
    fn mat(t: *Trainer, ck: *core.Checkpoint, src: *Source, base: []const u8) !Mat {
        var buf: [256]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
        if (w.dtype == .bf16) return .{ .w = try t.upload(src, w), .bf16 = true, .n = w.dim(0), .k = w.dim(1) };
        if (w.dtype != .u32) return error.UnsupportedQuantization;
        const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}));
        const b = try ck.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}));
        return .{ .w = try t.upload(src, w), .s = try t.upload(src, s), .b = try t.upload(src, b), .n = w.dim(0), .k = w.dim(1) * 8 };
    }

    /// An expert table [E, n, k] as the checkpoint holds it, 4-bit.
    fn four(t: *Trainer, ck: *core.Checkpoint, src: *Source, base: []const u8) ![3]u64 {
        var buf: [256]u8 = undefined;
        var out: [3]u64 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            const x = try ck.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, part }));
            if (i == 0 and x.dtype != .u32) return error.UnsupportedQuantization;
            out[i] = try t.upload(src, x);
        }
        return out;
    }

    fn targets(t: *const Trainer) []u32 {
        return t.host.slice(u32, pack_rows);
    }

    fn weights(t: *const Trainer) []f32 {
        return t.host.slice(f32, 2 * pack_rows)[pack_rows..];
    }

    fn stats(t: *const Trainer) [][2]f32 {
        return t.host.slice([2]f32, 2 * pack_rows)[pack_rows..];
    }

    fn scales(t: *const Trainer) []f32 {
        return t.host.slice(f32, 5 * pack_rows)[4 * pack_rows ..];
    }

    pub const tapRows = taps.rows;

    /// Layer i's output projection as loaded: the matrix its learned change sits after.
    pub fn outProj(t: *const Trainer, i: usize) Mat {
        return switch (t.raw[i]) {
            .mamba => |m| m.out_proj,
            .attention => |a| a.o,
            .moe => |m| m.down,
        };
    }

    /// Each block's router score at the last step's decision row.
    pub fn score(t: *const Trainer) []f32 {
        return t.sites.score.slice(f32, lesson_route.max_slots * lesson_route.max_blocks)[lesson_route.trainer_slot * lesson_route.max_blocks ..][0..lesson_route.max_blocks];
    }

    /// Each row along layer l's candidate directions ([rows, candidates]), as the last project step left them.
    pub fn projected(t: *const Trainer, l: usize, rows: usize) []const f32 {
        const k = sites_mod.candidates;
        const all = std.mem.bytesAsSlice(f32, t.host.host.bytes[proj_at..][0 .. t.c.layers * max_rows * k * 4]);
        return @alignCast(all[l * max_rows * k ..][0 .. rows * k]);
    }

    /// MoE layer l's normed input (bf16 [D]) at the pass's `row`, what its shared expert and pocket read.
    pub fn normed(t: *const Trainer, l: usize, row: usize) []const u16 {
        const d = t.c.hidden;
        return t.inputs.slice(u16, t.c.layers * pack_rows * d)[(l * pack_rows + row) * d ..][0..d];
    }

    /// The residual (bf16 [D]) at the pass's `row` entering layer l (layers: after the last), as a target pass kept it.
    pub fn residual(t: *const Trainer, l: usize, row: usize) []const u16 {
        const d = t.c.hidden;
        return t.residuals.slice(u16, (t.c.layers + 1) * pack_rows * d)[(l * pack_rows + row) * d ..][0..d];
    }

    /// Where sequence s's vector for key i after layer l sits in `delta` and `push`, in floats.
    fn vec(t: *const Trainer, s: usize, i: usize, l: usize) usize {
        return ((s * 3 + i) * t.c.layers + l) * t.c.hidden;
    }

    /// The loss's gradient sequence s's target pass read at its key i's row after layer l.
    pub fn pushAt(t: *const Trainer, s: usize, i: usize, l: usize) []const f32 {
        return t.push.slice(f32, max_seqs * 3 * t.c.layers * t.c.hidden)[t.vec(s, i, l)..][0..t.c.hidden];
    }

    /// Sequence s's nudge for key i after layer l, which its target pass adds at that key's row.
    pub fn deltaAt(t: *const Trainer, s: usize, i: usize, l: usize) []f32 {
        return t.delta.slice(f32, max_seqs * 3 * t.c.layers * t.c.hidden)[t.vec(s, i, l)..][0..t.c.hidden];
    }

    pub fn savedAt(t: *const Trainer, l: usize) u64 {
        return t.saved + @as(u64, l) * pack_rows * t.c.hidden * 2;
    }

    /// Whether a pass may hold several sequences: no lesson block is open, so no row's lessons depend on its stream.
    pub fn packs(t: *const Trainer) bool {
        return t.sites.rank == 0;
    }

    /// Room for `n` prompt prefixes, each a pass's sequence's states after its first rows (none: their memory freed).
    pub fn prefixSlots(t: *Trainer, n: usize) !void {
        if (n == 0) t.prefixes.drop(t.gpa) else try t.prefixes.ensure(t.gpa, n);
    }

    /// Whether `sq` may run in `mode`: a prefix ends before its answer, decision and nudges, in a slot that exists.
    fn check(t: *const Trainer, sq: Seq, mode: Mode) !void {
        if (mode == .prime and (sq.skip > 0 or sq.slot >= t.prefixes.slots.items.len)) return error.BadPrefix;
        if (sq.skip == 0) return;
        if (sq.skip >= sq.start or sq.decide < sq.skip or sq.slot >= t.prefixes.slots.items.len or !t.packs()) return error.BadPrefix;
        for (sq.nudge.layers, sq.nudge.rows) |mask, row| if (mask != 0 and row < sq.skip) return error.BadPrefix;
    }

    /// Layer l's index among the layers of its kind (a prefix keeps each kind's states by it).
    pub fn kindIndex(t: *const Trainer, l: usize) usize {
        var n: usize = 0;
        for (t.c.kinds[0..l]) |k| n += @intFromBool(k == t.c.kinds[l]);
        return n;
    }

    /// Layer l's site input: what its change reads, rows `stride` apart, `width` wide (from the forward's scratch).
    fn siteInput(t: *const Trainer, l: usize) struct { x: u64, stride: usize } {
        const c = t.c;
        const b = &t.buf;
        return switch (c.kinds[l]) {
            .mamba => .{ .x = b.g, .stride = c.inner() },
            .attention => .{ .x = b.att, .stride = c.heads * c.head_dim },
            .moe => .{ .x = b.act + @as(u64, c.top_k) * c.expert_width * 2, .stride = c.slots() * c.expert_width },
        };
    }

    /// The mean loss of `ids`' answer from `start`, and what `mode` adds to it; `turn`: which rows carry which lessons.
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode, turn: lesson_route.Turn) !Result {
        var out: [1]Result = undefined;
        try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
        return out[0];
    }

    /// `seqs` side by side in one pass (several: keys, prefixes, losses or targets), each's mean loss into `out`.
    pub fn pass(t: *Trainer, seqs: []const Seq, mode: Mode, turn: lesson_route.Turn, out: []Result) !void {
        if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
        if (seqs.len > 1 and (!t.packs() or !(mode == .keys or mode == .loss or mode == .target or mode == .prime))) return error.BadPass;
        if (!reads(mode) and mode != .loss and mode != .target and t.sites.rank == 0) return error.NoOpenBlock;
        const c = t.c;
        const e = t.e;
        const o = e.ops();
        const k = t.ops_();
        const b = &t.buf;
        const D: usize = c.hidden;
        var rows: usize = 0;
        for (seqs, t.spans[0..seqs.len]) |sq, *sp| {
            if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
            try t.check(sq, mode);
            sp.* = .{ .at = rows, .rows = sq.ids.len - 1 - sq.skip, .skip = sq.skip, .slot = sq.slot };
            rows += sp.rows;
        }
        if (rows > t.pack) return error.PassTooLong;
        const spans = t.spans[0..seqs.len];
        t.seqs = seqs.len;
        t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
        try e.stream.synchronize(); // the latch, scores and host rows are host memory the last step's kernels may still read
        const ids = try t.gpa.alloc(u32, rows);
        defer t.gpa.free(ids);
        t.answered = train_mode.answers(seqs, spans, t.answers[0..seqs.len]);
        const answers = t.answers[0..seqs.len];
        for (seqs, spans, answers) |sq, sp, a| {
            @memcpy(ids[sp.at..][0..sp.rows], sq.ids[sq.skip..][0..sp.rows]);
            @memset(t.scales()[sp.at..][0..sp.rows], sq.scale);
            const share = 1 / @as(f32, @floatFromInt(a.rows));
            for (0..a.rows) |r| {
                t.targets()[a.at + r] = sq.ids[sq.start + r];
                t.weights()[a.at + r] = share;
            }
        }
        try o.upload(b.p_ids, std.mem.sliceAsBytes(ids));
        try t.seq.arena.?.buf.fill8(0, e.stream.handle);
        lesson_route.flag(t.sites.latch.slice(u32, lesson_route.max_slots * lesson_route.max_blocks), lesson_route.trainer_slot, turn.bits);
        @memset(t.score(), 0);
        var tt = turn;
        tt.decide = @intCast(spans[0].row(seqs[0].decide));
        t.route = lesson_route.trained(rows, tt);
        var f = fwd.Forward.init(c, &e.w, b, o, seq_len, e.nch, false);
        f.route = t.route;
        const scale: tfw.Scale = .{ .rows = t.host.dev + 4 * pack_rows * 4, .from = &t.from };
        var low: usize = 0;
        if (mode == .target) {
            var mask: u64 = 0;
            for (seqs) |sq| mask |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
            low = if (mask == 0) c.layers else @ctz(mask);
        }
        var w: fwd.Walk = .{ .rows = rows, .pos = 0 };
        try f.chunkBegin(&w);
        for (0..c.layers) |l| {
            try f.chunkPre(&w, l);
            try o.copy(t.savedAt(l), w.x, rows * D * 2);
            if (c.kinds[l] != .mamba) try o.copy(t.kept[l].x, b.y, rows * D * 2);
            try tfw.mixer(&f, &w, l, spans, .{ .store = &t.prefixes, .prime = mode == .prime });
            if (c.kinds[l] == .moe) {
                try tfw.experts(&f, e.w.blocks[l].moe, rows, scale, l);
                w.delta = .moe;
            } else try f.chunkPost(&w, l);
            const in = t.siteInput(l);
            const s = &t.sites.list[l];
            const one = seqs[0];
            switch (mode) {
                .loss, .prime => {},
                .grad, .learn => try t.keep(l, rows),
                .avoid, .seek => {
                    const y = if (mode == .avoid) s.avoid.dev else s.seek.dev;
                    const n: usize = if (mode == .avoid) sites_mod.avoid_dims else sites_mod.candidates;
                    const seed: u32 = if (mode == .avoid) 1 else 2;
                    const first = one.start - 1;
                    if (mode == .seek and t.sites.whole) {
                        // an edit of the weights is keyed on the question's rows, where its subject is read
                        try k.sketch(in.x, in.stride, y, one.decide + 1, s.in, n, t.sketched, seed, 1);
                    } else {
                        try k.sketch(in.x, in.stride, y, rows, s.in, n, t.sketched, seed, 1);
                        try k.sketch(in.x + @as(u64, first) * in.stride * 2, in.stride, y, 1, s.in, n, (1 << 30) + t.sketched + first, seed, @floatFromInt(rows - first));
                    }
                },
                .project => try k.project(in.x, in.stride, s.seek.dev, t.host.dev + proj_at + l * max_rows * sites_mod.candidates * 4, rows, s.in, sites_mod.candidates),
                // a pocket reads each MoE layer's normed input; a target pass also keeps what its backward reads
                .keys, .target => if (c.kinds[l] == .moe) {
                    try o.copy(t.inputs.dev() + l * pack_rows * D * 2, t.kept[l].x, rows * D * 2);
                    if (mode == .target) {
                        if (l >= low) try t.keep(l, rows);
                        // its nudges go into the shared expert's slot, which the next norm sums in
                        for (seqs, spans, 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| if (mask >> @intCast(l) & 1 != 0) {
                            try o.addRow(b.ymoe + (sp.row(row) * c.slots() + c.top_k) * D * 2, false, t.delta.dev() + t.vec(si, i, l) * 4, D);
                        };
                    }
                } else if (mode == .target and l >= low) try t.keep(l, rows),
            }
        }
        if (reads(mode)) {
            try e.stream.synchronize();
            if (mode == .avoid or mode == .seek) t.sketched += rows;
            for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
            return;
        }
        try f.norm(&w.x, w.delta, e.w.norm_f, rows, false);
        try o.copy(t.savedAt(c.layers), w.x, rows * D * 2);
        // only the rows that predict an answer reach the head: any other row's loss and gradient there is zero
        for (answers) |a| try o.copy(t.dhn + a.at * D * 2, b.y + a.first * D * 2, a.rows * D * 2);
        try o.prefillDense(t.dhn, e.w.head, t.logits, t.answered);
        const tg = t.host.dev;
        try k.softmax(t.logits, tg, tg + pack_rows * 4, tg + pack_rows * 8, t.answered, c.vocab);
        // a target pass's residuals for the host, the scale of what its pocket adds
        if (mode == .target) for (0..c.layers + 1) |l| try o.copy(t.residuals.dev() + l * pack_rows * D * 2, t.savedAt(l), rows * D * 2);
        if (mode != .loss and !(mode == .target and low == c.layers)) try back.backward(t, rows, mode == .learn, if (mode == .target) seqs else &.{}, low);
        try e.stream.synchronize();
        for (out[0..seqs.len], answers) |*r, a| {
            r.* = .{ .loss = 0, .recalled = true };
            for (t.stats()[a.at..][0..a.rows], t.weights()[a.at..][0..a.rows]) |st_, wgt| {
                r.loss += st_[0] * wgt;
                r.recalled = r.recalled and st_[1] > 0.5;
            }
        }
    }

    /// Layer l's activations into its slot from the forward's scratch, for its backward.
    fn keep(t: *Trainer, l: usize, rows: usize) !void {
        const c = t.c;
        const o = t.e.ops();
        const b = &t.buf;
        const kp = t.kept[l];
        switch (c.kinds[l]) {
            .mamba => {
                try o.copy(kp.proj, b.proj, rows * c.projDim() * 2);
                try o.copy(kp.ya, b.g, rows * c.inner() * 2);
            },
            .attention => try o.copy(kp.ya, b.att, rows * c.heads * c.head_dim * 2),
            .moe => {
                const S = c.slots();
                const W = c.expert_width;
                try o.copy(kp.act, b.act, rows * S * W * 2);
                try t.ops_().copyRows(b.act + @as(u64, c.top_k) * W * 2, S * W, 0, kp.upr, c.shared_width, c.shared_width, rows);
                try o.copy(kp.ys, b.ymoe, rows * S * c.hidden * 2);
                try o.copy(kp.ids, b.pick, rows * S * 4);
                try o.copy(kp.wts, b.wts, rows * S * 4);
            },
        }
    }
};
