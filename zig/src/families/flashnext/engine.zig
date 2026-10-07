//! Flash Next served from the replay engine: one reply at a time, its prompt through prompt chunks (the MTP head's
//! keys for every prompt row), then GPU-side rounds whose drafts the depth rule sets; tokens leave as rounds are read.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const config = @import("config.zig");
const segments = @import("../../core/segments.zig");
const snapshot = @import("snapshot.zig");
const CallLog = @import("call_log.zig").CallLog;
const tp_settings = @import("tp_settings.zig");
const follow_mod = @import("follow.zig");
const marks_mod = @import("marks.zig");
const pack_io = @import("pack_io.zig");
const pack = @import("pack.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const CAP = fz.CAP;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const TOP = fz.TOP;
const PMAX = fz.PMAX;
const Buf = fz.Buf;
const Run = fz.Run;
const Model = fz.Model;
const Layer = fz.Layer;
const Prompt = fz.Prompt;
const GSelect = fz.GSelect;
const Select = fz.Select;
const Tmp = fz.Tmp;
const Slot = fz.Slot;
const DepthRule = fz.DepthRule;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const i32Buf = fz.i32Buf;
const f32Buf = fz.f32Buf;
const jsonInt = fz.jsonInt;

/// The embedded default draft vocabulary, written into the cache when a checkpoint is served with no dump.
const default_draft_vocab = @embedFile("draft_vocab_default.txt");

/// Rounds the token ring holds (fz_accept writes round & 511).
const RING = 512;
/// Positions a reply keeps free past its last token: two rounds in flight.
pub const MARGIN = 2 * MAXR;
/// Bytes a conversation's next kept state usually adds (about 4k tokens of reply and tool output): the pool readies that much more.
pub const NEXT_TURN = 4096 * 13 * 2304;

/// A kept state's name on both Macs: the hash of its tokens through the lookahead token (callers pass prompt[0 .. at + 1]).
pub fn keyOf(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}
/// Prompt chunks run as two staggered segments once each gets this many rows (core/segments.zig). Measured on the
/// M5 Ultra, two segments against one chunk: 2k rows a segment -7%, 3k level, 4k +1.6%, 8k +9%.
pub const SEG_MIN = 4096;
/// Speed-up mode splits a prompt chunk across the two Macs once each gets this many rows.
pub const PAIR_MIN: u32 = 256;

pub const Reason = enum { stop, length, cancelled };

pub const Result = struct {
    reason: Reason,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    min_rows: u32 = 0,
    copy_rounds: u64 = 0,
    copy_accepted: u64 = 0,
    copy_by_len: [9][2]u64 = @splat(.{ 0, 0 }), // copied rounds and drafts landed by match length
};

/// What a reply reports while it runs (called on the engine's thread).
pub const Out = struct {
    ctx: *anyopaque,
    /// The prompt is in the cache.
    prefilled: *const fn (ctx: *anyopaque) void,
    /// Tokens committed, in order; true ends the reply as a stop (a stop string matched).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    /// Checked between chunks and rounds.
    cancelled: *const fn (ctx: *anyopaque) bool,
    /// The prompt pass stands at one of generateFrom's marks (a chunk end): the prompt cache keeps its state here.
    marked: ?*const fn (ctx: *anyopaque, at: usize) void = null,
};

pub const Engine = struct {
    /// A mark a prompt call ran past: its DeltaNet states in m.marks' slot, and the n-gram tail rows and history there.
    pub const Passed = struct { at: usize, slot: usize, tail: Buf, hist: [2]i64 };

    gpa: Allocator,
    arena_state: std.heap.ArenaAllocator,
    r: *Run,
    m: *Model,
    pr: *Prompt,
    pr2: *Prompt, // the second staggered segment's buffers, selection and queue
    segments: bool = true, // prompt chunks as two staggered segments (false: one serial chunk at a time)
    g_cs: mtl.Buffer,
    g_so: mtl.Buffer,
    o_cs: mtl.Buffer,
    o_so: mtl.Buffer,
    cins: [2]Buf,
    ring: mtl.Buffer,
    hist: mtl.Buffer, // the token history the copy drafts search
    copy: bool = true, // copy drafts from the history when a long enough match exists
    copy_min: u32 = 3,
    copy_long: u32 = 6, // shorter matches copy only when the head's first draft agrees
    mark_taps: bool = true, // a prompt call writes the DeltaNet states at marks inside it (false: calls end at marks)
    pair_min: u32 = PAIR_MIN, // speed-up mode's pair chunks: rows each Mac takes at least (FZ_PAIR_MIN; rank 1 takes rank 0's)
    call_log: bool = false, // FZ_CALL_LOG: one line a prompt pass with each call's rows, wall and GPU time
    snap_pool: snapshot.Pool = .{}, // kept states' buffers, reused and readied ahead
    handoff_ns: i96 = 0, // speed-up rank 0: the last request's handoff to rank 1 and its resume answer
    passed: ?Passed = null, // the mark a prompt call ran past, while the prompt cache keeps the state there
    peer_drops: std.ArrayList(u64) = .empty, // speed-up rank 0: kept states rank 1 drops with the next request
    peer_kept: std.AutoHashMapUnmanaged(u64, *snapshot.State) = .empty, // speed-up rank 1: its halves of rank 0's kept states
    peer_budget: u64 = 0, // speed-up rank 1: the budget its kept states' buffers and free ones stay inside
    peer_held: u64 = 0, // speed-up rank 1: its kept states' buffers
    // while copies land worse than the head's drafts, only 8-token matches the head agrees with are copied
    wids: Buf,
    rows_w: [MAXR + 1]Buf,
    mdims_w: [MAXR + 1]Buf,
    host: struct { rows: Buf, mdims: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, slots: [2 * MAXR]Slot },
    gpu_slots: [2 * MAXR]Slot,
    tsel: GSelect,
    msel: GSelect,

    /// Load `model_dir` from a dump directory, or from the checked-in kernels when `dump_dir` is null.
    pub fn load(gpa: Allocator, io: std.Io, model_dir: []const u8, dump_dir: ?[]const u8) !*Engine {
        return loadWith(gpa, io, model_dir, dump_dir, null);
    }

    /// `load`, in speed-up mode when `speed_up` (or TF_FLASHNEXT_TP) names this Mac's settings: rank, MCDMA library and the link to the other Mac (tp.zig).
    pub fn loadWith(gpa: Allocator, io: std.Io, model_dir: []const u8, dump_dir: ?[]const u8, speed_up: ?[]const u8) !*Engine {
        const e = try gpa.create(Engine); // undefined memory: every defaulted field is set here
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.copy = true;
        e.copy_min = 3;
        e.copy_long = 6;
        e.segments = true;
        e.mark_taps = true;
        e.pair_min = tp_settings.pairMin(if (std.c.getenv("FZ_PAIR_MIN")) |v| std.mem.span(v) else null, PAIR_MIN); // rank 1 takes rank 0's
        e.call_log = std.c.getenv("FZ_CALL_LOG") != null;
        e.passed = null;
        e.snap_pool = .{};
        e.handoff_ns = 0;
        e.peer_drops = .empty;
        e.peer_kept = .empty;
        e.peer_budget = 0;
        e.peer_held = 0;
        e.arena_state = std.heap.ArenaAllocator.init(gpa);
        errdefer e.arena_state.deinit();
        const arena = e.arena_state.allocator();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const device = try mtl.Device.init();
        const r = try arena.create(Run);
        r.* = .{ .arena = arena, .device = device, .queue = try device.queue() };
        r.fused_xsum = true;
        r.serial = true;
        r.xnew = true;
        r.dense = true;
        r.hc_up = true;
        r.event = try device.sharedEvent();
        if (dump_dir) |d| try r.compile(d) else try r.compileChecked();
        r.sel = try Select.init(r, MAXR);
        r.lane_new = std.c.getenv("FZ_LANE") != null; // fz_lane and fz_gdn on the target (the recorded bits)
        if (std.c.getenv("FZ_GDN")) |v| {
            r.gdn_pipe = try fz.gdn_step.compile(r, false);
            if (v[0] == '2') r.gdn_kept = try fz.gdn_step.compile(r, true); // GPU-side rounds keep one state a layer
        }
        const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{model_dir}, 0));
        const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
        var files: std.StringHashMapUnmanaged(void) = .empty;
        var wit = index.object.get("weight_map").?.object.iterator();
        while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
        var fit = files.keyIterator();
        while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ model_dir, name.* }, 0));
        // A pack from another checkpoint is refused, and every mapping unmaps once the identity text exists.
        const identity = blk: {
            var maps: std.ArrayList(mtl.MappedFile) = .empty;
            errdefer for (maps.items) |*m| m.deinit();
            var mapped_shards: std.ArrayList(pack_io.MappedShard) = .empty;
            var sit = files.keyIterator();
            while (sit.next()) |name| {
                const shard = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ model_dir, name.* }, 0));
                try maps.append(arena, shard);
                try mapped_shards.append(arena, .{ .name = name.*, .size = shard.size, .bytes = shard.bytes[0..shard.size] });
            }
            const index_mapped = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{model_dir}, 0));
            try maps.append(arena, index_mapped);
            const id = try pack_io.identityFromMapped(arena, index_mapped.bytes[0..index_mapped.size], mapped_shards.items);
            for (maps.items) |*m| m.deinit();
            maps.clearRetainingCapacity();
            break :blk id;
        };
        const pack_dir = dump_dir orelse try std.fmt.allocPrintSentinel(arena, "{s}/zig-pack", .{model_dir}, 0);
        if (dump_dir == null) try ensurePacks(gpa, io, model_dir, pack_dir, identity);
        const pack_path = try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{pack_dir}, 0);
        {
            const pack_file = try mtl.MappedFile.open(pack_path);
            errdefer pack_file.deinit();
            try pack_io.checkSourceMapped(arena, pack_file.bytes[0..pack_file.size], identity, pack_path);
            pack_file.deinit();
        }
        try r.indexFile(pack_path);
        // the reference values: the dump's ref.json, or derived from the checkpoint's config (config.pleRef)
        var ple_values: config.PleRef = undefined;
        var attention_scale: f64 = undefined;
        if (dump_dir) |d| {
            const ref_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{d}, 0));
            const ref = try std.json.parseFromSliceLeaky(std.json.Value, arena, ref_file.bytes[0..ref_file.size], .{});
            const ple_ref = ref.object.get("ple").?.object;
            ple_values = .{
                .eos = jsonInt(ple_ref.get("eos").?),
                .multipliers = .{
                    jsonInt(ple_ref.get("multipliers").?.array.items[0]),
                    jsonInt(ple_ref.get("multipliers").?.array.items[1]),
                    jsonInt(ple_ref.get("multipliers").?.array.items[2]),
                },
                .sizes = undefined,
                .offsets = undefined,
            };
            for (0..16) |k| {
                ple_values.sizes[k] = jsonInt(ple_ref.get("sizes").?.array.items[k]);
                ple_values.offsets[k] = jsonInt(ple_ref.get("offsets").?.array.items[k]);
            }
            attention_scale = ref.object.get("attention_scale").?.float;
        } else {
            var cfg = try config.Config.read(arena, io, model_dir);
            ple_values = try config.pleRef(&cfg);
            attention_scale = 1.0 / std.math.sqrt(@as(f64, @floatFromInt(cfg.head_dim)));
        }
        const m = try arena.create(Model);
        m.* = .{ .r = r, .layers = undefined, .mix = undefined, .head = undefined, .embed = undefined, .ple = undefined, .t = undefined };
        for (0..LAYERS) |i| {
            const linear = i % 4 != 3;
            var L: Layer = .{ .ahc = try hcOf(r, "L{d}.ahc", .{i}), .mhc = try hcOf(r, "L{d}.mhc", .{i}), .linear = linear, .proj = undefined, .out = undefined, .router = try r.loadf("L{d}.moe.router", .{i}), .ex = undefined };
            if (linear) {
                L.proj = try laneOf(r, "L{d}.gdn.in", .{i});
                L.out = try laneOf(r, "L{d}.gdn.out", .{i});
                L.conv = try r.loadf("L{d}.gdn.conv", .{i});
                L.alog = try r.loadf("L{d}.gdn.alog", .{i});
                L.dt = try r.loadf("L{d}.gdn.dt", .{i});
                L.norm = try r.loadf("L{d}.gdn.norm", .{i}); // its state views (L.cs, L.so) come from rounds()
            } else {
                L.proj = try laneOf(r, "L{d}.att.proj", .{i});
                L.out = try laneOf(r, "L{d}.att.o", .{i});
                L.qn = try r.loadf("L{d}.att.qn", .{i});
                L.kn = try r.loadf("L{d}.att.kn", .{i});
                L.iqn = try r.loadf("L{d}.att.iqn", .{i});
                L.keys = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
                L.vals = .{ .b = try r.buffer(2 * CAP * 256 * 2) };
                L.raw = .{ .b = try r.buffer(CAP * 128 * 2) };
                L.pool = try r.loadf("L{d}.att.pool", .{i});
                L.pooled = .{ .b = try r.buffer(CAP / 4 * 128 * 2) };
            }
            const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
            for (projs, 0..) |proj, j| {
                for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                    L.ex[j * 3 + k] = try r.loadf("language_model.model.layers.{d}.mlp.{s}.{s}", .{ i, proj, suffix });
                }
            }
            if (r.xpack) try r.repack(&L.ex);
            m.layers[i] = L;
        }
        m.mix = try hcOf(r, "mix", .{});
        m.head = try laneOf(r, "head", .{});
        m.embed = .{ try r.load("language_model.model.embed_tokens.weight"), try r.load("language_model.model.embed_tokens.scales"), try r.load("language_model.model.embed_tokens.biases") };
        m.ple = .{
            .kv = try laneOf(r, "ple.kv", .{}),
            .ks = try r.load("ple.ks"),
            .qs = try r.load("ple.qs"),
            .cs = try r.load("ple.cs"),
            .conv = try r.load("ple.conv"),
            .starts = try r.load("ple.starts"),
            .tables = undefined,
            .cin = .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) },
            .hist = undefined,
            .eos = ple_values.eos,
            .mult = undefined,
            .sizes = undefined,
            .offsets = undefined,
        };
        for (0..3) |k| m.ple.mult[k] = ple_values.multipliers[k];
        for (0..16) |k| {
            m.ple.sizes[k] = ple_values.sizes[k];
            m.ple.offsets[k] = ple_values.offsets[k];
        }
        r.ngram = @import("index.zig").ngramSpelling(index.object.get("weight_map").?.object);
        for (0..GROUPS) |g| {
            m.ple.tables[3 * g + 0] = try r.group(16 * g, 16, "weight");
            m.ple.tables[3 * g + 1] = try r.group(16 * g, 16, "scales");
            m.ple.tables[3 * g + 2] = try r.group(16 * g, 16, "biases");
        }
        const B = struct {
            fn of(rr: *Run, n: usize) !Buf {
                return .{ .b = try rr.buffer(n) };
            }
        };
        m.t = .{
            .h = .{ try B.of(r, MAXR * WIDE * 2), try B.of(r, MAXR * WIDE * 2) },
            .ssp = try B.of(r, MAXR * 10 * 4 * 4),
            .part = try B.of(r, 10 * MAXR * 324 * 4),
            .mixed = try B.of(r, MAXR * D * 2),
            .inj_a = try B.of(r, MAXR * 4 * 2),
            .inj_m = try B.of(r, MAXR * 4 * 2),
            .xs = try B.of(r, 192 * 16 * 4),
            .p = try B.of(r, MAXR * 16480 * 2),
            .gout = try B.of(r, MAXR * 6144 * 2),
            .branch = try B.of(r, MAXR * D * 2),
            .lg = try B.of(r, MAXR * 513 * 4),
            .act = try B.of(r, MAXR * 11 * 640 * 2),
            .pick = try B.of(r, MAXR * 10 * 4),
            .wts = try B.of(r, MAXR * 10 * 4),
            .ydown = try B.of(r, MAXR * 11 * D * 2),
            .q = try B.of(r, MAXR * 24 * 256 * 2),
            .kout = try B.of(r, MAXR * 2 * 256 * 2),
            .iq = try B.of(r, MAXR * 4 * 128 * 2),
            .po = try B.of(r, MAXR * 24 * 16 * 256 * 4),
            .pm = try B.of(r, MAXR * 24 * 16 * 2 * 4),
            .aout = try B.of(r, MAXR * 6144 * 2),
            .emb = try B.of(r, MAXR * D * 2),
            .kvp = try B.of(r, MAXR * (WIDE + D) * 2),
            .gated = try B.of(r, MAXR * WIDE * 2),
            .hout = try B.of(r, MAXR * WIDE * 2),
            .logits = try B.of(r, MAXR * VOCAB * 2),
            .picks = try B.of(r, MAXR * 4),
            .rows = try i32Buf(r, &.{1}),
            .mdims = try i32Buf(r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .eps = try r.load("eps"),
            .ids8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
            .pos8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
            .nk8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
            .zero8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
            .ids81 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
            .scale = try f32Buf(r, @floatCast(attention_scale)),
            .log2base = try f32Buf(r, 23.253496170043945),
            .ple_ids = try i32Buf(r, &(@as([16 * MAXR]i32, @splat(0)))),
            .ple_meta = try B.of(r, 39 * 8),
            .kvmeta = try i32Buf(r, &.{ 0, CAP, 1 }),
            .vocab = try i32Buf(r, &.{VOCAB}),
        };
        {
            const ids_n = try fz.draftCount((try r.entry("mtp.draft_ids")).len);
            const ids = try r.load("mtp.draft_ids");
            var ex: [18]Buf = undefined;
            const projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
            for (projs, 0..) |proj, j| {
                for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                    ex[j * 3 + k] = try r.loadf("language_model.mtp.layers.0.mlp.{s}.{s}", .{ proj, suffix });
                }
            }
            if (r.xpack) try r.repack(&ex);
            m.mtp = .{
                .ahc = try hcOf(r, "mtp.ahc", .{}),
                .mhc = try hcOf(r, "mtp.mhc", .{}),
                .mix = try hcOf(r, "mtp.mix", .{}),
                .proj = try laneOf(r, "mtp.att.proj", .{}),
                .out = try laneOf(r, "mtp.att.o", .{}),
                .fce = try laneOf(r, "mtp.fce", .{}),
                .fch = try laneOf(r, "mtp.fch", .{}),
                .draft = try laneOf(r, "mtp.draft", .{}),
                .qn = try r.load("mtp.att.qn"),
                .kn = try r.load("mtp.att.kn"),
                .iqn = try r.load("mtp.att.iqn"),
                .enorm = try r.load("mtp.enorm.scale"),
                .hnorm = try r.load("mtp.hnorm.scale"),
                .router = try r.load("mtp.moe.router"),
                .ids = ids,
                .ids_n = ids_n,
                .ex = ex,
                .keys = .{ .b = try r.buffer(2 * CAP * 256 * 2) },
                .vals = .{ .b = try r.buffer(2 * CAP * 256 * 2) },
                .raw = .{ .b = try r.buffer(CAP * 128 * 2) },
                .pool = try r.load("mtp.att.pool"),
                .pooled = .{ .b = try r.buffer(CAP / 4 * 128 * 2) },
                .h = .{ try B.of(r, MAXR * WIDE * 2), try B.of(r, MAXR * WIDE * 2) },
                .emb = try B.of(r, MAXR * D * 2),
                .en = try B.of(r, MAXR * D * 2),
                .e = try B.of(r, MAXR * D * 2),
                .hn = try B.of(r, MAXR * WIDE * 2),
                .hs = try B.of(r, MAXR * WIDE * 2),
                .logits = try B.of(r, ids_n * 2),
                .pick = try B.of(r, 16),
                .md1 = try i32Buf(r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
                .n_ids = undefined,
                .slots = undefined,
            };
            for (&m.mtp.slots) |*sl| sl.* = .{
                .rows = try i32Buf(r, &.{1}),
                .md = try i32Buf(r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
                .md4 = try i32Buf(r, &.{ 4, 16, 0, 0, 0, 0, 0, 0 }),
                .ids8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
                .pos8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
                .nk8 = try i32Buf(r, &(@as([MAXR]i32, @splat(0)))),
                .kvmeta = try i32Buf(r, &.{ 0, CAP, 1 }),
                .n_add = try i32Buf(r, &.{0}),
            };
            m.mtp.n_ids = try i32Buf(r, &.{@intCast(m.mtp.ids_n)});
        }
        try r.shapes.put(arena, "Kc_shape", (try i32Buf(r, &.{ 1, 2, CAP, 256 })).b);
        try r.shapes.put(arena, "IDS_shape", (try i32Buf(r, &.{ MAXR, 1 })).b);

        e.r = r;
        e.m = m;
        e.pr = try arena.create(Prompt);
        e.pr.* = try Prompt.init(r, pack_dir, r.xnew_header);
        e.pr2 = try arena.create(Prompt);
        e.pr2.* = try e.pr.sibling();
        try e.rounds();
        e.tsel = try GSelect.init(r, 0);
        e.msel = try GSelect.init(r, 0);
        // speed-up mode: TP=2 with the peer the settings name (tp.zig); both ranks then run the same replies in lockstep
        const tp_path: ?[]const u8 = speed_up orelse if (std.c.getenv("TF_FLASHNEXT_TP")) |p| std.mem.span(p) else null;
        if (tp_path) |path| r.tp = try fz.Tp2.init(arena, r.device, path);
        return e;
    }

    /// The GPU-side rounds' buffers: DeltaNet states (kept, window rows), the PLE tail pair, the arena, the token ring,
    /// per-width row counts, and the MTP slots reading their positions from the arena.
    fn rounds(e: *Engine) !void {
        const r = e.r;
        const m = e.m;
        const n_lin: usize = 36;
        e.g_cs = try r.buffer(n_lin * CS_ROW);
        e.g_so = try r.buffer(n_lin * SO_ROW);
        e.o_cs = try r.buffer(n_lin * MAXR * CS_ROW);
        e.o_so = try r.buffer(n_lin * MAXR * SO_ROW);
        var gi: usize = 0;
        for (&m.layers) |*L| if (L.linear) {
            L.cs[0] = .{ .b = e.g_cs, .off = gi * CS_ROW };
            L.cs[1] = .{ .b = e.o_cs, .off = gi * MAXR * CS_ROW };
            L.so[0] = .{ .b = e.g_so, .off = gi * SO_ROW };
            L.so[1] = .{ .b = e.o_so, .off = gi * MAXR * SO_ROW };
            gi += 1;
        };
        if (r.gdn_kept != null) m.recs = .{ .{ .b = try r.buffer(n_lin * fz.gdn_step.RECORD) }, .{ .b = try r.buffer(n_lin * fz.gdn_step.RECORD) } };
        m.marks = try fz.Marks.init(r, n_lin);
        e.cins = .{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
        r.ar = .{ .b = try r.buffer(4 * fz.AR_WORDS) };
        e.ring = try r.buffer(fz.RING_WORDS * 4 * RING);
        e.hist = try r.buffer((CAP + 64) * 4);
        e.wids = .{ .b = try r.buffer(64) };
        for (1..MAXR + 1) |n| {
            e.rows_w[n] = try i32Buf(r, &.{@intCast(n)});
            e.mdims_w[n] = try i32Buf(r, &.{ @intCast(n), 16, 0, 0, 0, 0, 0, 0 });
        }
        e.host = .{ .rows = m.t.rows, .mdims = m.t.mdims, .pos8 = m.t.pos8, .nk8 = m.t.nk8, .kvmeta = m.t.kvmeta, .slots = m.mtp.slots };
        e.gpu_slots = m.mtp.slots;
        for (1..MAXR + 1) |n| { // the head absorbing a window of n rows: slot MAXR - 1 + n
            const sl = &e.gpu_slots[MAXR - 1 + n];
            try m.mtpMeta(sl, n);
            sl.pos8 = .{ .b = r.ar.b, .off = fz.AR_ABS_POS * 4 };
            sl.nk8 = .{ .b = r.ar.b, .off = fz.AR_ABS_NK * 4 };
            sl.kvmeta = .{ .b = r.ar.b, .off = fz.AR_ABS_KV * 4 };
        }
        for (1..MAXR - 1) |j| { // chained draft j: slot j
            try m.mtpMeta(&e.gpu_slots[j], 1);
            const b = (fz.AR_CHAIN + (j - 1) * fz.AR_CHAIN_STRIDE) * 4;
            e.gpu_slots[j].pos8 = .{ .b = r.ar.b, .off = b };
            e.gpu_slots[j].nk8 = .{ .b = r.ar.b, .off = b + 16 * 4 };
            e.gpu_slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 32 * 4 };
        }
    }

    /// Host-driven calls (prompt chunks, the head's first drafts) read the host's metadata buffers.
    pub fn hostMode(e: *Engine) void {
        const m = e.m;
        e.r.gpu_round = false;
        e.r.gsel = null;
        m.mtp.gsel = null;
        m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta = .{ e.host.rows, e.host.mdims, e.host.pos8, e.host.nk8, e.host.kvmeta };
        m.mtp.slots = e.host.slots;
        m.ple.cin = e.cins[0];
    }

    /// GPU-side rounds read positions and widths the previous round's verdict wrote into the arena.
    fn gpuMode(e: *Engine) void {
        const m = e.m;
        const r = e.r;
        r.gpu_round = true;
        m.t.pos8 = .{ .b = r.ar.b, .off = fz.AR_POS * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = fz.AR_NK * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = fz.AR_KV * 4 };
        m.mtp.slots = e.gpu_slots;
    }

    fn statesCopy(e: *Engine, states: bool) void { // the kept row of every DeltaNet layer's window output into its state
        const r = e.r;
        if (states) r.copyKept(.{ .b = e.o_so }, .{ .b = e.g_so }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
        r.copyKept(.{ .b = e.o_cs }, .{ .b = e.g_cs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
    }

    /// A prompt call of up to `left` rows: one chunk, as two staggered segments once each gets SEG_MIN rows.
    fn chunkCall(e: *const Engine, left: usize) segments.Call {
        return if (e.segments) segments.next(left, e.pr.step, SEG_MIN) else .{ .rows = @min(e.pr.step, left), .parts = 1 };
    }

    fn isEos(eos: []const u32, tok: u32) bool {
        return std.mem.indexOfScalar(u32, eos, tok) != null;
    }

    /// One greedy reply. `depth` fixes the drafts a round (0: no drafts, one token a round); null: the depth rule.
    pub fn generate(e: *Engine, prompt: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, out: Out) !Result {
        return e.generateFrom(prompt, 0, &.{}, max_tokens, eos, depth, out);
    }

    /// `generate` after snapshot.restore put `from` tokens in the model, the pass cut at each mark for out.marked.
    pub fn generateFrom(e: *Engine, prompt: []const u32, from: usize, marks: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, out: Out) !Result {
        const r = e.r;
        const m = e.m;
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + max_tokens + MARGIN > CAP) return error.ContextFull;
        if (from >= prompt.len or (from > 0 and m.pos != from)) return error.NotResumed;
        for (marks, 0..) |mk, i| if (mk <= from or mk >= prompt.len or (i > 0 and mk <= marks[i - 1])) return error.Marks;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        if (r.tp) |tp| if (tp.rank == 0) { // served speed-up mode: rank 1 runs the same request (follow)
            var head: [16]u32 = @splat(0);
            head[0] = @intCast(prompt.len);
            head[1] = @intCast(@min(max_tokens, std.math.maxInt(u32)));
            head[2] = @intCast(@min(eos.len, 8));
            head[3] = if (depth) |d| @intCast(d) else std.math.maxInt(u32);
            for (eos[0..head[2]], 0..) |t, i| head[4 + i] = t;
            head[12] = @intCast(from); // rank 1 restores its own state at the same tokens
            head[13] = @intCast(marks.len); // the marks, then the dropped states' keys, follow the prompt
            head[14] = @intCast(e.peer_drops.items.len);
            head[15] = @intCast(e.pair_min); // rank 1 splits its calls the same way
            const words = try e.gpa.alloc(u32, prompt.len + marks.len + 2 * e.peer_drops.items.len);
            defer e.gpa.free(words);
            @memcpy(words[0..prompt.len], prompt);
            @memcpy(words[prompt.len..][0..marks.len], marks);
            for (e.peer_drops.items, 0..) |k, i| words[prompt.len + marks.len + 2 * i ..][0..2].* = .{ @truncate(k), @truncate(k >> 32) };
            const t0 = std.c.mach_absolute_time();
            defer e.handoff_ns = @intCast((std.c.mach_absolute_time() - t0) * 125 / 3); // mach ticks are 125/3 ns on Apple silicon
            try tp.sendRequest(std.mem.asBytes(&head), words);
            e.peer_drops.clearRetainingCapacity();
            if (!try tp.waitAck()) return error.PeerNotResumed;
        };
        e.hostMode();
        defer e.hostMode();
        if (from == 0) m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var pick: u32 = 0;
        var at: usize = from;
        var last_n: usize = 1;
        var dr: marks_mod.Driver = .{ .marks = marks };
        const ps = [2]*Prompt{ e.pr, e.pr2 };
        var calls: CallLog = .{};
        while (at < prompt.len) {
            if (try e.agree(out.cancelled(out.ctx))) return .{ .reason = .cancelled };
            const t_call = std.c.mach_absolute_time();
            const g_call = m.gpu_seconds;
            if (r.tp) |tp| { // speed-up mode: the call's rows split across the two Macs, running on past up to MARKS marks
                const pair_taps = e.mark_taps and m.marks != null;
                var call = segments.next(if (pair_taps) prompt.len - at else dr.toNext(at, prompt.len), e.pr.step, e.pair_min);
                var passed: [fz.MARKS]u32 = undefined;
                var n_passed: usize = 0;
                if (call.parts == 2 and pair_taps) {
                    const in_ = dr.inside(at, call.rows, &passed);
                    n_passed = in_.n;
                    if (in_.cut) |cut| call = segments.next(cut, e.pr.step, e.pair_min);
                }
                if (call.parts == 2) {
                    const hist0 = m.ple.hist;
                    pick = try Prompt.chunkPair(e.pr, m, e.gpa, prompt[at .. at + call.rows], tp, passed[0..n_passed]);
                    var span: [2][2]usize = undefined; // each Mac's segment: first position, rows with an MTP key
                    for (0..2) |k| {
                        const s = at + segments.start(call.rows, 2, k);
                        const n = segments.rows(call.rows, 2, k);
                        span[k] = .{ s, if (s + n < prompt.len) n else n - 1 };
                    }
                    const mine = span[tp.rank];
                    try e.pr.mtpKeys(m, mine[0], prompt[mine[0] + 1 .. mine[0] + 1 + mine[1]], e.pr.last);
                    try e.pr.pairMtp(m, tp, mine, span[1 - tp.rank]);
                    const last_call = at + call.rows == prompt.len;
                    try dr.cross(e, passed[0..n_passed], segments.rows(call.rows, 2, 0), segments.start(call.rows, 2, tp.rank), last_call);
                    last_n = 1; // m.last points at the last row
                    for (passed[0..n_passed], 0..) |row, j| { // both Macs hold each passed mark's states once it crosses
                        dr.passed(e, out, .{ .at = at + row, .slot = j, .tail = m.marks.?.tail_at(j), .hist = marks_mod.histAt(hist0, prompt, at, row) }, last_call);
                    }
                    calls.note(call.rows, 2, t_call, g_call, m.gpu_seconds);
                    at += call.rows;
                    dr.after(out, at, n_passed); // a pair chunk can end on a mark too
                    continue;
                }
            }
            // a call runs on past marks and writes the DeltaNet states there (up to MARKS); more marks end it at the next
            const taps = e.mark_taps and m.marks != null;
            var c: segments.Call = e.chunkCall(if (taps) prompt.len - at else dr.toNext(at, prompt.len));
            var inside: [fz.MARKS]u32 = undefined;
            var n_in: usize = 0;
            if (taps) {
                const in_ = dr.inside(at, c.rows, &inside);
                n_in = in_.n;
                if (in_.cut) |cut| c = e.chunkCall(cut);
            }
            const hist0 = m.ple.hist; // the two tokens before the call: the n-gram history at a passed mark follows from them
            pick = try Prompt.chunkMarked(ps[0..c.parts], m, e.gpa, prompt[at .. at + c.rows], inside[0..n_in]);
            for (ps[0..c.parts], 0..) |p, k| { // the MTP head's keys for each segment's rows, from its streams
                const s = at + segments.start(c.rows, c.parts, k);
                const n = segments.rows(c.rows, c.parts, k);
                const nexts = if (s + n < prompt.len) n else n - 1;
                try p.mtpKeys(m, s, prompt[s + 1 .. s + 1 + nexts], p.last);
                last_n = n;
            }
            for (inside[0..n_in], 0..) |row, j| { // the passed marks: the cache keeps each from where the call left it
                const seg = marks_mod.markSeg(ps[0..c.parts], c.rows, row);
                dr.passed(e, out, .{ .at = at + row, .slot = j, .tail = .{ .b = seg.p.b.cin.b, .off = seg.p.b.cin.off + seg.row * WIDE * 2 }, .hist = marks_mod.histAt(hist0, prompt, at, row) }, at + c.rows == prompt.len);
            }
            calls.note(c.rows, c.parts, t_call, g_call, m.gpu_seconds);
            at += c.rows;
            dr.after(out, at, n_in);
        }
        out.prefilled(out.ctx);
        var res: Result = .{ .reason = .length };
        const first_stop = out.tokens(out.ctx, &.{pick}) or isEos(eos, pick);
        try dr.flush(e, out); // after the first token: the slots, rows and tail stay put till the rounds
        if (e.call_log) calls.log(if (r.tp) |tp| tp.rank else 0);
        if (try e.agree(first_stop)) return .{ .reason = .stop };
        if (max_tokens <= 1) return res;
        var emitted: usize = 1;
        const ar = r.ar.b.slice(i32, fz.AR_WORDS);
        if (m.state == 1) { // the state into the rounds' state buffers
            ar[0] = 1;
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(.serial);
            e.statesCopy(true);
            try m.finish(cb);
            m.state = 0;
            m.state_row = 0;
        }
        var rule: DepthRule = .{ .pair = r.tp != null };
        var copy_rate: f64 = 0.6; // drafts landed over offered: copied rounds, the head's rounds
        var head_rate: f64 = 0.6;
        const fixed = depth;
        const depth0 = fixed orelse rule.pick();
        var next_depth: usize = depth0; // the rule's drafts for the next round to encode (rank 0's in speed-up mode)
        const w = e.wids.b.slice(u32, 16);
        w[0] = pick;
        if (depth0 > 0) { // the head: the last prompt row with the first token, then its chain
            m.mtp.pos = prompt.len - 1;
            w[1] = try m.mtpRun(&.{pick}, .{ .b = m.last.b, .off = m.last.off + (last_n - 1) * WIDE * 2 });
            for (2..depth0 + 1) |j| w[j] = try m.mtpChain(w[j - 1]);
        }
        const W0 = depth0 + 1;
        const T: i32 = @intCast(m.pos);
        const long = prompt.len + max_tokens + 8 > 4 * TOP;
        if (long) { // selection on the GPU: every complete block the target and the head hold, pooled
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(.serial);
            try r.sel.?.catchUp(r, m, m.pos / 4);
            try m.finish(cb);
            e.tsel.sel.b.slice(i32, 128)[fz.SEL_POOLED] = @intCast(m.pos / 4);
            e.msel.sel.b.slice(i32, 128)[fz.SEL_POOLED] = @intCast(m.pos / 4);
        }
        @memset(ar, 0);
        ar[1] = T;
        const hs = e.hist.slice(u32, CAP + 64); // the history: the prompt and the first token
        @memcpy(hs[0..prompt.len], prompt);
        hs[prompt.len] = pick;
        ar[fz.AR_HLEN] = @intCast(prompt.len + 1);
        for (0..MAXR) |i| {
            ar[fz.AR_POS + i] = if (i < W0) T + @as(i32, @intCast(i)) else 0;
            ar[fz.AR_NK + i] = if (i < W0) T + @as(i32, @intCast(i)) + 1 else 0;
        }
        ar[fz.AR_KV], ar[fz.AR_KV + 1], ar[fz.AR_KV + 2] = .{ T, CAP, @intCast(W0) };
        const pm = m.t.ple_meta.b.slice(i64, 39);
        pm[0], pm[1], pm[2], pm[3] = .{ m.ple.hist[0], m.ple.hist[1], m.ple.eos, 0 };
        for (0..3) |k| pm[4 + k] = m.ple.mult[k];
        for (0..16) |k| {
            pm[7 + k] = m.ple.sizes[k];
            pm[23 + k] = m.ple.offsets[k];
        }
        e.gpuMode();
        if (long) {
            r.gsel = e.tsel;
            m.mtp.gsel = e.msel;
        }
        const base = r.event_value;
        var cbs: [4]?mtl.CommandBuffer = .{ null, null, null, null };
        var widths: [4]usize = undefined;
        widths[0] = W0;
        var round: usize = 0;
        var done: usize = 0;
        var w_sum: usize = 0;
        res.min_rows = @intCast(W0);
        const rg = e.ring.slice(u32, fz.RING_WORDS * RING);
        var failed: ?anyerror = null;
        while (true) {
            const wr = widths[round % 4];
            const cb = r.queue.commandBuffer();
            if (round > 0) cb.waitFor(r.event, base + round);
            r.enc = cb.compute(.serial);
            m.t.rows = e.rows_w[wr];
            m.t.mdims = e.mdims_w[wr];
            if (round > 0) {
                const wp = widths[(round - 1) % 4];
                e.statesCopy(r.gdn_kept == null); // kept states replay the kept rows themselves
                r.copyKept(e.cins[(round - 1) % 2], e.cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
                if (wr > 1) {
                    try m.mtpEncode(MAXR - 1 + wp, wp, m.t.picks, m.last, .{ .b = e.wids.b, .off = 4 });
                    for (1..wr - 1) |j| try m.mtpEncode(j, 1, .{ .b = e.wids.b, .off = 4 * j }, m.mtp.h[1], .{ .b = e.wids.b, .off = 4 * (j + 1) });
                    if (e.copy) { // copied drafts replace the head's when the history repeats its last tokens
                        r.enc.setPipeline(r.lookup_pipe);
                        r.enc.setBuffer(e.hist, 0, 0);
                        r.enc.setBuffer(r.ar.b, r.ar.off, 1);
                        r.enc.setBuffer(e.wids.b, e.wids.off, 2);
                        const lenient = copy_rate + 0.05 >= head_rate;
                        const lc = [4]u32{ @intCast(wr), if (lenient) e.copy_min else 4, if (lenient) e.copy_long else 9, if (lenient) 1 else 2 };
                        r.enc.setBytes(std.mem.asBytes(&lc), 3);
                        r.enc.dispatchThreads(mtl.Size.of(1024, 1, 1), mtl.Size.of(1024, 1, 1));
                    }
                }
            }
            m.ple.cin = e.cins[round % 2];
            m.rec_slot = round % 2;
            m.pleIdsGpu(wr, e.wids);
            w_sum += wr;
            const ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            if (r.gsel) |*g| g.nb_ub = ub;
            if (m.mtp.gsel) |*g| g.nb_ub = ub;
            try m.windowEncode(wr, e.wids);
            // the widest windows while copies land whole; otherwise the head's rule
            const copying = e.copy and copy_rate + 0.05 >= head_rate and copy_rate > 0.9;
            const wn = (if (fixed) |d| d else if (copying) MAXR - 1 else next_depth) + 1;
            widths[(round + 1) % 4] = wn;
            const cfg = [4]u32{ @intCast(wr), @intCast(wn), CAP, 0 };
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ e.wids, m.t.picks, r.ar, .{ .b = e.ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
            r.enc.setBuffer(e.hist, 0, 6);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            r.enc.end();
            cb.signal(r.event, base + round + 1);
            cb.commit();
            cbs[round % 4] = cb;
            round += 1;
            if (round < 2) continue;
            const prev = cbs[done % 4].?;
            prev.wait();
            cbs[done % 4] = null;
            if (prev.failure()) |msg| {
                std.log.err("command buffer failed: {s}", .{msg});
                failed = error.GpuFailed;
                break;
            }
            if (r.tp) |tp| if (tp.failed.load(.acquire)) { // the link failed: the host drained the round, its tokens are wrong
                failed = error.TpLinkFailed;
                break;
            };
            const slot = (done % RING) * fz.RING_WORDS;
            const keep = rg[slot];
            const wd = widths[done % 4];
            if (wd > 1) {
                const frac = @as(f64, @floatFromInt(keep - 1)) / @as(f64, @floatFromInt(wd - 1));
                if (rg[slot + 1] != 0) copy_rate = 0.8 * copy_rate + 0.2 * frac else head_rate = 0.8 * head_rate + 0.2 * frac;
                if (rg[slot + 1] == 0 and wd == MAXR) copy_rate *= 0.8; // a full window that found no copy: copies stopped landing
            }
            if (rg[slot + 1] != 0) {
                res.copy_rounds += 1;
                res.copy_accepted += keep - 1;
                const n = @min(rg[slot + 2], 8);
                res.copy_by_len[n][0] += 1;
                res.copy_by_len[n][1] += keep - 1;
            }
            if (fixed == null and wd > 1 and rg[slot + 1] == 0) rule.update(wd - 1, keep - 1);
            res.rounds += 1;
            res.drafted += wd - 1;
            res.accepted += keep - 1;
            res.min_rows = @min(res.min_rows, @as(u32, @intCast(wd)));
            done += 1;
            const got = rg[slot + 3 .. slot + 3 + keep];
            var take: usize = 0;
            var stop = false;
            while (take < got.len and emitted + take < max_tokens) {
                take += 1;
                if (isEos(eos, got[take - 1])) {
                    stop = true;
                    break;
                }
            }
            if (take > 0 and out.tokens(out.ctx, got[0..take])) stop = true;
            emitted += take;
            const cancel = !stop and emitted < max_tokens and out.cancelled(out.ctx);
            const ag = e.agreeRound(stop or emitted >= max_tokens or cancel, rule.pick()) catch |err| {
                failed = err;
                break;
            };
            next_depth = ag.depth;
            const quit = ag.quit;
            if (quit) {
                res.reason = if (stop) .stop else if (emitted >= max_tokens) .length else .cancelled;
                break;
            }
        }
        for (&cbs) |*c| if (c.*) |cb| {
            cb.wait();
            c.* = null;
        };
        r.event_value = base + round + 1;
        if (failed) |err| return err;
        return res;
    }

    /// Speed-up mode: whether to stop at a step both ranks take (rank 0 decides, rank 1 follows); alone, `quit`.
    fn agree(e: *Engine, quit: bool) !bool {
        return if (e.r.tp) |tp| (try tp.agree(quit, 0)).quit else quit;
    }

    /// A round's stop decision and the next round's drafts: rank 0's for both Macs (each Mac's own timings differ).
    fn agreeRound(e: *Engine, quit: bool, depth: usize) !struct { quit: bool, depth: usize } {
        const tp = e.r.tp orelse return .{ .quit = quit, .depth = depth };
        const a = try tp.agree(quit, @intCast(depth));
        return .{ .quit = a.quit, .depth = a.depth };
    }

    /// Speed-up mode's rank 1: it runs rank 0's requests (follow) and serves none of its own.
    pub fn followsPeer(e: *const Engine) bool {
        return if (e.r.tp) |tp| tp.rank == 1 else false;
    }

    /// Served speed-up mode, rank 1: end `follow`'s wait for rank 0's next request (before deinit).
    pub fn stopFollowing(e: *Engine) void {
        if (e.r.tp) |tp| tp.quitting.store(true, .release);
    }

    /// Served speed-up mode, rank 1: rank 0's requests in step until its empty request (follow.zig).
    pub fn follow(e: *Engine) !void {
        return follow_mod.run(e);
    }

    /// Rank 1: rank 0's next request, its reply into `out` (null: hashed and logged, this Mac's states kept); null once rank 0 closes (follow.zig).
    pub fn followWith(e: *Engine, out: ?Out) !?Result {
        return follow_mod.one(e, out);
    }

    const Quiet = struct {
        fn prefilled(_: *anyopaque) void {}
        fn tokens(_: *anyopaque, _: []const u32) bool {
            return false;
        }
        fn cancelled(_: *anyopaque) bool {
            return false;
        }
    };

    /// A short reply on fixed tokens: the kernels' first launches and the weights' first reads happen here, not in
    /// the first request. In speed-up mode rank 1 runs rank 0's warm-up request instead.
    pub fn warm(e: *Engine) !void {
        if (e.r.tp) |tp| if (tp.rank == 1) {
            _ = try follow_mod.one(e, null);
            return;
        };
        var toks: [96]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
        var dummy: u8 = 0;
        _ = try e.generate(&toks, 24, &.{}, null, .{ .ctx = &dummy, .prefilled = Quiet.prefilled, .tokens = Quiet.tokens, .cancelled = Quiet.cancelled });
    }

    pub fn deinit(e: *Engine) void {
        if (e.r.tp) |tp| { // rank 1's follow ends on rank 0's empty request; then the link closes, before the memory
            if (tp.rank == 0) {
                const none: [12]u32 = @splat(0);
                tp.sendRequest(std.mem.asBytes(&none), &.{}) catch {};
            }
            tp.deinit();
        }
        const gpa = e.gpa;
        var kept = e.peer_kept.valueIterator();
        while (kept.next()) |st| snapshot.drop(gpa, st.*);
        e.peer_kept.deinit(gpa);
        e.snap_pool.deinit();
        e.peer_drops.deinit(gpa);
        e.arena_state.deinit();
        gpa.destroy(e);
    }
};

/// The pack cache beside the checkpoint is built once and rebuilt when its source no longer matches.
fn ensurePacks(gpa: Allocator, io: std.Io, model_dir: []const u8, cache_dir: []const u8, identity: []const u8) !void {
    Io.Dir.cwd().createDir(io, cache_dir, .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    if (pack_io.packsReady(gpa, io, cache_dir, identity)) |ready| {
        if (ready) return;
    } else |_| {}
    const vocab_path = try std.fmt.allocPrintSentinel(gpa, "{s}/draft_vocab.txt", .{cache_dir}, 0);
    defer gpa.free(vocab_path);
    {
        var file = try Io.Dir.cwd().createFile(io, vocab_path, .{});
        defer file.close(io);
        var wbuf: [64 << 10]u8 = undefined;
        var fw = file.writerStreaming(io, &wbuf);
        try fw.interface.writeAll(default_draft_vocab);
        try fw.interface.flush();
    }
    _ = try pack.build(gpa, io, model_dir, cache_dir, vocab_path);
}

test "load sets no DeltaNet state view before rounds() points them at the round buffers" {
    const src = @embedFile("engine.zig");
    const at = std.mem.indexOf(u8, src, "pub fn loadWith(").?;
    const body = src[at..std.mem.indexOfPos(u8, src, at, "try e.rounds();").?];
    try std.testing.expect(std.mem.indexOf(u8, body, ".cs[") == null and std.mem.indexOf(u8, body, ".so[") == null);
}
