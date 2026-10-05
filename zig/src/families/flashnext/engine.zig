//! Flash Next served from the replay engine: one reply at a time, its prompt through prompt chunks (the MTP head's
//! keys for every prompt row), then GPU-side rounds whose drafts the depth rule sets; tokens leave as rounds are read.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
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

/// Rounds the token ring holds (fz_accept writes round & 511).
const RING = 512;
/// Positions a reply keeps free past its last token: two rounds in flight.
pub const MARGIN = 2 * MAXR;

pub const Reason = enum { stop, length, cancelled };

pub const Result = struct { reason: Reason, rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, min_rows: u32 = 0 };

/// What a reply reports while it runs (called on the engine's thread).
pub const Out = struct {
    ctx: *anyopaque,
    /// The prompt is in the cache.
    prefilled: *const fn (ctx: *anyopaque) void,
    /// Tokens committed, in order; true ends the reply as a stop (a stop string matched).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    /// Checked between chunks and rounds.
    cancelled: *const fn (ctx: *anyopaque) bool,
};

pub const Engine = struct {
    gpa: Allocator,
    arena_state: std.heap.ArenaAllocator,
    r: *Run,
    m: *Model,
    pr: *Prompt,
    g_cs: mtl.Buffer,
    g_so: mtl.Buffer,
    o_cs: mtl.Buffer,
    o_so: mtl.Buffer,
    cins: [2]Buf,
    ring: mtl.Buffer,
    wids: Buf,
    rows_w: [MAXR + 1]Buf,
    mdims_w: [MAXR + 1]Buf,
    host: struct { rows: Buf, mdims: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, slots: [16]Slot },
    gpu_slots: [16]Slot,
    tsel: GSelect,
    msel: GSelect,

    /// The checkpoint in `model_dir` with the recorded kernels and packs in `dump_dir` (tools/zig/flashnext_dump.py).
    pub fn load(gpa: Allocator, model_dir: []const u8, dump_dir: []const u8) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
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
        r.event = try device.sharedEvent();
        try r.compile(dump_dir);
        r.sel = try Select.init(r, MAXR);
        const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{model_dir}, 0));
        const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
        var files: std.StringHashMapUnmanaged(void) = .empty;
        var wit = index.object.get("weight_map").?.object.iterator();
        while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
        var fit = files.keyIterator();
        while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ model_dir, name.* }, 0));
        try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{dump_dir}, 0));
        const ref_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{dump_dir}, 0));
        const ref = try std.json.parseFromSliceLeaky(std.json.Value, arena, ref_file.bytes[0..ref_file.size], .{});
        const ple_ref = ref.object.get("ple").?.object;
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
                L.norm = try r.loadf("L{d}.gdn.norm", .{i});
                for (0..2) |j| {
                    L.cs[j] = .{ .b = try r.buffer(MAXR * CS_ROW) };
                    L.so[j] = .{ .b = try r.buffer(MAXR * SO_ROW) };
                }
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
            .eos = jsonInt(ple_ref.get("eos").?),
            .mult = undefined,
            .sizes = undefined,
            .offsets = undefined,
        };
        for (0..3) |k| m.ple.mult[k] = jsonInt(ple_ref.get("multipliers").?.array.items[k]);
        for (0..16) |k| {
            m.ple.sizes[k] = jsonInt(ple_ref.get("sizes").?.array.items[k]);
            m.ple.offsets[k] = jsonInt(ple_ref.get("offsets").?.array.items[k]);
        }
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
            .ids8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
            .pos8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
            .nk8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
            .zero8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
            .ids81 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
            .scale = try f32Buf(r, @floatCast(ref.object.get("attention_scale").?.float)),
            .log2base = try f32Buf(r, 23.253496170043945),
            .ple_ids = try i32Buf(r, &(@as([16 * MAXR]i32, @splat(0)))),
            .ple_meta = try B.of(r, 39 * 8),
            .kvmeta = try i32Buf(r, &.{ 0, CAP, 1 }),
            .vocab = try i32Buf(r, &.{VOCAB}),
        };
        {
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
                .ids_n = (try r.entry("mtp.draft_ids")).len / 4,
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
                .logits = try B.of(r, 80000 * 2),
                .pick = try B.of(r, 16),
                .md1 = try i32Buf(r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
                .n_ids = undefined,
                .slots = undefined,
            };
            for (&m.mtp.slots) |*sl| sl.* = .{
                .rows = try i32Buf(r, &.{1}),
                .md = try i32Buf(r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
                .md4 = try i32Buf(r, &.{ 4, 16, 0, 0, 0, 0, 0, 0 }),
                .ids8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
                .pos8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
                .nk8 = try i32Buf(r, &(@as([8]i32, @splat(0)))),
                .kvmeta = try i32Buf(r, &.{ 0, CAP, 1 }),
                .n_add = try i32Buf(r, &.{0}),
            };
            m.mtp.n_ids = try i32Buf(r, &.{@intCast(m.mtp.ids_n)});
            if (m.mtp.ids_n * 2 > 80000 * 2) return error.DraftVocab;
        }
        try r.shapes.put(arena, "Kc_shape", (try i32Buf(r, &.{ 1, 2, CAP, 256 })).b);
        try r.shapes.put(arena, "IDS_shape", (try i32Buf(r, &.{ 8, 1 })).b);

        e.r = r;
        e.m = m;
        e.pr = try arena.create(Prompt);
        e.pr.* = try Prompt.init(r, dump_dir, r.xnew_header);
        try e.rounds();
        e.tsel = try GSelect.init(r, 0);
        e.msel = try GSelect.init(r, 0);
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
        e.cins = .{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
        r.ar = .{ .b = try r.buffer(4 * 256) };
        e.ring = try r.buffer(9 * 4 * RING);
        e.wids = .{ .b = try r.buffer(64) };
        m.mtp.mixsel = .{ .b = try r.buffer(D * 2) };
        m.mtp.hsel = .{ .b = try r.buffer(WIDE * 2) };
        for (1..MAXR + 1) |n| {
            e.rows_w[n] = try i32Buf(r, &.{@intCast(n)});
            e.mdims_w[n] = try i32Buf(r, &.{ @intCast(n), 16, 0, 0, 0, 0, 0, 0 });
        }
        e.host = .{ .rows = m.t.rows, .mdims = m.t.mdims, .pos8 = m.t.pos8, .nk8 = m.t.nk8, .kvmeta = m.t.kvmeta, .slots = m.mtp.slots };
        e.gpu_slots = m.mtp.slots;
        for (1..MAXR + 1) |n| { // the head absorbing a window of n rows: slot 7 + n
            const sl = &e.gpu_slots[7 + n];
            try m.mtpMeta(sl, n);
            sl.pos8 = .{ .b = r.ar.b, .off = 24 * 4 };
            sl.nk8 = .{ .b = r.ar.b, .off = 32 * 4 };
            sl.kvmeta = .{ .b = r.ar.b, .off = 40 * 4 };
        }
        for (1..MAXR - 1) |j| { // chained draft j: slot j
            try m.mtpMeta(&e.gpu_slots[j], 1);
            const b = (44 + (j - 1) * 20) * 4;
            e.gpu_slots[j].pos8 = .{ .b = r.ar.b, .off = b };
            e.gpu_slots[j].nk8 = .{ .b = r.ar.b, .off = b + 32 };
            e.gpu_slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 64 };
        }
    }

    /// Host-driven calls (prompt chunks, the head's first drafts) read the host's metadata buffers.
    fn hostMode(e: *Engine) void {
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
        m.t.pos8 = .{ .b = r.ar.b, .off = 4 * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = 12 * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = 20 * 4 };
        m.mtp.slots = e.gpu_slots;
    }

    fn statesCopy(e: *Engine) void { // the kept row of every DeltaNet layer's window output into its state
        const r = e.r;
        r.copyKept(.{ .b = e.o_so }, .{ .b = e.g_so }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
        r.copyKept(.{ .b = e.o_cs }, .{ .b = e.g_cs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
    }

    fn isEos(eos: []const u32, tok: u32) bool {
        return std.mem.indexOfScalar(u32, eos, tok) != null;
    }

    /// One greedy reply. `depth` fixes the drafts a round (0: no drafts, one token a round); null: the depth rule.
    pub fn generate(e: *Engine, prompt: []const u32, max_tokens: usize, eos: []const u32, depth: ?usize, out: Out) !Result {
        const r = e.r;
        const m = e.m;
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + max_tokens + MARGIN > CAP) return error.ContextFull;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        e.hostMode();
        defer e.hostMode();
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var pick: u32 = 0;
        var at: usize = 0;
        var last_n: usize = 1;
        while (at < prompt.len) {
            if (out.cancelled(out.ctx)) return .{ .reason = .cancelled };
            const n = @min(PMAX, prompt.len - at);
            pick = try e.pr.chunk(m, e.gpa, prompt[at .. at + n]);
            const k = if (at + n < prompt.len) n else n - 1;
            try e.pr.mtpKeys(m, at, prompt[at + 1 .. at + 1 + k], m.last);
            at += n;
            last_n = n;
        }
        out.prefilled(out.ctx);
        var res: Result = .{ .reason = .length };
        if (out.tokens(out.ctx, &.{pick}) or isEos(eos, pick)) return .{ .reason = .stop };
        if (max_tokens <= 1) return res;
        var emitted: usize = 1;
        const ar = r.ar.b.slice(i32, 256);
        if (m.state == 1) { // the state into the rounds' state buffers
            ar[0] = 1;
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(.serial);
            e.statesCopy();
            try m.finish(cb);
            m.state = 0;
            m.state_row = 0;
        }
        var rule: DepthRule = .{};
        const fixed = depth;
        const depth0 = fixed orelse rule.pick();
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
            for (&m.layers) |*L| if (!L.linear) try r.sel.?.catchUp(r, L, m.t.eps, m.t.log2base, m.pos / 4);
            try r.sel.?.catchUp(r, &m.mtp, m.t.eps, m.t.log2base, m.pos / 4);
            try m.finish(cb);
            e.tsel.sel.b.slice(i32, 64)[34] = @intCast(m.pos / 4);
            e.msel.sel.b.slice(i32, 64)[34] = @intCast(m.pos / 4);
        }
        @memset(ar, 0);
        ar[1] = T;
        for (0..8) |i| {
            ar[4 + i] = if (i < W0) T + @as(i32, @intCast(i)) else 0;
            ar[12 + i] = if (i < W0) T + @as(i32, @intCast(i)) + 1 else 0;
        }
        ar[20], ar[21], ar[22] = .{ T, CAP, @intCast(W0) };
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
        const rg = e.ring.slice(u32, 9 * RING);
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
                e.statesCopy();
                r.copyKept(e.cins[(round - 1) % 2], e.cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
                if (wr > 1) {
                    try m.mtpEncode(7 + wp, wp, m.t.picks, m.last, .{ .b = e.wids.b, .off = 4 });
                    for (1..wr - 1) |j| {
                        const streams = if (j == 1) m.mtp.hsel else Buf{ .b = m.mtp.h[1].b, .off = 0 };
                        try m.mtpEncode(j, 1, .{ .b = e.wids.b, .off = 4 * j }, streams, .{ .b = e.wids.b, .off = 4 * (j + 1) });
                    }
                }
            }
            m.ple.cin = e.cins[round % 2];
            m.pleIdsGpu(wr, e.wids);
            w_sum += wr;
            const ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            if (r.gsel) |*g| g.nb_ub = ub;
            if (m.mtp.gsel) |*g| g.nb_ub = ub;
            try m.windowEncode(wr, e.wids);
            const wn = (if (fixed) |d| d else rule.pick()) + 1;
            widths[(round + 1) % 4] = wn;
            const cfg = [4]u32{ @intCast(wr), @intCast(wn), CAP, 0 };
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ e.wids, m.t.picks, r.ar, .{ .b = e.ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
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
            const slot = (done % RING) * 9;
            const keep = rg[slot];
            const wd = widths[done % 4];
            if (fixed == null) rule.update(wd - 1, keep - 1);
            res.rounds += 1;
            res.drafted += wd - 1;
            res.accepted += keep - 1;
            res.min_rows = @min(res.min_rows, @as(u32, @intCast(wd)));
            done += 1;
            const got = rg[slot + 1 .. slot + 1 + keep];
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
            if (stop) {
                res.reason = .stop;
                break;
            }
            if (emitted >= max_tokens) break;
            if (out.cancelled(out.ctx)) {
                res.reason = .cancelled;
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

    /// A short reply on fixed tokens: the kernels' first launches and the weights' first reads happen here, not in
    /// the first request.
    pub fn warm(e: *Engine) !void {
        var toks: [96]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
        const Quiet = struct {
            fn prefilled(_: *anyopaque) void {}
            fn tokens(_: *anyopaque, _: []const u32) bool {
                return false;
            }
            fn cancelled(_: *anyopaque) bool {
                return false;
            }
        };
        var dummy: u8 = 0;
        _ = try e.generate(&toks, 24, &.{}, null, .{ .ctx = &dummy, .prefilled = Quiet.prefilled, .tokens = Quiet.tokens, .cancelled = Quiet.cancelled });
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.arena_state.deinit();
        gpa.destroy(e);
    }
};
