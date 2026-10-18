//! Flash Next engine loading and resource construction.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const fz = @import("replay.zig");
const config = @import("config.zig");
const segments = @import("../../core/segments.zig");
const tp_settings = @import("tp_settings.zig");
const pack_io = @import("pack_io.zig");
const pack = @import("pack.zig");
const pack_cache = @import("pack_cache.zig");
const load_admission = @import("load_admission.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const Buf = fz.Buf;
const Run = fz.Run;
const Model = fz.Model;
const Layer = fz.Layer;
const Prompt = fz.Prompt;
const GSelect = fz.GSelect;
const Select = fz.Select;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const i32Buf = fz.i32Buf;
const f32Buf = fz.f32Buf;
const jsonInt = fz.jsonInt;
const expert_projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
const PAIR_MIN = @import("engine.zig").PAIR_MIN;
const Engine = @import("engine.zig").Engine;
const load = Engine.load;
const rounds = Engine.rounds;
const deinit = Engine.deinit;

pub fn loadWith(gpa: Allocator, io: std.Io, model_dir: []const u8, dump_dir: ?[]const u8, speed_up: ?[]const u8) !*Engine {
    return loadConfigured(gpa, io, model_dir, dump_dir, speed_up, .{});
}

pub fn loadConfigured(gpa: Allocator, io: std.Io, model_dir: []const u8, dump_dir: ?[]const u8, speed_up: ?[]const u8, window: fz.WindowSettings) !*Engine {
    try window.validate(speed_up != null or platform.c.getenv("TF_FLASHNEXT_TP") != null);
    try load_admission.check(gpa, io, model_dir);
    const memory = @import("flashnext_memory");
    const weights = try memory.weightBytes(@import("core").checkpoint_host, gpa, io, model_dir);
    const inventory = try memory.weights_plan.read(@import("core").checkpoint_host.Checkpoint, @import("core").quant.mlx, gpa, io, model_dir);
    const home = if (platform.c.getenv("HOME")) |v| std.mem.span(v) else "";
    const route = try @import("pack_route.zig").plan(gpa, io, model_dir, dump_dir, home);
    defer route.deinit(gpa);
    const paired = speed_up != null or platform.c.getenv("TF_FLASHNEXT_TP") != null;
    const requested = try std.math.add(u64, window.startup_extra, if (paired) fz.Tp2.startupBytes() else 0);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    errdefer device.deinit();
    const existing = device.allocated();
    const extra = try std.math.add(u64, requested, existing);
    const startup = try memory.budgetLive(weights, window.capacity, inventory, extra, existing);
    const ram = std.process.totalSystemMemory() catch null;
    const limit_text: ?[]const u8 = if (platform.c.getenv("FLASHNEXT_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const load_limit = memory.admitPhase(device, weights, window.capacity, ram, limit_text, extra, inventory, existing, route.needs_build) catch |err| {
        if (err == error.FlashNextMemoryShortfall) {
            const need = startup.forPhase(route.needs_build);
            const limit = try memory.ceiling(ram, device.maxWorkingSet(), limit_text);
            const gib: f64 = 1 << 30;
            std.log.err("flash next: startup plans {d:.2} GiB ({s} phase) for raw/packed and vision weights, applicable phase scratch, both prompt planes/selectors, KV/state, existing device allocations and requested sessions/cache and a {d} GiB runtime reserve; the RAM/GPU-bounded load ceiling {d:.2} GiB is short by {d:.2} GiB (GPU working set {d:.2} GiB); FLASHNEXT_LOAD_LIMIT_GB may raise the 70% policy within those bounds; {s}", .{ @as(f64, @floatFromInt(need)) / gib, if (route.needs_build and startup.building > startup.steady) @as([]const u8, "pack construction") else "load/warm-up", memory.margin >> 30, @as(f64, @floatFromInt(limit)) / gib, @as(f64, @floatFromInt(need - limit)) / gib, @as(f64, @floatFromInt(device.maxWorkingSet())) / gib, memory.options });
        }
        return err;
    };
    // Pack construction has no engine buffers or numerical pipelines live beside its host work.
    const built_dir = if (route.needs_build) try pack_cache.ensure(gpa, io, model_dir, route.root.?, route.identity, PackBuilder{}) else null;
    defer if (built_dir) |dir| gpa.free(dir);
    const pack_dir = route.dir orelse built_dir.?;
    const e = try gpa.create(Engine); // undefined memory: every defaulted field is set here
    errdefer gpa.destroy(e);
    e.gpa = gpa;
    e.copy = true;
    e.copy_min = 3;
    e.copy_long = 6;
    e.segments = true;
    e.mark_taps = true;
    e.pair_min = tp_settings.pairMin(if (platform.c.getenv("FZ_PAIR_MIN")) |v| std.mem.span(v) else null, PAIR_MIN); // rank 1 takes rank 0's
    e.call_log = platform.c.getenv("FZ_CALL_LOG") != null;
    e.passed = null;
    e.snap_pool = .{ .load_limit = load_limit };
    e.handoff_ns = 0;
    e.peer_drops = .empty;
    e.peer_kept = .empty;
    e.peer_budget = 0;
    e.peer_held = 0;
    e.identity_hash = 0;
    e.arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer e.arena_state.deinit();
    const arena = e.arena_state.allocator();

    const r = try arena.create(Run);
    r.* = .{ .arena = arena, .device = device, .queue = try device.queue(), .window = window, .load_limit = load_limit };
    r.fused_xsum = true;
    r.serial = true;
    r.xnew = true;
    r.dense = true;
    r.hc_up = true;
    r.event = try device.sharedEvent();
    if (dump_dir) |d| try r.compile(d) else try r.compileChecked();
    r.sel = try Select.init(r, MAXR);
    r.lane_new = platform.c.getenv("FZ_LANE") != null; // core lane and fz_gdn on the target (the recorded bits)
    if (platform.c.getenv("FZ_GDN")) |v| {
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
    const identity = route.identity;
    e.identity_hash = std.hash.Wyhash.hash(0x6964, identity);
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
    var cfg = try config.Config.read(arena, io, model_dir); // read once: the reference values and the widths
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
        ple_values = try config.pleRef(&cfg);
        attention_scale = 1.0 / std.math.sqrt(@as(f64, @floatFromInt(cfg.head_dim)));
    }
    const m = try arena.create(Model);
    m.* = .{ .r = r, .cap = r.window.capacity, .layers = undefined, .mix = undefined, .head = undefined, .embed = undefined, .ple = undefined, .t = undefined };
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
            L.keys = .{ .b = try r.buffer(2 * r.window.capacity * 256 * 2) };
            L.vals = .{ .b = try r.buffer(2 * r.window.capacity * 256 * 2) };
            L.raw = .{ .b = try r.buffer(r.window.capacity * 128 * 2) };
            L.pool = try r.loadf("L{d}.att.pool", .{i});
            L.pooled = .{ .b = try r.buffer(r.window.capacity / 4 * 128 * 2) };
        }
        for (expert_projs, 0..) |proj, j| {
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
    m.ple.scale = try fz.ple_scale.of(r, index.object.get("weight_map").?.object);
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
        .log2base = try fz.ropeBuf(r, 23.253496170043945, r.window.factor),
        .ple_ids = try i32Buf(r, &(@as([16 * MAXR]i32, @splat(0)))),
        .ple_meta = try B.of(r, 39 * 8),
        .kvmeta = try i32Buf(r, &.{ 0, @intCast(r.window.capacity), 1 }),
        .vocab = try i32Buf(r, &.{VOCAB}),
    };
    {
        const ids_n = try fz.draftCount((try r.entry("mtp.draft_ids")).len);
        const ids = try r.load("mtp.draft_ids");
        const mtp_root = @import("index.zig").mtpRoot(index.object.get("weight_map").?.object);
        var ex: [18]Buf = undefined;
        for (expert_projs, 0..) |proj, j| {
            for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |suffix, k| {
                ex[j * 3 + k] = try r.loadf("{s}.layers.0.mlp.{s}.{s}", .{ mtp_root, proj, suffix });
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
            .keys = .{ .b = try r.buffer(2 * r.window.capacity * 256 * 2) },
            .vals = .{ .b = try r.buffer(2 * r.window.capacity * 256 * 2) },
            .raw = .{ .b = try r.buffer(r.window.capacity * 128 * 2) },
            .pool = try r.load("mtp.att.pool"),
            .pooled = .{ .b = try r.buffer(r.window.capacity / 4 * 128 * 2) },
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
            .kvmeta = try i32Buf(r, &.{ 0, @intCast(r.window.capacity), 1 }),
            .n_add = try i32Buf(r, &.{0}),
        };
        m.mtp.n_ids = try i32Buf(r, &.{@intCast(m.mtp.ids_n)});
    }
    try r.shapes.put(arena, "Kc_shape", (try i32Buf(r, &.{ 1, 2, @intCast(r.window.capacity), 256 })).b);
    try r.shapes.put(arena, "IDS_shape", (try i32Buf(r, &.{ MAXR, 1 })).b);

    e.r = r;
    e.m = m;
    e.pr = try arena.create(Prompt);
    e.pr.* = try Prompt.init(r, pack_dir, r.xnew_header);
    const uniform = try @import("engine_widths.zig").apply(arena, &cfg, index.object.get("weight_map").?.object, e.pr, m);
    e.pr2 = try arena.create(Prompt);
    e.pr2.* = try e.pr.sibling();
    try e.rounds();
    e.tsel = try GSelect.init(r, 0);
    e.msel = try GSelect.init(r, 0);
    // speed-up mode: TP=2 with the peer the settings name (tp.zig); both ranks then run the same replies in...
    const tp_path: ?[]const u8 = speed_up orelse if (platform.c.getenv("TF_FLASHNEXT_TP")) |p| std.mem.span(p) else null;
    if (tp_path != null and !uniform) return error.SpeedUpNeeds6Bit32; // TP splits the 6/32 kernels only
    if (tp_path) |path| r.tp = try fz.Tp2.init(arena, r.device, path);
    if (r.tp) |t| try r.held.appendSlice(arena, &.{ t.wbuf, t.one, t.pick_tmp }); // every pair round uses them
    if (r.tp == null) try fz.rows_load.attach(
        e.pr,
        m,
        @import("index.zig").mtpRoot(index.object.get("weight_map").?.object),
    );
    return e;
}

pub const PackBuilder = struct {
    pub fn build(_: PackBuilder, gpa: Allocator, io: Io, model_dir: []const u8, out_dir: []const u8, vocab_path: []const u8) !void {
        _ = try pack.build(gpa, io, model_dir, out_dir, vocab_path);
    }
};
