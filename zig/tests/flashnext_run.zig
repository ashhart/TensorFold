//! Flash Next 1-8 row windows on Python's recorded kernels: greedy tokens, drafted rows with rollback, then timings.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;

const opts = fz.opts;
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
const Buf = fz.Buf;
const Entry = fz.Entry;
const Variant = fz.Variant;
const Site = fz.Site;
const readAll = fz.readAll;
const Run = fz.Run;
const TOP = fz.TOP;
const KW = fz.KW;
const Select = fz.Select;
const GSelect = fz.GSelect;
const Hc = fz.Hc;
const Layer = fz.Layer;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const f32Buf = fz.f32Buf;
const i32Buf = fz.i32Buf;
const Mtp = fz.Mtp;
const Model = fz.Model;
const DepthRule = fz.DepthRule;
const copyDrafts = fz.copyDrafts;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const jsonInt = fz.jsonInt;
const compileCheck = @import("flashnext_modes_a.zig").compileCheck;
const mmaPeak = @import("flashnext_modes_a.zig").mmaPeak;
const engine = @import("flashnext_modes_a.zig").engine;
const profile = @import("flashnext_modes_b.zig").profile;
const dual = @import("flashnext_modes_b.zig").dual;
const hcCheck = @import("flashnext_modes_b.zig").hcCheck;
const gCheck = @import("flashnext_modes_b.zig").gCheck;
const xtime = @import("flashnext_modes_b.zig").xtime;
const prefill = @import("flashnext_modes_b.zig").prefill;
const qmm6 = @import("flashnext_modes_c.zig").qmm6;
const overlap = @import("flashnext_modes_c.zig").overlap;
const runDefault = @import("flashnext_default.zig").runDefault;

fn armName(seg_ab: bool, arm: usize) []const u8 {
    if (seg_ab) return if (arm == 1) "segments on " else "segments off";
    if (platform.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(platform.c.getenv("FZ_AB").?), "fz")) return if (arm == 1) "fz kernels" else "recorded  ";
    return if (arm == 1) "copy on " else "copy off";
}

/// FZ_PROMPT_N: only the prompt's first n tokens (all when unset).
fn promptCut() !usize {
    const v = platform.c.getenv("FZ_PROMPT_N") orelse return std.math.maxInt(usize);
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

/// FZ_CATCH_CHECK: catchUp's layers in one command buffer against each layer alone, then one shared slot (the control).
fn catchCheck(r: *Run, arena: std.mem.Allocator) !void {
    const upto = 1300;
    const m = try arena.create(Model);
    m.r = r;
    m.gpu_seconds = 0;
    m.t.eps = try f32Buf(r, 1e-6);
    m.t.log2base = try f32Buf(r, std.math.log2(10_000_000.0));
    const raw: Buf = .{ .b = try r.buffer(4 * upto * 256) };
    var rng = std.Random.DefaultPrng.init(7);
    for (raw.b.slice(u16, 4 * upto * 128)) |*v| v.* = @truncate(@as(u32, @bitCast(rng.random().float(f32) * 2 - 1)) >> 16);
    const w: Buf = .{ .b = try r.buffer(128 * 4) };
    for (w.b.slice(f32, 128), 0..) |*v, i| v.* = 1 + @as(f32, @floatFromInt(i)) / 256;
    var firsts: [fz.CATCH]usize = undefined;
    for (0..fz.CATCH) |k| firsts[k] = if (k + 1 < fz.CATCH) 600 + 31 * k else 0; // the head last, from block 0
    var outs: [3][fz.CATCH]Buf = undefined; // arms: each layer alone (the reference), one command buffer, one shared slot
    for (&outs) |*o| for (o) |*b| {
        b.* = .{ .b = try r.buffer(upto * 256) };
    };
    var differ: [3]usize = .{ 0, 0, 0 };
    for (0..3) |arm| {
        var k: usize = 0;
        for (&m.layers, 0..) |*L, i| {
            L.linear = i % 4 != 3;
            if (L.linear) continue;
            L.raw, L.pool, L.pooled, L.pooled_n = .{ raw, w, outs[arm][k], firsts[k] };
            k += 1;
        }
        m.mtp.raw, m.mtp.pool, m.mtp.pooled, m.mtp.pooled_n = .{ raw, w, outs[arm][k], firsts[k] };
        var cb = r.queue.commandBuffer();
        r.enc = cb.compute(if (r.serial) .serial else .concurrent);
        switch (arm) {
            0 => for (0..fz.CATCH) |j| {
                if (j + 1 < fz.CATCH) try r.sel.?.catchLayer(r, &m.layers[4 * j + 3], m.t.eps, m.t.log2base, upto, 0) else try r.sel.?.catchLayer(r, &m.mtp, m.t.eps, m.t.log2base, upto, 0);
                try m.finish(cb);
                cb = r.queue.commandBuffer();
                r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            },
            1 => try r.sel.?.catchUp(r, m, upto),
            else => {
                for (&m.layers) |*L| if (!L.linear) try r.sel.?.catchLayer(r, L, m.t.eps, m.t.log2base, upto, 0);
                try r.sel.?.catchLayer(r, &m.mtp, m.t.eps, m.t.log2base, upto, 0);
            },
        }
        try m.finish(cb);
        if (arm == 0) continue;
        for (0..fz.CATCH) |j| {
            const span = outs[arm][j].b.contents()[firsts[j] * 256 .. upto * 256];
            if (!std.mem.eql(u8, span, outs[0][j].b.contents()[firsts[j] * 256 .. upto * 256])) differ[arm] += 1;
        }
    }
    std.debug.print("catch-up: {d} layers in one command buffer, {d} differ from each layer alone; the control's one shared slot: {d} differ\n", .{ fz.CATCH, differ[1], differ[2] });
    if (differ[1] != 0 or differ[2] == 0) return error.CatchUpCheck;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-flashnext-run MODEL_DIR DUMP_DIR\n", .{});
        std.process.exit(2);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const device = try mtl.Device.init();
    if (platform.c.getenv("FZ_COMPILE_CHECK") != null) try compileCheck(device, gpa, arena);
    if (platform.c.getenv("FZ_OPORDER") != null) if (platform.c.getenv("FZ_OPORDER") != null) return @import("flashnext_bench.zig").opOrder(device); // no model loaded
    if (platform.c.getenv("FZ_MMA_PEAK") != null) {
        try mmaPeak(device, arena);
        return;
    }
    if (platform.c.getenv("FZ_ENGINE") != null) {
        try engine(init, gpa, arena, args);
        return;
    }

    // the replay below reads every expert, embedding and table at 6/32: other widths are the engine's (Engine.load)
    if (!try fz.engine_widths.uniformDir(arena, init.io, args[1])) return error.HarnessNeeds6Bit32;
    var r = Run{ .arena = arena, .device = device, .queue = try device.queue() };
    r.fused_xsum = platform.c.getenv("FZ_FUSED_XSUM") != null;
    r.serial = platform.c.getenv("FZ_SERIAL") != null;
    r.xnew = platform.c.getenv("FZ_XNEW") != null;
    r.split = platform.c.getenv("FZ_SPLIT") != null;
    r.gdn_step = platform.c.getenv("FZ_GDN_STEP") != null;
    r.hc_mma = platform.c.getenv("FZ_HC_MMA") != null;
    r.hc_up = platform.c.getenv("FZ_HCCHECK") != null or if (platform.c.getenv("FZ_HC_UP")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    r.event = try device.event();
    r.dense = r.xnew and platform.c.getenv("FZ_DENSE") != null;
    r.dense_target = r.dense and platform.c.getenv("FZ_DENSE_TARGET") != null;
    r.xpack = r.xnew and platform.c.getenv("FZ_XPACK") != null;
    r.grouped = r.xnew and !r.xpack and platform.c.getenv("FZ_GROUPED") != null;
    r.xfused = r.xnew and !r.xpack and !r.grouped and platform.c.getenv("FZ_XFUSED") != null;
    r.xsx = r.xnew and platform.c.getenv("FZ_XSX") != null;
    r.copy = platform.c.getenv("FZ_COPY") != null;
    r.prefetch = !r.serial and platform.c.getenv("FZ_PREFETCH") != null;
    const t0 = mtl.clock.seconds();
    try r.compile(args[2]);
    r.sel = try Select.init(&r, MAXR);
    if (platform.c.getenv("FZ_CATCH_CHECK") != null) return catchCheck(&r, arena);
    r.lane_new = r.xnew and platform.c.getenv("FZ_LANE") != null; // core lane and fz_gdn on the target (same bits)
    if (r.xnew and platform.c.getenv("FZ_GDN") != null) r.gdn_pipe = try fz.gdn_step.compile(&r, false);
    const t1 = mtl.clock.seconds();

    const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{args[1]}, 0));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
    var files: std.StringHashMapUnmanaged(void) = .empty;
    var wit = index.object.get("weight_map").?.object.iterator();
    while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    var fit = files.keyIterator();
    while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ args[1], name.* }, 0));
    try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{args[2]}, 0));

    const own_prompt = platform.c.getenv("FZ_REF") != null; // another prompt: passes against Python's records are skipped
    const ref_path = if (platform.c.getenv("FZ_REF")) |p| try std.fmt.allocPrintSentinel(arena, "{s}", .{std.mem.span(p)}, 0) else try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{args[2]}, 0);
    const ref_file = try mtl.MappedFile.open(ref_path);
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, arena, ref_file.bytes[0..ref_file.size], .{});
    const ple_ref = ref.object.get("ple").?.object;

    const m = try arena.create(Model);
    m.* = .{ .r = &r, .layers = undefined, .mix = undefined, .head = undefined, .embed = undefined, .ple = undefined, .t = undefined };
    for (0..LAYERS) |i| {
        const linear = i % 4 != 3;
        var L: Layer = .{ .ahc = try hcOf(&r, "L{d}.ahc", .{i}), .mhc = try hcOf(&r, "L{d}.mhc", .{i}), .linear = linear, .proj = undefined, .out = undefined, .router = try r.loadf("L{d}.moe.router", .{i}), .ex = undefined };
        if (linear) {
            L.proj = try laneOf(&r, "L{d}.gdn.in", .{i});
            L.out = try laneOf(&r, "L{d}.gdn.out", .{i});
            L.conv = try r.loadf("L{d}.gdn.conv", .{i});
            L.alog = try r.loadf("L{d}.gdn.alog", .{i});
            L.dt = try r.loadf("L{d}.gdn.dt", .{i});
            L.norm = try r.loadf("L{d}.gdn.norm", .{i});
            for (0..2) |j| {
                L.cs[j] = .{ .b = try r.buffer(MAXR * CS_ROW) };
                L.so[j] = .{ .b = try r.buffer(MAXR * SO_ROW) };
            }
        } else {
            L.proj = try laneOf(&r, "L{d}.att.proj", .{i});
            L.out = try laneOf(&r, "L{d}.att.o", .{i});
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
    m.mix = try hcOf(&r, "mix", .{});
    m.head = try laneOf(&r, "head", .{});
    m.embed = .{ try r.load("language_model.model.embed_tokens.weight"), try r.load("language_model.model.embed_tokens.scales"), try r.load("language_model.model.embed_tokens.biases") };
    m.ple = .{
        .kv = try laneOf(&r, "ple.kv", .{}),
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
        .h = .{ try B.of(&r, MAXR * WIDE * 2), try B.of(&r, MAXR * WIDE * 2) },
        .ssp = try B.of(&r, MAXR * 10 * 4 * 4),
        .part = try B.of(&r, 10 * MAXR * 324 * 4),
        .mixed = try B.of(&r, MAXR * D * 2),
        .inj_a = try B.of(&r, MAXR * 4 * 2),
        .inj_m = try B.of(&r, MAXR * 4 * 2),
        .xs = try B.of(&r, 192 * 16 * 4),
        .p = try B.of(&r, MAXR * 16480 * 2),
        .gout = try B.of(&r, MAXR * 6144 * 2),
        .branch = try B.of(&r, MAXR * D * 2),
        .lg = try B.of(&r, MAXR * 513 * 4),
        .act = try B.of(&r, MAXR * 11 * 640 * 2),
        .pick = try B.of(&r, MAXR * 10 * 4),
        .wts = try B.of(&r, MAXR * 10 * 4),
        .ydown = try B.of(&r, MAXR * 11 * D * 2),
        .q = try B.of(&r, MAXR * 24 * 256 * 2),
        .kout = try B.of(&r, MAXR * 2 * 256 * 2),
        .iq = try B.of(&r, MAXR * 4 * 128 * 2),
        .po = try B.of(&r, MAXR * 24 * 16 * 256 * 4),
        .pm = try B.of(&r, MAXR * 24 * 16 * 2 * 4),
        .aout = try B.of(&r, MAXR * 6144 * 2),
        .emb = try B.of(&r, MAXR * D * 2),
        .kvp = try B.of(&r, MAXR * (WIDE + D) * 2),
        .gated = try B.of(&r, MAXR * WIDE * 2),
        .hout = try B.of(&r, MAXR * WIDE * 2),
        .logits = try B.of(&r, MAXR * VOCAB * 2),
        .picks = try B.of(&r, MAXR * 4),
        .rows = try i32Buf(&r, &.{1}),
        .mdims = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
        .eps = try r.load("eps"),
        .ids8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
        .pos8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
        .nk8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
        .zero8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
        .ids81 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
        .scale = try f32Buf(&r, @floatCast(ref.object.get("attention_scale").?.float)),
        .log2base = try f32Buf(&r, 23.253496170043945),
        .ple_ids = try i32Buf(&r, &(@as([16 * MAXR]i32, @splat(0)))),
        .ple_meta = try B.of(&r, 39 * 8),
        .kvmeta = try i32Buf(&r, &.{ 0, CAP, 1 }),
        .vocab = try i32Buf(&r, &.{VOCAB}),
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
            .ahc = try hcOf(&r, "mtp.ahc", .{}),
            .mhc = try hcOf(&r, "mtp.mhc", .{}),
            .mix = try hcOf(&r, "mtp.mix", .{}),
            .proj = try laneOf(&r, "mtp.att.proj", .{}),
            .out = try laneOf(&r, "mtp.att.o", .{}),
            .fce = try laneOf(&r, "mtp.fce", .{}),
            .fch = try laneOf(&r, "mtp.fch", .{}),
            .draft = try laneOf(&r, "mtp.draft", .{}),
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
            .h = .{ try B.of(&r, MAXR * WIDE * 2), try B.of(&r, MAXR * WIDE * 2) },
            .emb = try B.of(&r, MAXR * D * 2),
            .en = try B.of(&r, MAXR * D * 2),
            .e = try B.of(&r, MAXR * D * 2),
            .hn = try B.of(&r, MAXR * WIDE * 2),
            .hs = try B.of(&r, MAXR * WIDE * 2),
            .logits = try B.of(&r, ids_n * 2),
            .pick = try B.of(&r, 16),
            .md1 = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .n_ids = undefined,
            .slots = undefined,
        };
        for (&m.mtp.slots) |*sl| sl.* = .{
            .rows = try i32Buf(&r, &.{1}),
            .md = try i32Buf(&r, &.{ 1, 16, 0, 0, 0, 0, 0, 0 }),
            .md4 = try i32Buf(&r, &.{ 4, 16, 0, 0, 0, 0, 0, 0 }),
            .ids8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
            .pos8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
            .nk8 = try i32Buf(&r, &(@as([MAXR]i32, @splat(0)))),
            .kvmeta = try i32Buf(&r, &.{ 0, CAP, 1 }),
            .n_add = try i32Buf(&r, &.{0}),
        };
        m.mtp.n_ids = try i32Buf(&r, &.{@intCast(m.mtp.ids_n)});
    }
    try r.shapes.put(arena, "Kc_shape", (try i32Buf(&r, &.{ 1, 2, CAP, 256 })).b);
    try r.shapes.put(arena, "IDS_shape", (try i32Buf(&r, &.{ MAXR, 1 })).b);
    const t2 = mtl.clock.seconds();
    std.debug.print("compiled in {d:.2} s, loaded {d:.1} GB in {d:.1} s\n", .{ t1 - t0, @as(f64, @floatFromInt(r.loaded)) / 1e9, t2 - t1 });

    if (platform.c.getenv("FZ_DBENCH") != null) { // dense classes and DeltaNet timed by rows (flashnext_bench.zig)
        var toks: [MAXR]u32 = undefined;
        for (0..MAXR) |i| toks[i] = @intCast(ref.object.get("prompt").?.array.items[i].integer);
        return @import("flashnext_bench.zig").run(&r, m, &toks, gpa, arena);
    }
    if (platform.c.getenv("FZ_PROFILE") != null) {
        try profile(&r, m, ref, init.io, arena);
        return;
    }
    if (platform.c.getenv("FZ_DUAL") != null) {
        try dual(&r, m, ref, device);
        return;
    }
    if (platform.c.getenv("FZ_HCCHECK") != null) {
        try hcCheck(&r, m, ref, gpa);
        return;
    }
    if (platform.c.getenv("FZ_GCHECK") != null) {
        try gCheck(&r, m, ref, gpa);
        return;
    }
    if (platform.c.getenv("FZ_XTIME") != null) {
        try xtime(&r, m, ref);
        return;
    }
    if (platform.c.getenv("FZ_PREFILL") != null) {
        try prefill(&r, m, ref, gpa, args);
        return;
    }
    if (platform.c.getenv("FZ_QMM6") != null) {
        try qmm6(&r, m, device, arena);
        return;
    }
    if (platform.c.getenv("FZ_OVERLAP") != null) {
        try overlap(&r, m, ref, gpa);
        return;
    }

    try runDefault(&r, m, ref, gpa, args, own_prompt);
}
