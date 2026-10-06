//! Flash Next windows of 1-8 rows on the Python engine's own kernels (tools/zig/flashnext_dump.py): every launch is a
//! recorded variant, every weight the pack's or the checkpoint's. Checks one-row greedy tokens and every row of the
//! Python engine's drafted windows (with rollback), then times each window size.
const std = @import("std");
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
const Lane = fz.Lane;
const Layer = fz.Layer;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const f32Buf = fz.f32Buf;
const i32Buf = fz.i32Buf;
const Slot = fz.Slot;
const Mtp = fz.Mtp;
const Model = fz.Model;
const DepthRule = fz.DepthRule;
const copyDrafts = fz.copyDrafts;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const jsonInt = fz.jsonInt;

fn armName(seg_ab: bool, arm: usize) []const u8 {
    if (seg_ab) return if (arm == 1) "segments on " else "segments off";
    if (std.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(std.c.getenv("FZ_AB").?), "fz")) return if (arm == 1) "fz kernels" else "recorded  ";
    return if (arm == 1) "copy on " else "copy off";
}

/// FZ_PROMPT_N: only the prompt's first n tokens (all when unset).
fn promptCut() !usize {
    const v = std.c.getenv("FZ_PROMPT_N") orelse return std.math.maxInt(usize);
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
    if (std.c.getenv("FZ_COMPILE_CHECK") != null) { // every prompt-kernel source and the 6-bit file on this Metal compiler
        var failed: usize = 0;
        var files: std.ArrayList(ks.File) = .empty;
        try files.appendSlice(gpa, &ks.prefill);
        try files.append(gpa, .{ .name = "qmm6_nax", .text = ks.flashnext_qmm6 });
        try files.append(gpa, .{ .name = "fn_attn", .text = ks.flashnext_attn });
        try files.append(gpa, .{ .name = "qmm6_nax (simdgroup fragments)", .text = "#define TF_SIMD_FRAGS 1\n" ++ ks.flashnext_qmm6 });
        try files.append(gpa, .{ .name = "fn_attn (simdgroup fragments)", .text = "#define TF_SIMD_FRAGS 1\n" ++ ks.flashnext_attn });
        for (files.items) |f| {
            const text = try std.mem.replaceOwned(u8, gpa, f.text, "#include \"../nax.h\"", ks.nax);
            defer gpa.free(text);
            if (mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx())) |_| {
                std.debug.print("compiled {s}\n", .{f.name});
            } else |e| {
                failed += 1;
                std.debug.print("FAILED {s}: {s}\n", .{ f.name, @errorName(e) });
            }
        }
        std.debug.print("{d} of {d} sources compiled\n", .{ files.items.len - failed, files.items.len });
        if (fz.frags.check(device, try device.queue(), arena)) |_| std.debug.print("fragment layout checked ({s})\n", .{if (device.tensorUnits()) "tensor units" else "simdgroup matrices"}) else |e| {
            failed += 1;
            std.debug.print("FAILED fragment check: {s}\n", .{@errorName(e)});
        }
        std.process.exit(if (failed == 0) 0 else 1);
    }
    if (std.c.getenv("FZ_OPORDER") != null) return @import("flashnext_bench.zig").opOrder(device); // no model loaded
    if (std.c.getenv("FZ_MMA_PEAK") != null) { // the tensor units' rate on register fragments: 16x32x16 ops a second
        const alib = try mtl.Library.fromSource(device, try fz.frags.source(device, arena, ks.flashnext_attn), mtl.CompileOptions.mlx());
        const pipe = try mtl.Pipeline.init(device, alib, "tf_mma_peak", false);
        const queue = try device.queue();
        const x = try device.buffer(64 * 64 * 2, opts);
        @memset(x.contents()[0 .. 64 * 64 * 2], 0);
        const y = try device.buffer(4 * 4 * 65536, opts);
        const iters: i32 = 4096;
        for ([_]usize{ 1024, 4096, 16384 }) |groups| {
            const cb = queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(pipe);
            enc.setBuffer(x, 0, 0);
            enc.setBytes(std.mem.asBytes(&iters), 1);
            enc.setBuffer(y, 0, 2);
            enc.dispatchThreads(mtl.Size.of(groups * 128, 1, 1), mtl.Size.of(128, 1, 1));
            enc.end();
            cb.commit();
            cb.wait();
            const flops = @as(f64, @floatFromInt(groups * 4)) * @as(f64, @floatFromInt(iters)) * 4.0 * (16.0 * 32.0 * 16.0 * 2.0);
            std.debug.print("tensor-op peak, {d} threadgroups of 4 simdgroups: {d:.1} TFLOPS ({d:.2} ms)\n", .{ groups, flops / cb.gpuSeconds() / 1e12, cb.gpuSeconds() * 1e3 });
        }
        return;
    }
    if (std.c.getenv("FZ_ENGINE") != null) { // the served engine: its own load, warm-up and reply, on FZ_REF's prompt
        const fx = @import("tensorfold").flashnext_engine;
        const rp = std.mem.span(std.c.getenv("FZ_REF") orelse return error.NoRef);
        const rf = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}", .{rp}, 0));
        const rj = try std.json.parseFromSliceLeaky(std.json.Value, arena, rf.bytes[0..rf.size], .{});
        const all_items = rj.object.get("prompt").?.array.items;
        const items = all_items[0..@min(all_items.len, try promptCut())];
        const toks = try arena.alloc(u32, items.len);
        for (items, 0..) |x, i| toks[i] = @intCast(x.integer);
        const e = try fx.Engine.load(gpa, init.io, args[1], args[2]);
        if (std.c.getenv("FZ_COPY_MIN")) |v| e.copy_min = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("FZ_COPY_LONG")) |v| e.copy_long = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("FZ_ENGINE_WARM") != null) try e.warm();
        const Show = struct {
            got: std.ArrayList(u32) = .empty,
            a: std.mem.Allocator,
            fn prefilled(_: *anyopaque) void {}
            fn tokens(ctx: *anyopaque, t: []const u32) bool {
                const sh: *@This() = @ptrCast(@alignCast(ctx));
                sh.got.appendSlice(sh.a, t) catch {};
                return false;
            }
            fn cancelled(_: *anyopaque) bool {
                return false;
            }
        };
        const n_out: usize = if (std.c.getenv("FZ_N")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 256;
        var replies: [2][]u32 = undefined;
        const seg_ab = std.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(std.c.getenv("FZ_AB").?), "seg");
        var depths: std.ArrayList(?usize) = .empty; // FZ_DEPTHS=3,5,7: each depth's arms in turn (0: the depth rule, p: plain)
        if (std.c.getenv("FZ_DEPTHS")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            while (it.next()) |d| {
                if (std.mem.eql(u8, d, "p")) {
                    try depths.append(arena, 0);
                    continue;
                }
                const n = try std.fmt.parseInt(usize, d, 10);
                try depths.append(arena, if (n == 0) null else n);
            }
        } else try depths.append(arena, if (std.c.getenv("FZ_DEPTH")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null);
        // FZ_AB=fz (with FZ_LANE=1 FZ_GDN=2): the recorded dense and DeltaNet kernels, then fz_lane, fz_gdn and kept states
        const fz_ab = std.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(std.c.getenv("FZ_AB").?), "fz");
        const fz_on = .{ e.r.lane_new, e.r.gdn_pipe, e.r.gdn_kept };
        for (depths.items) |depth| for (0..2) |arm| { // copy drafts off, then on (FZ_AB=seg: staggered prompt segments off, then on)
            const plain = depth != null and depth.? == 0; // one token a round: one arm (copies need drafts)
            if (plain and arm == 1) continue;
            const name = if (plain) "plain   " else armName(seg_ab, arm);
            if (seg_ab) e.segments = arm == 1 else if (fz_ab) {
                e.r.lane_new = arm == 1 and fz_on[0];
                e.r.gdn_pipe = if (arm == 1) fz_on[1] else null;
                e.r.gdn_kept = if (arm == 1) fz_on[2] else null;
            } else e.copy = arm == 1;
            var sh: Show = .{ .a = arena };
            const warm: fx.Out = .{ .ctx = &sh, .prefilled = Show.prefilled, .tokens = Show.tokens, .cancelled = Show.cancelled };
            _ = if (e.followsPeer()) try e.followWith(warm) else try e.generate(toks, 8, &.{}, null, warm); // speed-up rank 1 runs rank 0's requests
            sh = .{ .a = arena };
            var first_at: f64 = 0;
            const Timed = struct {
                sh: *Show,
                t0: f64,
                first: *f64,
                fn prefilled(ctx: *anyopaque) void {
                    const tt: *@This() = @ptrCast(@alignCast(ctx));
                    tt.first.* = mtl.clock.seconds() - tt.t0;
                }
                fn tokens(ctx: *anyopaque, t: []const u32) bool {
                    const tt: *@This() = @ptrCast(@alignCast(ctx));
                    return Show.tokens(tt.sh, t);
                }
            };
            const s0 = mtl.clock.seconds();
            var tm: Timed = .{ .sh = &sh, .t0 = s0, .first = &first_at };
            const timed: fx.Out = .{ .ctx = &tm, .prefilled = Timed.prefilled, .tokens = Timed.tokens, .cancelled = Show.cancelled };
            const res = (if (e.followsPeer()) try e.followWith(timed) else try e.generate(toks, n_out, &.{}, depth, timed)) orelse return error.PeerClosed;
            const wall = mtl.clock.seconds() - s0;
            const made: f64 = @floatFromInt(sh.got.items.len - 1);
            std.debug.print("{s} depth {d}: {d} tokens; prompt {d:.2} s; decode {d:.1} tok/s; {d:.2} tokens a round ({d} rounds, {d} copied rounds landing {d:.2})\n", .{ name, depth orelse 0, sh.got.items.len, first_at, made / (wall - first_at), made / @as(f64, @floatFromInt(@max(res.rounds, 1))), res.rounds, res.copy_rounds, @as(f64, @floatFromInt(res.copy_accepted)) / @as(f64, @floatFromInt(@max(res.copy_rounds, 1))) });
            replies[arm] = sh.got.items;
            std.debug.print("{s}: reply hash {x:0>16}\n", .{ name, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(sh.got.items)) });
            if (arm == 1) for (res.copy_by_len, 0..) |cl, n| if (cl[0] > 0) std.debug.print("    match {d}: {d} rounds, {d:.2} landed\n", .{ n, cl[0], @as(f64, @floatFromInt(cl[1])) / @as(f64, @floatFromInt(cl[0])) });
        };
        if (std.c.getenv("FZ_NO_REF") != null) return; // the reply hashes are the check (two Macs against one)
        // the reference: the same prompt chunks, then one row a step
        e.hostMode();
        const mm = e.m;
        mm.reset();
        var at: usize = 0;
        var tok: u32 = 0;
        while (at < toks.len) {
            const n = @min(e.pr.step, toks.len - at);
            tok = try e.pr.chunk(mm, gpa, toks[at .. at + n]);
            at += n;
        }
        var refr: std.ArrayList(u32) = .empty;
        try refr.append(arena, tok);
        var pk: [MAXR]u32 = undefined;
        while (refr.items.len < n_out) {
            const last = refr.items[refr.items.len - 1];
            try mm.window(&.{last}, &pk);
            mm.keepRows(&.{last}, 1);
            try refr.append(arena, pk[0]);
        }
        for (replies, 0..) |rep_toks, arm| {
            var same: usize = 0;
            while (same < @min(rep_toks.len, refr.items.len) and rep_toks[same] == refr.items[same]) same += 1;
            std.debug.print("{s}: {d}/{d} tokens equal to the one-row reference\n", .{ armName(seg_ab, arm), same, n_out });
        }
        return;
    }
    var r = Run{ .arena = arena, .device = device, .queue = try device.queue() };
    r.fused_xsum = std.c.getenv("FZ_FUSED_XSUM") != null;
    r.serial = std.c.getenv("FZ_SERIAL") != null;
    r.xnew = std.c.getenv("FZ_XNEW") != null;
    r.split = std.c.getenv("FZ_SPLIT") != null;
    r.gdn_step = std.c.getenv("FZ_GDN_STEP") != null;
    r.hc_mma = std.c.getenv("FZ_HC_MMA") != null;
    r.hc_up = std.c.getenv("FZ_HCCHECK") != null or if (std.c.getenv("FZ_HC_UP")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    r.event = try device.sharedEvent();
    r.dense = r.xnew and std.c.getenv("FZ_DENSE") != null;
    r.dense_target = r.dense and std.c.getenv("FZ_DENSE_TARGET") != null;
    r.xpack = r.xnew and std.c.getenv("FZ_XPACK") != null;
    r.grouped = r.xnew and !r.xpack and std.c.getenv("FZ_GROUPED") != null;
    r.xfused = r.xnew and !r.xpack and !r.grouped and std.c.getenv("FZ_XFUSED") != null;
    r.xsx = r.xnew and std.c.getenv("FZ_XSX") != null;
    r.copy = std.c.getenv("FZ_COPY") != null;
    r.prefetch = !r.serial and std.c.getenv("FZ_PREFETCH") != null;
    const t0 = mtl.clock.seconds();
    try r.compile(args[2]);
    r.sel = try Select.init(&r, MAXR);
    if (std.c.getenv("FZ_CATCH_CHECK") != null) return catchCheck(&r, arena);
    r.lane_new = r.xnew and std.c.getenv("FZ_LANE") != null; // fz_lane and fz_gdn on the target (same bits)
    if (r.xnew and std.c.getenv("FZ_GDN") != null) r.gdn_pipe = try fz.gdn_step.compile(&r, false);
    const t1 = mtl.clock.seconds();

    const index_file = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}/model.safetensors.index.json", .{args[1]}, 0));
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_file.bytes[0..index_file.size], .{});
    var files: std.StringHashMapUnmanaged(void) = .empty;
    var wit = index.object.get("weight_map").?.object.iterator();
    while (wit.next()) |kv| try files.put(arena, kv.value_ptr.string, {});
    var fit = files.keyIterator();
    while (fit.next()) |name| try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/{s}", .{ args[1], name.* }, 0));
    try r.indexFile(try std.fmt.allocPrintSentinel(arena, "{s}/pack.safetensors", .{args[2]}, 0));

    const own_prompt = std.c.getenv("FZ_REF") != null; // another prompt: passes against Python's records are skipped
    const ref_path = if (std.c.getenv("FZ_REF")) |p| try std.fmt.allocPrintSentinel(arena, "{s}", .{std.mem.span(p)}, 0) else try std.fmt.allocPrintSentinel(arena, "{s}/ref.json", .{args[2]}, 0);
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

    if (std.c.getenv("FZ_DBENCH") != null) { // dense classes and DeltaNet timed by rows (flashnext_bench.zig)
        var toks: [MAXR]u32 = undefined;
        for (0..MAXR) |i| toks[i] = @intCast(ref.object.get("prompt").?.array.items[i].integer);
        return @import("flashnext_bench.zig").run(&r, m, &toks, gpa, arena);
    }
    if (std.c.getenv("FZ_PROFILE") != null) {
        if (std.c.getenv("TF_FLASHNEXT_TP")) |path| r.tp = try fz.Tp2.init(arena, r.device, std.mem.span(path));
        var hcskips: std.ArrayList(u32) = .empty; // FZ_HCSKIP=1,2,4: the profile again under each hc knock-out
        if (std.c.getenv("FZ_HCSKIP")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            while (it.next()) |x| try hcskips.append(arena, try std.fmt.parseInt(u32, x, 10));
        } else try hcskips.append(arena, 0);
        const names = [_][]const u8{ "none", "hc", "dense", "experts", "router", "gdn", "attn", "ple", "head", "tp" };
        var toks: [MAXR]u32 = undefined;
        for (0..MAXR) |i| toks[i] = @intCast(ref.object.get("prompt").?.array.items[i].integer);
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (0..3) |_| try m.window(toks[0..1], &pk);
        if (std.c.getenv("FZ_PROFILE_AB") != null) { // in-model A/B, interleaved: recorded, fz_lane, fz_gdn, both
            const gpipe = try fz.gdn_step.compile(&r, false);
            const arms = [_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } };
            for ([_]usize{ 1, 2, 4, 6, 8, 16 }) |rows| {
                var ms: [4]f64 = @splat(0);
                for (0..5) |_| for (arms, 0..) |arm, a| {
                    r.lane_new = arm[0];
                    r.gdn_pipe = if (arm[1]) gpipe else null;
                    m.gpu_seconds = 0;
                    for (0..20) |_| try m.window(toks[0..rows], &pk);
                    ms[a] += m.gpu_seconds * 1e3 / 100;
                };
                std.debug.print("rows {d:2}: window ms recorded {d:6.3}, fz_lane {d:6.3}, fz_gdn {d:6.3}, both {d:6.3}\n", .{ rows, ms[0], ms[1], ms[2], ms[3] });
            }
            return;
        }
        var ups: std.ArrayList(bool) = .empty; // FZ_HC_UP=0,1: the profile with the recorded up projection, then fz_hc_up
        if (std.c.getenv("FZ_HC_UP")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            while (it.next()) |x| try ups.append(arena, !std.mem.eql(u8, x, "0"));
        } else try ups.append(arena, false);
        var prow: std.ArrayList(usize) = .empty; // FZ_PROFILE_ROWS=2,3: the windows' widths (default 1, 4, 8)
        if (std.c.getenv("FZ_PROFILE_ROWS")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            while (it.next()) |x| try prow.append(arena, try std.fmt.parseInt(usize, x, 10));
        } else try prow.appendSlice(arena, &.{ 1, 4, 8 });
        for (ups.items) |up| for (hcskips.items) |hs| for (prow.items) |rows| {
            r.hc_up = up;
            r.hcskip = hs;
            var base: f64 = 0;
            for (names, 0..) |name, c| {
                const only = std.c.getenv("FZ_PROFILE_ONLY") != null; // every class but this one knocked out
                r.skip = if (c == 0) (if (only) 0x1ff else 0) else if (only) 0x1ff & ~(@as(u32, 1) << @intCast(c - 1)) else @as(u32, 1) << @intCast(c - 1);
                m.gpu_seconds = 0;
                for (0..20) |_| try m.window(toks[0..rows], &pk);
                const ms = m.gpu_seconds * 1e3 / 20;
                if (c == 0) base = ms;
                std.debug.print("hc_up {d} hcskip {d} rows {d}: without {s:8} {d:6.2} ms GPU  ({d:5.2} ms)\n", .{ @intFromBool(up), hs, rows, name, ms, base - ms });
            }
        };
        r.hcskip = 0;
        r.skip = 0;
        for ([_]usize{ 1, 4, 8 }) |rows| { // the MTP head: a chain step (one row) and a window's catch-up
            m.gpu_seconds = 0;
            for (0..20) |_| {
                m.mtp.drafted = 0;
                _ = try m.mtpRun(toks[0..rows], m.mtp.last);
                m.mtp.pos -= rows;
            }
            std.debug.print("mtp rows {d}: {d:6.2} ms GPU\n", .{ rows, m.gpu_seconds * 1e3 / 20 });
        }
        return;
    }
    if (std.c.getenv("FZ_DUAL") != null) { // two lane groups on two queues: one window, two in a row, two at once
        const want0 = ref.object.get("tokens").?.array.items;
        const pr = ref.object.get("prompt").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        const q2 = try device.queue();
        const ta = m.t;
        var tb = m.t;
        const info = @typeInfo(Tmp).@"struct";
        inline for (info.field_names, info.field_types) |fname, ftype| {
            const f = .{ .name = fname, .type = ftype };
            if (f.type == Buf) {
                const old = @field(ta, f.name);
                const n = old.b.length() - old.off;
                const nb = try r.buffer(n);
                @memcpy(nb.contents()[0..n], old.b.contents()[old.off .. old.off + n]);
                @field(tb, f.name) = .{ .b = nb };
            } else if (f.type == [2]Buf) {
                for (0..2) |k| {
                    const old = @field(ta, f.name)[k];
                    const n = old.b.length() - old.off;
                    @field(tb, f.name)[k] = .{ .b = try r.buffer(n) };
                }
            }
        }
        for ([_]usize{ 1, 4, 8 }) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            m.windowMeta(rows);
            const ids = ta.ids8.b.slice(u32, MAXR);
            for (0..MAXR) |i| ids[i] = if (i < rows) toks[i] else 0;
            m.pleIds(toks[0..rows]);
            var toks_b: [MAXR]u32 = undefined; // the second group: other tokens at the same positions
            for (0..rows) |i| toks_b[i] = @intCast(want0[16 + i].integer);
            const ids_b = tb.ids8.b.slice(u32, MAXR);
            for (0..MAXR) |i| ids_b[i] = if (i < rows) toks_b[i] else 0;
            m.t = tb;
            m.pleIds(toks_b[0..rows]);
            m.t = ta;
            var wall: [3]f64 = undefined;
            for (0..3) |mode| {
                const n_it: usize = 20;
                const a = mtl.clock.seconds();
                for (0..n_it) |_| {
                    const cb1 = r.queue.commandBuffer();
                    r.enc = cb1.compute(.serial);
                    m.t = ta;
                    try m.windowEncode(rows, ta.ids8);
                    r.enc.end();
                    if (mode == 0) {
                        cb1.commit();
                        cb1.wait();
                        continue;
                    }
                    const cb2 = (if (mode == 1) r.queue else q2).commandBuffer();
                    r.enc = cb2.compute(.serial);
                    m.t = tb;
                    try m.windowEncode(rows, tb.ids8);
                    r.enc.end();
                    cb1.commit();
                    cb2.commit();
                    cb1.wait();
                    cb2.wait();
                }
                wall[mode] = (mtl.clock.seconds() - a) * 1e3 / @as(f64, @floatFromInt(n_it));
            }
            m.t = ta;
            std.debug.print("{d} lanes: one group {d:.2} ms, two groups in a row {d:.2} ms ({d:.2}x), two groups at once {d:.2} ms ({d:.2}x)\n", .{ rows, wall[0], wall[1], wall[1] / wall[0], wall[2], wall[2] / wall[0] });
        }
        return;
    }
    if (std.c.getenv("FZ_HCCHECK") != null) { // fz_hc_up against the recorded up projection: every row's logits and streams, bit for bit
        const want0 = ref.object.get("tokens").?.array.items;
        const pr = ref.object.get("prompt").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        const keep = try gpa.alloc(u8, MAXR * VOCAB * 2);
        defer gpa.free(keep);
        const keep_h = try gpa.alloc(u8, MAXR * WIDE * 2);
        defer gpa.free(keep_h);
        const Diff = struct {
            fn of(a: []const u8, b: Buf, n: usize) usize {
                var d: usize = 0;
                const c = b.b.contents()[b.off .. b.off + 2 * n];
                for (0..n) |i| d += @intFromBool(a[2 * i] != c[2 * i] or a[2 * i + 1] != c[2 * i + 1]);
                return d;
            }
        };
        var toks: [MAXR]u32 = undefined;
        for (0..MAXR) |i| toks[i] = @intCast(want0[i].integer);
        var bad: usize = 0;
        for (1..MAXR + 1) |rows| {
            r.hc_up = false;
            try m.window(toks[0..rows], &pk);
            @memcpy(keep[0 .. rows * VOCAB * 2], m.t.logits.b.contents()[0 .. rows * VOCAB * 2]);
            @memcpy(keep_h[0 .. rows * WIDE * 2], m.last.b.contents()[m.last.off .. m.last.off + rows * WIDE * 2]);
            r.hc_up = true;
            try m.window(toks[0..rows], &pk);
            const dl = Diff.of(keep, m.t.logits, rows * VOCAB);
            const dh = Diff.of(keep_h, m.last, rows * WIDE);
            std.debug.print("rows {d}: {d} of {d} logits and {d} of {d} stream values differ\n", .{ rows, dl, rows * VOCAB, dh, rows * WIDE });
            bad += dl + dh;
        }
        std.debug.print("hc check: {s}\n", .{if (bad == 0) "bit-identical" else "DIFFERENT"});
        return;
    }
    if (std.c.getenv("FZ_GCHECK") != null) { // grouped or fused experts against fz_xgu/fz_xdown: every row's logits, bit for bit
        const want0 = ref.object.get("tokens").?.array.items;
        const pr = ref.object.get("prompt").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        const keep = try gpa.alloc(u16, MAXR * VOCAB);
        defer gpa.free(keep);
        for (1..MAXR + 1) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            const g0, const f0, const x0 = .{ r.grouped, r.xfused, r.xsx };
            r.grouped, r.xfused, r.xsx = .{ false, false, false };
            try m.window(toks[0..rows], &pk);
            @memcpy(keep[0 .. rows * VOCAB], m.t.logits.b.slice(u16, rows * VOCAB));
            r.grouped, r.xfused, r.xsx = .{ g0, f0, x0 };
            try m.window(toks[0..rows], &pk);
            const now = m.t.logits.b.slice(u16, rows * VOCAB);
            var diff: usize = 0;
            for (keep[0 .. rows * VOCAB], now) |a, b| diff += @intFromBool(a != b);
            std.debug.print("rows {d}: {d} of {d} logits differ\n", .{ rows, diff, rows * VOCAB });
        }
        return;
    }
    if (std.c.getenv("FZ_XTIME") != null) { // the experts' share of a window: chain rows against one token repeated
        const want0 = ref.object.get("tokens").?.array.items;
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (0..3) |_| try m.window(&.{@intCast(want0[0].integer)}, &pk);
        if (std.c.getenv("FZ_GPARTS") != null) for ([_]usize{ 2, 8 }) |rows| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
            for ([_]u32{ 0, 1, 4, 6 }) |gs| {
                r.gskip = gs;
                m.gpu_seconds = 0;
                for (0..30) |_| try m.window(toks[0..rows], &pk);
                std.debug.print("rows {d} gskip {d}: window {d:6.2} ms\n", .{ rows, gs, m.gpu_seconds * 1e3 / 30 });
            }
            r.gskip = 0;
        };
        for ([_]usize{ 1, 2, 4, 8 }) |rows| {
            for ([_]bool{ false, true }) |same| {
                var toks: [MAXR]u32 = undefined;
                for (0..rows) |i| toks[i] = @intCast(want0[if (same) 0 else i].integer);
                var ms: [2]f64 = undefined;
                for ([_]u32{ 0, 1 << 2 }, 0..) |skip, j| {
                    r.skip = skip;
                    m.gpu_seconds = 0;
                    for (0..30) |_| try m.window(toks[0..rows], &pk);
                    ms[j] = m.gpu_seconds * 1e3 / 30;
                }
                r.skip = 0;
                std.debug.print("rows {d} {s}: window {d:6.2} ms, experts {d:5.2} ms\n", .{ rows, if (same) "one token" else "chain    ", ms[0], ms[0] - ms[1] });
            }
        }
        return;
    }
    if (std.c.getenv("FZ_PREFILL") != null) { // the prompt path against the exact 8-row windows, then one-row decode
        const all_items = ref.object.get("prompt").?.array.items;
        const pr_items = all_items[0..@min(all_items.len, try promptCut())]; // FZ_PROMPT_N: the prompt's first n tokens
        const toks = try gpa.alloc(u32, pr_items.len);
        defer gpa.free(toks);
        for (pr_items, 0..) |x, i| toks[i] = @intCast(x.integer);
        if (std.c.getenv("FZ_SATTN_CHECK") != null) { // prompt attention: the decode's kernels vs the tensor units
            var pr = try Prompt.init(&r, args[2], r.xnew_header);
            if (std.c.getenv("FZ_STEP")) |v| pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
            const ab0: []const u8 = if (std.c.getenv("FZ_AB")) |v| std.mem.span(v) else "attn";
            const seg2 = std.mem.eql(u8, ab0, "seg2");
            // seg2: one chunk at a time against staggered segments (core/segments.zig): FZ_SEGS=2 (default) the
            // engine's rule (two segments of at least FZ_SEG_MIN rows), 1 one segment, 3-4 that many of up to a chunk
            const nseg: usize = if (std.c.getenv("FZ_SEGS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2;
            const seg_min: usize = if (std.c.getenv("FZ_SEG_MIN")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 1;
            if (seg2 and (nseg == 0 or nseg > tf.segments.MAX)) return error.Segments;
            var extra: [3]Prompt = undefined;
            var ps: [4]*Prompt = undefined;
            ps[0] = &pr;
            if (seg2) for (1..nseg) |k| {
                extra[k - 1] = try pr.sibling();
                ps[k] = &extra[k - 1];
            };
            var outs: [2][48]u32 = undefined;
            const logits: [2][]u16 = .{ try gpa.alloc(u16, VOCAB), try gpa.alloc(u16, VOCAB) };
            var best: [2]f64 = .{ 1e9, 1e9 };
            const ab: []const u8 = if (std.c.getenv("FZ_AB")) |v| std.mem.span(v) else "attn";
            for (0..2) |arm| { // arm 0 the old kernels, arm 1 the new: FZ_AB attn (default), scan or tiles
                if (std.mem.eql(u8, ab, "scan")) pr.scan4 = arm == 1 else if (std.mem.eql(u8, ab, "tiles")) pr.tall_tiles = arm == 1 else if (std.mem.eql(u8, ab, "gu")) pr.fused_gu = arm == 1 else if (std.mem.eql(u8, ab, "bm32")) pr.expert_bm = if (arm == 1) 32 else 0 else if (std.mem.eql(u8, ab, "scan8")) pr.scan8 = arm == 1 else if (std.mem.eql(u8, ab, "seg2")) {} else pr.fast_attn = arm == 1;
                for (0..3) |run| {
                    m.reset();
                    const c0 = mtl.clock.seconds();
                    var at: usize = 0;
                    var first: u32 = 0;
                    while (at < toks.len) {
                        const left = toks.len - at;
                        if (seg2 and arm == 1) {
                            const c: tf.segments.Call = switch (nseg) {
                                1 => .{ .rows = @min(pr.step, left), .parts = 1 },
                                2 => tf.segments.next(left, pr.step, seg_min),
                                else => .{ .rows = @min(nseg * pr.step, left), .parts = (@min(nseg * pr.step, left) + pr.step - 1) / pr.step },
                            };
                            first = try Prompt.chunkN(ps[0..c.parts], m, gpa, toks[at .. at + c.rows]);
                            at += c.rows;
                        } else {
                            const n = @min(pr.step, left);
                            first = try pr.chunk(m, gpa, toks[at .. at + n]);
                            at += n;
                        }
                    }
                    best[arm] = @min(best[arm], mtl.clock.seconds() - c0);
                    if (run > 0) continue;
                    @memcpy(logits[arm], m.t.logits.b.slice(u16, VOCAB));
                    var pk: [MAXR]u32 = undefined;
                    outs[arm][0] = first;
                    for (1..48) |k| {
                        try m.window(&.{outs[arm][k - 1]}, &pk);
                        m.keepRows(&.{outs[arm][k - 1]}, 1);
                        outs[arm][k] = pk[0];
                    }
                }
            }
            var worst: f64 = 0;
            for (logits[0], logits[1]) |x, y| {
                const fx: f64 = @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
                const fy: f64 = @floatCast(@as(f32, @bitCast(@as(u32, y) << 16)));
                worst = @max(worst, @abs(fx - fy));
            }
            var same: usize = 0;
            while (same < 48 and outs[0][same] == outs[1][same]) same += 1;
            std.debug.print("prompt {d} tokens, chunks of {d}: old {d:.3} s ({d:.0} tok/s), new {d:.3} s ({d:.0} tok/s)\n", .{ toks.len, pr.step, best[0], @as(f64, @floatFromInt(toks.len)) / best[0], best[1], @as(f64, @floatFromInt(toks.len)) / best[1] });
            std.debug.print("first token {d} vs {d}; last-row logits max diff {d:.4}; 48 decoded after each: {d} equal\n", .{ outs[0][0], outs[1][0], worst, same });
            std.debug.print("  old {any}\n  new {any}\n", .{ outs[0][0..16], outs[1][0..16] });
            return;
        }
        var pk: [MAXR]u32 = undefined;
        const n_dec: usize = 48;
        // the reference: windows of up to 8 rows (each row's bits equal a one-row step); FZ_PF_REF=n stops it after n
        // tokens (0: none), to tell what the prompt path reads from an earlier pass
        m.reset();
        const a0 = mtl.clock.seconds();
        var at: usize = 0;
        var last_n: usize = 1;
        const ref_upto: usize = if (std.c.getenv("FZ_PF_REF")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else toks.len;
        while (at < @min(ref_upto, toks.len)) {
            const n: usize = @min(MAXR, toks.len - at);
            try m.window(toks[at .. at + n], &pk);
            m.keepRows(toks[at .. at + n], n);
            at += n;
            last_n = n;
        }
        const a1 = mtl.clock.seconds();
        const ref_logits = try gpa.alloc(u16, VOCAB);
        defer gpa.free(ref_logits);
        @memcpy(ref_logits, m.t.logits.b.slice(u16, MAXR * VOCAB)[(last_n - 1) * VOCAB .. last_n * VOCAB]);
        var want_t: std.ArrayList(u32) = .empty;
        var margins: [64]f32 = @splat(0);
        try want_t.append(gpa, pk[last_n - 1]);
        while (want_t.items.len < n_dec) {
            const tk = want_t.items[want_t.items.len - 1];
            try m.window(&.{tk}, &pk);
            m.keepRows(&.{tk}, 1);
            { // the top-two logit margin of this step's pick
                var top1: f32 = -std.math.inf(f32);
                var top2: f32 = -std.math.inf(f32);
                for (m.t.logits.b.slice(u16, VOCAB)) |raw| {
                    const v: f32 = @bitCast(@as(u32, raw) << 16);
                    if (v > top1) {
                        top2 = top1;
                        top1 = v;
                    } else if (v > top2) top2 = v;
                }
                margins[want_t.items.len] = top1 - top2;
            }
            try want_t.append(gpa, pk[0]);
        }
        // the prompt path
        var pr = try Prompt.init(&r, args[2], r.xnew_header);
        if (std.c.getenv("FZ_STEP")) |v| pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        m.reset();
        const b0 = mtl.clock.seconds();
        var first: u32 = 0;
        at = 0;
        while (at < toks.len) {
            const n: usize = @min(pr.step, toks.len - at);
            first = try pr.chunk(m, gpa, toks[at .. at + n]);
            at += n;
        }
        const b1 = mtl.clock.seconds();
        const lg = m.t.logits.b.slice(u16, VOCAB);
        var worst: f64 = 0;
        var scale: f64 = 0;
        for (lg, ref_logits) |x, y| {
            const fx: f64 = @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
            const fy: f64 = @floatCast(@as(f32, @bitCast(@as(u32, y) << 16)));
            worst = @max(worst, @abs(fx - fy));
            scale = @max(scale, @abs(fy));
        }
        var got_t: std.ArrayList(u32) = .empty;
        try got_t.append(gpa, first);
        while (got_t.items.len < n_dec) {
            const tk = got_t.items[got_t.items.len - 1];
            try m.window(&.{tk}, &pk);
            m.keepRows(&.{tk}, 1);
            try got_t.append(gpa, pk[0]);
        }
        var same: usize = 0;
        while (same < n_dec and got_t.items[same] == want_t.items[same]) same += 1;
        std.debug.print("prompt {d} tokens: 8-row windows {d:.3} s ({d:.0} tok/s); prompt path {d:.3} s ({d:.0} tok/s)\n", .{ toks.len, a1 - a0, @as(f64, @floatFromInt(toks.len)) / (a1 - a0), b1 - b0, @as(f64, @floatFromInt(toks.len)) / (b1 - b0) });
        std.debug.print("first token {d} vs {d}; last-row logits max diff {d:.4} (max |logit| {d:.2}); decode after it: {d}/{d} tokens equal\n", .{ first, want_t.items[0], worst, scale, same, n_dec });
        if (same < n_dec) std.debug.print("  the exact path's top-two margin where they part (token {d}): {d:.4}; median margin {d:.3}\n", .{ same, margins[same], blk: {
            var sorted = margins;
            std.mem.sort(f32, sorted[1..n_dec], {}, std.sort.asc(f32));
            break :blk sorted[n_dec / 2];
        } });
        try pr.classes(m, @min(toks.len, pr.step));
        const skips = [_]u32{ 0, 4 | 1024, 1024, 2, 16, 1 };
        const what = [_][]const u8{ "none", "attention", "sparse attention", "DeltaNet", "hyper-connections", "experts" };
        var base_s: f64 = 0;
        for (skips, what) |sk, w| {
            pr.skip = sk;
            var best: f64 = 1e9;
            for (0..3) |_| {
                m.reset();
                m.gpu_seconds = 0;
                const c0 = mtl.clock.seconds();
                at = 0;
                while (at < toks.len) {
                    const n: usize = @min(pr.step, toks.len - at);
                    _ = try pr.chunk(m, gpa, toks[at .. at + n]);
                    at += n;
                }
                best = @min(best, mtl.clock.seconds() - c0);
            }
            if (sk == 0) base_s = best;
            std.debug.print("warm, without {s:18}: {d:.3} s ({d:.0} tok/s), saves {d:.3} s\n", .{ w, best, @as(f64, @floatFromInt(toks.len)) / best, base_s - best });
        }
        pr.skip = 0;
        return;
    }
    if (std.c.getenv("FZ_QMM6") != null) { // 6-bit tensor-unit projections for prompt chunks: checked and timed
        const lib = try mtl.Library.fromSource(device, try fz.frags.source(device, arena, ks.flashnext_qmm6), mtl.CompileOptions.mlx());
        const qmm_pipe = try mtl.Pipeline.init(device, lib, "tf_qmm6_t_nax", false);
        const off_pipe = try mtl.Pipeline.init(device, lib, "tf_expert_offsets6", false);
        const g64_pipe = try mtl.Pipeline.init(device, lib, "tf_gather_qmm6_nax_64", false);
        const W = m.layers[0].ex[0];
        const S = m.layers[0].ex[1];
        const Bi = m.layers[0].ex[2];
        const K: usize = 2560;
        const WPR = K * 6 / 32;
        const KG = K / 32;
        const Ref = struct {
            fn bf(v: u16) f64 {
                return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
            }
            fn code(w: []const u32, j: usize) u32 { // value j of a row's little-endian 6-bit stream
                const bit = 6 * j;
                const word = bit / 32;
                const off: u5 = @intCast(bit % 32);
                var v = w[word] >> off;
                if (@as(usize, off) > 26) v |= w[word + 1] << @intCast(32 - @as(usize, off));
                return v & 63;
            }
            fn dot(x: []const u16, w: []const u32, sc: []const u16, bi: []const u16, k: usize) f64 {
                var acc: f64 = 0;
                for (0..k) |j| acc += bf(x[j]) * (bf(sc[j / 32]) * @as(f64, @floatFromInt(code(w, j))) + bf(bi[j / 32]));
                return acc;
            }
        };
        var seed: u64 = 12345;
        const Rand = struct {
            fn next(st: *u64) u32 {
                st.* = st.* *% 6364136223846793005 +% 1442695040888963407;
                return @intCast(st.* >> 33);
            }
            fn bf16(st: *u64) u16 {
                const f: f32 = (@as(f32, @floatFromInt(next(st) % 20001)) - 10000.0) / 10000.0;
                return @intCast(@as(u32, @bitCast(f)) >> 16);
            }
        };
        const wv = W.b.contents();
        const w32: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, wv[0..W.b.length()]));
        const s16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, S.b.contents()[0..S.b.length()]));
        const b16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, Bi.b.contents()[0..Bi.b.length()]));
        // dense: experts 0..15's gate rows as one [10240, 2560] projection of 512 prompt rows
        {
            const M: usize = 512;
            const N: usize = 10240;
            const X = try r.buffer(M * K * 2);
            const x16 = X.slice(u16, M * K);
            for (x16) |*v| v.* = Rand.bf16(&seed);
            const Y = try r.buffer(M * N * 2);
            const P = try i32Buf(&r, &.{ @intCast(K), @intCast(N), @intCast(M) });
            var gpu: f64 = 0;
            for (0..21) |it| {
                const cb = r.queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(qmm_pipe);
                for ([_]Buf{ W, S, Bi, .{ .b = X }, P, .{ .b = Y } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
                enc.dispatchThreads(mtl.Size.of(((N + 63) / 64) * 128, (M + 63) / 64, 1), mtl.Size.of(128, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.err("qmm6: {s}", .{msg});
                    return error.GpuFailed;
                }
                if (it > 0) gpu += cb.gpuSeconds();
            }
            const y16 = Y.slice(u16, M * N);
            var worst: f64 = 0;
            for (0..64) |q| {
                const row = Rand.next(&seed) % M;
                const n = Rand.next(&seed) % N;
                const want_v = Ref.dot(x16[row * K .. (row + 1) * K], w32[n * WPR .. (n + 1) * WPR], s16[n * KG .. (n + 1) * KG], b16[n * KG .. (n + 1) * KG], K);
                const got = Ref.bf(y16[row * N + n]);
                const err = @abs(got - want_v) / @max(1.0, @abs(want_v));
                worst = @max(worst, err);
                _ = q;
            }
            const ms = gpu * 1e3 / 20;
            std.debug.print("qmm6 dense {d}x{d}x{d}: {d:.3} ms, {d:.1} TFLOP/s, worst rel err {e:.2}\n", .{ M, N, K, ms, 2.0 * @as(f64, @floatFromInt(M * N * K)) / (ms * 1e9), worst });
        }
        // gather: 512 prompt rows x 10 experts, pairs sorted by expert, through all 512 experts' gate projections
        {
            const M: usize = 512 * 10;
            const N: usize = 640;
            const E: usize = 512;
            const ids = try r.buffer(M * 4);
            const idv = ids.slice(u32, M);
            for (idv) |*v| v.* = @intCast(Rand.next(&seed) % E);
            std.mem.sort(u32, idv, {}, std.sort.asc(u32));
            const X = try r.buffer(M * K * 2);
            const x16 = X.slice(u16, M * K);
            for (x16) |*v| v.* = Rand.bf16(&seed);
            const Y = try r.buffer(M * N * 2);
            const O = try r.buffer((E + 1) * 4);
            const PO = try i32Buf(&r, &.{@intCast(M)});
            const P = try i32Buf(&r, &.{ @intCast(M), @intCast(N), @intCast(K), @intCast(E) });
            const tiles = M / 64 + E;
            var gpu: f64 = 0;
            for (0..21) |it| {
                const cb = r.queue.commandBuffer();
                const enc = cb.compute(.serial);
                enc.setPipeline(off_pipe);
                for ([_]Buf{ .{ .b = ids }, PO, .{ .b = O } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
                enc.dispatchThreads(mtl.Size.of(E, 1, 1), mtl.Size.of(256, 1, 1));
                enc.setPipeline(g64_pipe);
                for ([_]Buf{ .{ .b = X }, W, S, Bi, .{ .b = O }, P, .{ .b = Y } }, 0..) |b, j| enc.setBuffer(b.b, b.off, j);
                enc.dispatchThreads(mtl.Size.of(((N + 63) / 64) * 128, tiles, 1), mtl.Size.of(128, 1, 1));
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.log.err("gather6: {s}", .{msg});
                    return error.GpuFailed;
                }
                if (it > 0) gpu += cb.gpuSeconds();
            }
            const y16 = Y.slice(u16, M * N);
            var worst: f64 = 0;
            for (0..64) |_| {
                const row = Rand.next(&seed) % M;
                const n = Rand.next(&seed) % N;
                const g = idv[row] * N + n;
                const want_v = Ref.dot(x16[row * K .. (row + 1) * K], w32[g * WPR .. (g + 1) * WPR], s16[g * KG .. (g + 1) * KG], b16[g * KG .. (g + 1) * KG], K);
                const got = Ref.bf(y16[row * N + n]);
                worst = @max(worst, @abs(got - want_v) / @max(1.0, @abs(want_v)));
            }
            const ms = gpu * 1e3 / 20;
            std.debug.print("gather6 {d} pairs x{d}x{d} over {d} experts: {d:.3} ms, {d:.1} TFLOP/s, worst rel err {e:.2}\n", .{ M, N, K, E, ms, 2.0 * @as(f64, @floatFromInt(M * N * K)) / (ms * 1e9), worst });
        }
        return;
    }
    if (std.c.getenv("FZ_OVERLAP") != null) { // how many experts a window's rows share, layer by layer
        const pr = ref.object.get("prompt").?.array.items;
        const want0 = ref.object.get("tokens").?.array.items;
        r.ar = .{ .b = try r.buffer(64) };
        r.ar.b.slice(i32, 1)[0] = 1;
        const pb: Buf = .{ .b = try r.buffer(LAYERS * MAXR * 10 * 4) };
        var pk: [MAXR]u32 = undefined;
        m.reset();
        for (pr) |x| {
            try m.window(&.{@intCast(x.integer)}, &pk);
            m.keepRows(&.{@intCast(x.integer)}, 1);
        }
        r.probe = pb;
        const picks = pb.b.slice(u32, LAYERS * MAXR * 10);
        const Count = struct {
            fn distinct(sets: []std.AutoHashMap(u32, void)) f64 {
                var n: usize = 0;
                for (sets) |*u| n += u.count();
                return @as(f64, @floatFromInt(n)) / LAYERS;
            }
            fn add(sets: []std.AutoHashMap(u32, void), all: []const u32, rows: usize) !void {
                for (0..LAYERS) |l| for (all[l * MAXR * 10 .. l * MAXR * 10 + rows * 10]) |e| try sets[l].put(e, {});
            }
        };
        var chain_sum: [4]f64 = @splat(0);
        var sib_sum: [4]f64 = @splat(0);
        var far: [LAYERS]std.AutoHashMap(u32, void) = undefined;
        for (&far) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
        const starts = [_]usize{ 0, 8, 16, 24 };
        for (starts) |s0| {
            for ([_]usize{ 8, 4, 2, 1 }, 0..) |rows, ri| { // chain windows from the pending token, widest first
                var toks: [MAXR]u32 = undefined;
                for (0..rows) |i| toks[i] = @intCast(want0[s0 + i].integer);
                try m.window(toks[0..rows], &pk);
                var sets: [LAYERS]std.AutoHashMap(u32, void) = undefined;
                for (&sets) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
                try Count.add(&sets, picks, rows);
                chain_sum[3 - ri] += Count.distinct(&sets);
                if (rows == 1) try Count.add(&far, picks, 1);
            }
            // the target's 8 best tokens for the next position, each a row from the same state (tree siblings)
            const lg = m.t.logits.b.slice(u16, VOCAB);
            var best: [8]u32 = undefined;
            var bestv: [8]f32 = @splat(-std.math.inf(f32));
            for (lg, 0..) |raw, id| {
                const v: f32 = @bitCast(@as(u32, raw) << 16);
                if (v <= bestv[7]) continue;
                var k: usize = 7;
                while (k > 0 and v > bestv[k - 1]) : (k -= 1) {
                    bestv[k] = bestv[k - 1];
                    best[k] = best[k - 1];
                }
                bestv[k] = v;
                best[k] = @intCast(id);
            }
            m.keepRows(&.{@intCast(want0[s0].integer)}, 1);
            var sib: [LAYERS]std.AutoHashMap(u32, void) = undefined;
            for (&sib) |*u| u.* = std.AutoHashMap(u32, void).init(gpa);
            for (best, 0..) |cand, ci| {
                try m.window(&.{cand}, &pk);
                try Count.add(&sib, picks, 1);
                const at: ?usize = switch (ci) {
                    0 => 0,
                    1 => 1,
                    3 => 2,
                    7 => 3,
                    else => null,
                };
                if (at) |j| sib_sum[j] += Count.distinct(&sib);
            }
            var toks: [MAXR]u32 = undefined;
            for (0..7) |i| toks[i] = @intCast(want0[s0 + 1 + i].integer);
            try m.window(toks[0..7], &pk);
            m.keepRows(toks[0..7], 7);
        }
        const nstarts: f64 = @floatFromInt(starts.len);
        for ([_]usize{ 1, 2, 4, 8 }, 0..) |rows, j| {
            std.debug.print("{d} rows (of {d} picks a layer): chain {d:.1} distinct, siblings {d:.1}\n", .{ rows, rows * 10, chain_sum[j] / nstarts, sib_sum[j] / nstarts });
        }
        std.debug.print("4 rows 8 tokens apart: {d:.1} distinct\n", .{Count.distinct(&far)});
        return;
    }
    const prompt = ref.object.get("prompt").?.array.items;
    const want = ref.object.get("tokens").?.array.items;
    var pick: [MAXR]u32 = undefined;

    // FZ_LONG: prompts of any length through the prompt chunks (reference and GPU-side rounds alike)
    const long_mode = std.c.getenv("FZ_LONG") != null;
    if (std.c.getenv("FZ_PRE1") != null) try m.window(&.{1000}, &pick); // a decode window before the prompt chunks
    var pr_long: ?Prompt = if (long_mode) try Prompt.init(&r, args[2], r.xnew_header) else null;
    if (pr_long) |*pr| if (std.c.getenv("FZ_STEP")) |v| {
        pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    };
    const ptoks = try gpa.alloc(u32, prompt.len);
    defer gpa.free(ptoks);
    for (prompt, 0..) |x, i| ptoks[i] = @intCast(x.integer);

    // 1. one-row greedy steps against the Python engine's tokens
    m.reset();
    if (pr_long) |*pr| {
        const p0 = mtl.clock.seconds();
        var at: usize = 0;
        while (at < ptoks.len) {
            const n = @min(pr.step, ptoks.len - at);
            pick[0] = try pr.chunk(m, gpa, ptoks[at .. at + n]);
            at += n;
        }
        const ps = mtl.clock.seconds() - p0;
        std.debug.print("prompt {d} tokens in prompt chunks: {d:.2} s ({d:.0} tok/s)\n", .{ ptoks.len, ps, @as(f64, @floatFromInt(ptoks.len)) / ps });
    } else for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    }
    var got: std.ArrayList(u32) = .empty;
    try got.append(gpa, pick[0]);
    const t3 = mtl.clock.seconds();
    m.gpu_seconds = 0;
    while (got.items.len < want.len) {
        const last = got.items[got.items.len - 1];
        try m.window(&.{last}, &pick);
        m.keepRows(&.{last}, 1);
        try got.append(gpa, pick[0]);
    }
    const t4 = mtl.clock.seconds();
    var same: usize = 0;
    while (same < want.len and got.items[same] == @as(u32, @intCast(want[same].integer))) same += 1;
    const steps: f64 = @floatFromInt(want.len - 1);
    std.debug.print("one-row reference starts {any}\n", .{got.items[0..@min(24, got.items.len)]});
    std.debug.print("one row: {d}/{d} tokens equal to Python's; {d:.1} tok/s ({d:.2} ms a step, GPU {d:.2} ms)\n", .{ same, want.len, steps / (t4 - t3), (t4 - t3) * 1e3 / steps, m.gpu_seconds * 1e3 / steps });
    if (std.c.getenv("FZ_ONLY1") != null) {
        if (same < want.len) std.debug.print("first difference at token {d}: got {d}, Python {d}\n", .{ same, got.items[same], want[same].integer });
        return;
    }

    // 2. the Python engine's drafted windows, every row's pick, with rollback
    m.reset();
    if (!long_mode and std.c.getenv("FZ_ONLY17") == null) for (prompt) |tok| {
        try m.window(&.{@intCast(tok.integer)}, &pick);
        m.keepRows(&.{@intCast(tok.integer)}, 1);
    };
    var bad: usize = 0;
    var total_rows: usize = 0;
    const skip_mid = long_mode or std.c.getenv("FZ_ONLY17") != null; // the one-row reference, then GPU-side rounds only
    const rounds = if (own_prompt or skip_mid) &[_]std.json.Value{} else ref.object.get("rounds").?.array.items;
    for (rounds, 0..) |round, ri| {
        const o = round.object;
        const win = o.get("window").?.array.items;
        var tokens: [MAXR]u32 = undefined;
        for (win, 0..) |x, i| tokens[i] = @intCast(x.integer);
        try m.window(tokens[0..win.len], &pick);
        const exp = o.get("picks").?.array.items;
        for (exp, 0..) |x, i| {
            total_rows += 1;
            if (pick[i] != @as(u32, @intCast(x.integer))) {
                bad += 1;
                if (bad <= 5) std.debug.print("round {d} ({d} rows) row {d}: got {d}, Python {d}\n", .{ ri, win.len, i, pick[i], x.integer });
            }
        }
        m.keepRows(tokens[0..win.len], @intCast(o.get("keep").?.integer));
    }
    std.debug.print("drafted windows: {d} rounds, {d}/{d} rows equal to Python's\n", .{ rounds.len, total_rows - bad, total_rows });

    const ref_tokens: []const u32 = got.items;
    // 4. the MTP head against the Python engine's drafts (absorb windows of 1-8 rows, then a chain)
    if (!own_prompt and !skip_mid) {
        const mref = ref.object.get("mtp").?.object;
        var seq: std.ArrayList(u32) = .empty;
        for (prompt) |x| try seq.append(gpa, @intCast(x.integer));
        for (want[0..16]) |x| try seq.append(gpa, @intCast(x.integer));
        const all = try r.buffer(seq.items.len * WIDE * 2);
        m.reset();
        for (seq.items, 0..) |tok, i| {
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var ok: usize = 0;
        var n: usize = 0;
        var d: u32 = 0;
        for (mref.get("absorb").?.array.items) |a| {
            const start: usize = @intCast(a.object.get("start").?.integer);
            const nx = a.object.get("next").?.array.items;
            var nexts: [MAXR]u32 = undefined;
            for (nx, 0..) |x, i| nexts[i] = @intCast(x.integer);
            d = try m.mtpRun(nexts[0..nx.len], .{ .b = all, .off = start * WIDE * 2 });
            n += 1;
            if (d == @as(u32, @intCast(a.object.get("draft").?.integer))) ok += 1 else std.debug.print("absorb of {d} rows: draft {d}, Python {d}\n", .{ nx.len, d, a.object.get("draft").?.integer });
        }
        for (mref.get("chain").?.array.items) |x| {
            d = try m.mtpChain(d);
            n += 1;
            if (d == @as(u32, @intCast(x.integer))) ok += 1 else std.debug.print("chained draft {d}, Python {d}\n", .{ d, x.integer });
        }
        std.debug.print("MTP drafts equal to Python's: {d}/{d}\n", .{ ok, n });
    }

    // 5. one stream with MTP chains of a fixed depth: tokens against the one-row reference, lanes landed, tok/s
    const depths5: []const usize = if (own_prompt or skip_mid) &.{} else &.{ 1, 2, 3, 4, 6 };
    for (depths5) |depth| {
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        for (prompt, 0..) |x, i| {
            const tok: u32 = @intCast(x.integer);
            try m.window(&.{tok}, &pick);
            m.keepRows(&.{tok}, 1);
            @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
        }
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        const s0 = mtl.clock.seconds();
        var nexts: std.ArrayList(u32) = .empty;
        try nexts.appendSlice(gpa, ptoks[1..]);
        try nexts.append(gpa, pick[0]);
        var drafts: [MAXR]u32 = undefined;
        drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        var n_rounds: usize = 0;
        var landed: usize = 0;
        while (out.items.len < want.len) {
            var win: [MAXR]u32 = undefined;
            win[0] = out.items[out.items.len - 1];
            @memcpy(win[1 .. depth + 1], drafts[0..depth]);
            try m.window(win[0 .. depth + 1], &pick);
            var keep: usize = 1;
            while (keep <= depth and win[keep] == pick[keep - 1]) keep += 1;
            try out.appendSlice(gpa, pick[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            m.keepRows(win[0 .. depth + 1], keep);
            drafts[0] = try m.mtpAbsorb(pick[0..keep], m.last);
            for (1..depth) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        std.debug.print("depth {d}: {d}/{d} tokens equal; {d} rounds, {d:.2} tokens a round, {d:.2} of {d} drafts landing; {d:.1} tok/s\n", .{ depth, eq, want.len, n_rounds, @as(f64, @floatFromInt(out.items.len - 1)) / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), depth, @as(f64, @floatFromInt(out.items.len - 1)) / wall });
        if (eq < want.len) bad += 1;
    }

    // 6. one command buffer a round: the head absorbs the kept rows and chains its drafts into the next window's
    //    token slots on the GPU, the window hashes its n-grams on the GPU, and the host reads the picks once
    const wids: Buf = .{ .b = try r.buffer(64) };
    const adapt_story = [_][3]usize{ .{ 3, 3, 0 }, .{ 0, 7, 9 } };
    const adapt_own = [_][3]usize{ .{ 3, 3, 0 }, .{ 6, 6, 0 }, .{ 0, 7, 9 } };
    const adapt: []const [3]usize = if (own_prompt) &adapt_own else &adapt_story;
    const copy_min: usize = if (std.c.getenv("FZ_COPY_MIN")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 3;
    var hist: std.ArrayList(u32) = .empty;
    for (prompt) |x| try hist.append(gpa, @intCast(x.integer));
    const n_prompt = hist.items.len;
    for (0..if (skip_mid) 0 else if (r.copy) 2 else adapt.len) |run6| {
        const cfg_a = if (r.copy) adapt[0] else adapt[run6];
        const use_copy = r.copy and run6 == 1;
        var copy_rounds: usize = 0;
        var copy_landed: usize = 0;
        const ruled = cfg_a[2] == 9;
        var rule: DepthRule = .{};
        var depth: usize = if (ruled) rule.pick() else cfg_a[0];
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        const all = try r.buffer(prompt.len * WIDE * 2);
        var nexts: std.ArrayList(u32) = .empty;
        const p0 = mtl.clock.seconds();
        var at_p: usize = 0;
        var last_n: usize = 1;
        while (at_p < prompt.len) { // the prompt in windows of up to MAXR rows (each row's bits equal a one-row step)
            const n: usize = @min(MAXR, prompt.len - at_p);
            var toks: [MAXR]u32 = undefined;
            for (0..n) |i| toks[i] = @intCast(prompt[at_p + i].integer);
            try m.window(toks[0..n], &pick);
            m.keepRows(toks[0..n], n);
            @memcpy(all.contents()[at_p * WIDE * 2 .. (at_p + n) * WIDE * 2], m.last.b.contents()[m.last.off .. m.last.off + n * WIDE * 2]);
            for (0..n) |i| if (at_p + i > 0) try nexts.append(gpa, toks[i]);
            at_p += n;
            last_n = n;
        }
        const prefill_s = mtl.clock.seconds() - p0;
        const first = pick[last_n - 1];
        try nexts.append(gpa, first);
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, first);
        const w = wids.b.slice(u32, 16);
        var absorb_rows: []const u32 = nexts.items;
        var absorb_from: Buf = .{ .b = all };
        if (absorb_rows.len > MAXR) { // a long prompt: the head absorbs it before the rounds, a command buffer a chunk
            w[1] = try m.mtpAbsorb(absorb_rows, absorb_from);
            absorb_rows = &.{};
        }
        var n_rounds: usize = 0;
        var landed: usize = 0;
        const s0 = mtl.clock.seconds();
        m.gpu_seconds = 0;
        while (out.items.len < want.len) {
            w[0] = out.items[out.items.len - 1];
            var d = depth;
            var copied: usize = 0;
            if (use_copy) { // copy lanes: the window takes the tokens after an earlier match of its suffix
                hist.shrinkRetainingCapacity(n_prompt);
                try hist.appendSlice(gpa, out.items);
                copied = copyDrafts(hist.items, copy_min, w[1..MAXR]);
                if (copied > 0) d = copied;
            }
            const cb = r.queue.commandBuffer();
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            // the head: absorb the kept rows (chunks of up to MAXR), its draft into slot 1, then chain into 2..depth
            m.mtp.pos -= m.mtp.drafted;
            m.mtp.drafted = 0;
            var at: usize = 0;
            var slot: usize = 0;
            while (at < absorb_rows.len) : (slot += 1) {
                const n = @min(MAXR, absorb_rows.len - at);
                const sl = &m.mtp.slots[slot];
                const ids = sl.ids8.b.slice(u32, MAXR);
                for (0..MAXR) |i| ids[i] = if (i < n) absorb_rows[at + i] else 0;
                try m.mtpEncode(slot, n, sl.ids8, .{ .b = absorb_from.b, .off = absorb_from.off + at * WIDE * 2 }, .{ .b = wids.b, .off = if (copied > 0) 4 * 12 else 4 });
                m.mtp.pos += n;
                at += n;
            }
            if (copied == 0) for (1..depth) |j| {
                try m.mtpEncode(slot, 1, .{ .b = wids.b, .off = 4 * j }, m.mtp.last, .{ .b = wids.b, .off = 4 * (j + 1) });
                m.mtp.pos += 1;
                m.mtp.drafted += 1;
                slot += 1;
            };
            // the target window [pending, drafts]: with FZ_SPLIT the head's part is committed first and the window
            // is encoded while the GPU runs it (same queue, so the window still follows it)
            var wcb = cb;
            if (r.split) { // untracked buffers: the window waits on the head's signal, not on queue order alone
                r.enc.end();
                r.event_value += 1;
                cb.signal(r.event, r.event_value);
                cb.commit();
                wcb = r.queue.commandBuffer();
                wcb.waitFor(r.event, r.event_value);
                r.enc = wcb.compute(if (r.serial) .serial else .concurrent);
            }
            m.windowMeta(d + 1);
            m.pleIdsGpu(d + 1, wids);
            try m.windowEncode(d + 1, wids);
            try m.finish(wcb);
            if (r.split) m.gpu_seconds += cb.gpuSeconds();
            const picks = m.t.picks.b.slice(u32, d + 1);
            var keep: usize = 1;
            while (keep <= d and w[keep] == picks[keep - 1]) keep += 1;
            try out.appendSlice(gpa, picks[0..keep]);
            n_rounds += 1;
            landed += keep - 1;
            if (copied > 0) {
                copy_rounds += 1;
                copy_landed += keep - 1;
            }
            var win: [MAXR]u32 = undefined;
            @memcpy(win[0 .. d + 1], w[0 .. d + 1]);
            m.keepRows(win[0 .. d + 1], keep);
            @memcpy(pick[0..keep], picks[0..keep]);
            absorb_rows = pick[0..keep];
            absorb_from = m.last;
            if (ruled) {
                rule.update(d, keep - 1);
                depth = rule.pick();
            } else if (cfg_a[0] != cfg_a[1]) depth = @max(cfg_a[0], @min(cfg_a[1], keep - 1 + cfg_a[2]));
        }
        const wall = mtl.clock.seconds() - s0;
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == (if (r.xnew) ref_tokens[eq] else @as(u32, @intCast(want[eq].integer)))) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        std.debug.print("one buffer a round, depth {d}-{d} (+{d}){s}: {d}/{d} tokens equal; {d:.2} tokens a round, {d:.2} drafts landing; {d:.1} tok/s (GPU busy {d:.0}%)\n", .{ cfg_a[0], cfg_a[1], cfg_a[2], if (use_copy) " + copy lanes" else "", eq, want.len, made / @as(f64, @floatFromInt(n_rounds)), @as(f64, @floatFromInt(landed)) / @as(f64, @floatFromInt(n_rounds)), made / wall, 100 * m.gpu_seconds / wall });
        std.debug.print("  prompt {d} tokens in {d:.2} s ({d:.0} tok/s){s}\n", .{ prompt.len, prefill_s, @as(f64, @floatFromInt(prompt.len)) / prefill_s, if (ruled) ", drafts chosen each round by the depth rule" else "" });
        if (use_copy) std.debug.print("  copy rounds {d} of {d}, {d:.2} copied lanes landing a copy round\n", .{ copy_rounds, n_rounds, @as(f64, @floatFromInt(copy_landed)) / @as(f64, @floatFromInt(@max(copy_rounds, 1))) });
        if (eq < want.len) bad += 1;
    }

    // 7. GPU-side rounds: the verdict, positions, history and kept states stay on the GPU; the host encodes round
    //    N+1 while round N runs and reads the emitted tokens from a ring. Depth 0: the depth rule picks each round's
    //    drafts from the rounds read back so far (the window two rounds behind the one being encoded).
    var depths7: std.ArrayList(usize) = .empty;
    if (std.c.getenv("FZ_D7")) |v| { // e.g. FZ_D7=6,0
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |d| try depths7.append(gpa, try std.fmt.parseInt(usize, d, 10));
    } else try depths7.appendSlice(gpa, &.{ 3, 6, 0 });
    if (r.xnew) for (depths7.items) |depth_cfg| {
        const ruled = depth_cfg == 0;
        var rule: DepthRule = .{};
        const depth0 = if (ruled) rule.pick() else depth_cfg;
        const n_lin: usize = 36;
        const g_cs = try r.buffer(n_lin * CS_ROW);
        const g_so = try r.buffer(n_lin * SO_ROW);
        const o_cs = try r.buffer(n_lin * MAXR * CS_ROW);
        const o_so = try r.buffer(n_lin * MAXR * SO_ROW);
        var saved: [LAYERS][4]Buf = undefined;
        var gi: usize = 0;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            saved[i] = .{ L.cs[0], L.cs[1], L.so[0], L.so[1] };
            L.cs[0] = .{ .b = g_cs, .off = gi * CS_ROW };
            L.cs[1] = .{ .b = o_cs, .off = gi * MAXR * CS_ROW };
            L.so[0] = .{ .b = g_so, .off = gi * SO_ROW };
            L.so[1] = .{ .b = o_so, .off = gi * MAXR * SO_ROW };
            gi += 1;
        };
        const cin_saved = m.ple.cin;
        const cins = [2]Buf{ m.ple.cin, .{ .b = try r.buffer((PLE_TAIL + MAXR) * WIDE * 2) } };
        r.ar = .{ .b = try r.buffer(4 * fz.AR_WORDS) };
        const ring = try r.buffer(fz.RING_WORDS * 4 * 512);
        const g_hist = try r.buffer((CAP + 64) * 4);
        const ar = r.ar.b.slice(i32, fz.AR_WORDS);
        const Copy = struct { // the kept row of every DeltaNet layer's window output into its state
            fn states(rr: *Run, gcs: mtl.Buffer, gso: mtl.Buffer, ocs: mtl.Buffer, oso: mtl.Buffer) void {
                rr.copyKept(.{ .b = oso }, .{ .b = gso }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
                rr.copyKept(.{ .b = ocs }, .{ .b = gcs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
            }
        };
        m.reset();
        m.mtp.pos = 0;
        m.mtp.drafted = 0;
        var drafts: [MAXR]u32 = undefined;
        const p0 = mtl.clock.seconds();
        if (pr_long) |*pr| { // prompt chunks; the head's keys for every prompt row, its layer on the last
            var at: usize = 0;
            var last_n: usize = 1;
            while (at < ptoks.len) {
                const n = @min(pr.step, ptoks.len - at);
                pick[0] = try pr.chunk(m, gpa, ptoks[at .. at + n]);
                const k = if (at + n < ptoks.len) n else n - 1;
                try pr.mtpKeys(m, at, ptoks[at + 1 .. at + 1 + k], m.last);
                at += n;
                last_n = n;
            }
            if (m.state == 1) { // the state into the rounds' state buffers
                ar[0] = 1;
                const cb = r.queue.commandBuffer();
                r.enc = cb.compute(if (r.serial) .serial else .concurrent);
                Copy.states(&r, g_cs, g_so, o_cs, o_so);
                try m.finish(cb);
                m.state = 0;
                m.state_row = 0;
            }
            m.mtp.pos = ptoks.len - 1;
            drafts[0] = try m.mtpRun(&.{pick[0]}, .{ .b = m.last.b, .off = m.last.off + (last_n - 1) * WIDE * 2 });
        } else {
            const all = try r.buffer(prompt.len * WIDE * 2);
            var nexts: std.ArrayList(u32) = .empty;
            for (prompt, 0..) |x, i| {
                const tok: u32 = @intCast(x.integer);
                try m.window(&.{tok}, &pick);
                ar[0] = 1;
                const cb = r.queue.commandBuffer();
                r.enc = cb.compute(if (r.serial) .serial else .concurrent);
                Copy.states(&r, g_cs, g_so, o_cs, o_so);
                try m.finish(cb);
                const cin = m.ple.cin.b.contents()[m.ple.cin.off..];
                std.mem.copyForwards(u8, cin[0 .. PLE_TAIL * WIDE * 2], cin[WIDE * 2 .. (PLE_TAIL + 1) * WIDE * 2]);
                m.ple.hist = .{ m.ple.hist[1], tok };
                m.pos += 1;
                @memcpy(all.contents()[i * WIDE * 2 .. (i + 1) * WIDE * 2], m.last.b.contents()[0 .. WIDE * 2]);
                if (i > 0) try nexts.append(gpa, tok);
            }
            try nexts.append(gpa, pick[0]);
            drafts[0] = try m.mtpAbsorb(nexts.items, .{ .b = all });
        }
        for (1..depth0) |j| drafts[j] = try m.mtpChain(drafts[j - 1]);
        const first_s = mtl.clock.seconds() - p0;
        const w = wids.b.slice(u32, 16);
        w[0] = pick[0];
        for (0..depth0) |j| w[1 + j] = drafts[j];
        const W0 = depth0 + 1;
        const T: i32 = @intCast(m.pos);
        @memset(ar, 0);
        ar[1] = T;
        ar[fz.AR_HLEN] = @intCast(prompt.len + 1); // the accept kernel appends emitted tokens to the history
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
        const t_saved = .{ m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta };
        var rows_w: [MAXR + 1]Buf = undefined; // the window's row-count buffers, one set a width
        var mdims_w: [MAXR + 1]Buf = undefined;
        for (1..MAXR + 1) |n| {
            rows_w[n] = try i32Buf(&r, &.{@intCast(n)});
            mdims_w[n] = try i32Buf(&r, &.{ @intCast(n), 16, 0, 0, 0, 0, 0, 0 });
        }
        m.t.pos8 = .{ .b = r.ar.b, .off = fz.AR_POS * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = fz.AR_NK * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = fz.AR_KV * 4 };
        const slots_saved = m.mtp.slots;
        for (1..MAXR + 1) |n| { // the head absorbing a window of n rows: slot MAXR - 1 + n
            const sl = &m.mtp.slots[MAXR - 1 + n];
            try m.mtpMeta(sl, n);
            sl.pos8 = .{ .b = r.ar.b, .off = fz.AR_ABS_POS * 4 };
            sl.nk8 = .{ .b = r.ar.b, .off = fz.AR_ABS_NK * 4 };
            sl.kvmeta = .{ .b = r.ar.b, .off = fz.AR_ABS_KV * 4 };
        }
        for (1..MAXR - 1) |j| { // chained draft j: slot j
            try m.mtpMeta(&m.mtp.slots[j], 1);
            const b = (fz.AR_CHAIN + (j - 1) * fz.AR_CHAIN_STRIDE) * 4;
            m.mtp.slots[j].pos8 = .{ .b = r.ar.b, .off = b };
            m.mtp.slots[j].nk8 = .{ .b = r.ar.b, .off = b + 16 * 4 };
            m.mtp.slots[j].kvmeta = .{ .b = r.ar.b, .off = b + 32 * 4 };
        }
        if (r.sel != null and prompt.len + want.len + 8 > 4 * TOP) { // long context: selection on the GPU
            const cb0 = r.queue.commandBuffer();
            r.enc = cb0.compute(if (r.serial) .serial else .concurrent);
            try r.sel.?.catchUp(&r, m, m.pos / 4); // the target's layers and the head's absorbed rows
            try m.finish(cb0);
            r.gsel = try GSelect.init(&r, m.pos / 4);
            m.mtp.gsel = try GSelect.init(&r, m.pos / 4);
        }
        r.gpu_round = true;
        const base = r.event_value;
        var cbs: std.ArrayList(mtl.CommandBuffer) = .empty;
        var widths: std.ArrayList(usize) = .empty; // each round's window rows (decided a round ahead)
        try widths.append(gpa, W0);
        var out: std.ArrayList(u32) = .empty;
        try out.append(gpa, pick[0]);
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        var s1 = s0;
        var round: usize = 0;
        var done: usize = 0;
        var w_sum: usize = 0;
        var landed: usize = 0;
        var offered: usize = 0;
        var at_w: [MAXR + 1]usize = @splat(0);
        const rg = ring.slice(u32, fz.RING_WORDS * 512);
        while (true) {
            const wr = widths.items[round];
            const cb = r.queue.commandBuffer();
            if (round > 0) cb.waitFor(r.event, base + round);
            r.enc = cb.compute(if (r.serial) .serial else .concurrent);
            m.t.rows = rows_w[wr];
            m.t.mdims = mdims_w[wr];
            if (round > 0) {
                const wp = widths.items[round - 1];
                Copy.states(&r, g_cs, g_so, o_cs, o_so);
                r.copyKept(cins[(round - 1) % 2], cins[round % 2], PLE_TAIL * WIDE / 2, WIDE / 2, 0, 0, 1, 0);
                try m.mtpEncode(MAXR - 1 + wp, wp, m.t.picks, m.last, .{ .b = wids.b, .off = 4 });
                for (1..wr - 1) |j| try m.mtpEncode(j, 1, .{ .b = wids.b, .off = 4 * j }, m.mtp.h[1], .{ .b = wids.b, .off = 4 * (j + 1) });
            }
            m.ple.cin = cins[round % 2];
            m.pleIdsGpu(wr, wids);
            w_sum += wr;
            if (r.gsel) |*g| g.nb_ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            if (m.mtp.gsel) |*g| g.nb_ub = (@as(usize, @intCast(T)) + w_sum) / 4 + 2;
            try m.windowEncode(wr, wids);
            const wn = (if (ruled) rule.pick() else depth_cfg) + 1;
            try widths.append(gpa, wn);
            const cfg = [4]u32{ @intCast(wr), @intCast(wn), CAP, 0 };
            r.enc.setPipeline(r.accept_pipe);
            for ([_]Buf{ wids, m.t.picks, r.ar, .{ .b = ring }, m.t.ple_meta }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
            r.enc.setBytes(std.mem.asBytes(&cfg), 5);
            r.enc.setBuffer(g_hist, 0, 6);
            r.enc.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
            r.enc.end();
            cb.signal(r.event, base + round + 1);
            cb.commit();
            try cbs.append(gpa, cb);
            round += 1;
            if (round >= 2) {
                const prev = cbs.items[done];
                prev.wait();
                if (prev.failure()) |msg| {
                    std.log.err("command buffer failed: {s}", .{msg});
                    return error.GpuFailed;
                }
                m.gpu_seconds += prev.gpuSeconds();
                const slot = (done % 512) * fz.RING_WORDS;
                const keep = rg[slot];
                try out.appendSlice(gpa, rg[slot + 3 .. slot + 3 + keep]);
                const wd = widths.items[done];
                if (ruled) rule.update(wd - 1, keep - 1);
                landed += keep - 1;
                offered += wd - 1;
                at_w[wd] += 1;
                done += 1;
                if (out.items.len >= want.len) {
                    s1 = mtl.clock.seconds();
                    break;
                }
            }
        }
        for (cbs.items[done..]) |cb| cb.wait();
        r.event_value = base + round + 1;
        r.gpu_round = false;
        r.gsel = null;
        m.mtp.gsel = null;
        m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta = t_saved;
        m.mtp.slots = slots_saved;
        m.ple.cin = cin_saved;
        for (&m.layers, 0..) |*L, i| if (L.linear) {
            L.cs[0], L.cs[1], L.so[0], L.so[1] = saved[i];
        };
        var eq: usize = 0;
        while (eq < want.len and out.items[eq] == ref_tokens[eq]) eq += 1;
        const made: f64 = @floatFromInt(out.items.len - 1);
        const nd: f64 = @floatFromInt(done);
        if (ruled) {
            std.debug.print("GPU-side rounds, depth rule", .{});
            for (at_w, 0..) |c, n| if (c > 0) std.debug.print(" {d} rounds at {d} drafts", .{ c, n - 1 });
        } else std.debug.print("GPU-side rounds, depth {d}", .{depth_cfg});
        std.debug.print(": {d}/{d} tokens equal; {d:.2} tokens a round, {d} of {d} drafts landed; {d:.1} tok/s (GPU busy {d:.0}%); prompt to first drafts {d:.2} s\n", .{ eq, want.len, made / nd, landed, offered, made / (s1 - s0), 100 * m.gpu_seconds / (s1 - s0), first_s });
        if (eq < want.len) bad += 1;
    };

    // 3. each window size's cost
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = got.items[i];
        m.gpu_seconds = 0;
        const s0 = mtl.clock.seconds();
        for (0..10) |_| try m.window(toks[0..rows], &pick);
        const wall = (mtl.clock.seconds() - s0) * 1e2;
        std.debug.print("window of {d} rows: {d:.2} ms (GPU {d:.2} ms)\n", .{ rows, wall, m.gpu_seconds * 1e2 });
    }
    if (same != want.len or bad != 0) std.process.exit(1);
}
