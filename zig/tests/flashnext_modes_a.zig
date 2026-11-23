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

// every prompt-kernel source and the 6-bit file on this Metal compiler
pub fn compileCheck(device: mtl.Device, gpa: std.mem.Allocator, arena: std.mem.Allocator) !void {
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

// the tensor units' rate on register fragments: 16x32x16 ops a second
pub fn mmaPeak(device: mtl.Device, arena: std.mem.Allocator) !void {
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

// the served engine: its own load, warm-up and reply, on FZ_REF's prompt
pub fn engine(init: std.process.Init, gpa: std.mem.Allocator, arena: std.mem.Allocator, args: anytype) !void {
    const fx = @import("tensorfold").flashnext_engine;
    const rp = std.mem.span(platform.c.getenv("FZ_REF") orelse return error.NoRef);
    const rf = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(arena, "{s}", .{rp}, 0));
    const rj = try std.json.parseFromSliceLeaky(std.json.Value, arena, rf.bytes[0..rf.size], .{});
    const all_items = rj.object.get("prompt").?.array.items;
    const items = all_items[0..@min(all_items.len, try promptCut())];
    const toks = try arena.alloc(u32, items.len);
    for (items, 0..) |x, i| toks[i] = @intCast(x.integer);
    const e = try fx.Engine.load(gpa, init.io, args[1], args[2]);
    if (platform.c.getenv("FZ_COPY_MIN")) |v| e.copy_min = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (platform.c.getenv("FZ_COPY_LONG")) |v| e.copy_long = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (platform.c.getenv("FZ_ENGINE_WARM") != null) try e.warm();
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
    const n_out: usize = if (platform.c.getenv("FZ_N")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 256;
    var replies: [2][]u32 = undefined;
    const seg_ab = platform.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(platform.c.getenv("FZ_AB").?), "seg");
    var depths: std.ArrayList(?usize) = .empty; // FZ_DEPTHS=3,5,7: each depth's arms in turn (0: the depth rule, p: plain)
    if (platform.c.getenv("FZ_DEPTHS")) |v| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |d| {
            if (std.mem.eql(u8, d, "p")) {
                try depths.append(arena, 0);
                continue;
            }
            const n = try std.fmt.parseInt(usize, d, 10);
            try depths.append(arena, if (n == 0) null else n);
        }
    } else try depths.append(arena, if (platform.c.getenv("FZ_DEPTH")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null);
    // FZ_AB=fz (with FZ_LANE=1 FZ_GDN=2): recorded dense and DeltaNet kernels, then core lane, fz_gdn, kept states
    const fz_ab = platform.c.getenv("FZ_AB") != null and std.mem.eql(u8, std.mem.span(platform.c.getenv("FZ_AB").?), "fz");
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
    if (platform.c.getenv("FZ_NO_REF") != null) return; // the reply hashes are the check (two Macs against one)
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
