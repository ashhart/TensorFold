//! `tf-dsv41 forward-digests PACK REF OUT`: our whole forward on one GPU (TP=2 rank 0) against REF's digests.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const forms = @import("forms.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const Value = std.json.Value;

/// The widths block.zig needs, from our loaded weights (each trellis's words) and the plan's expert layouts.
pub fn widths(a: std.mem.Allocator, w: *const load.Weights, p: *const plan.Plan) !block.Widths {
    var out: block.Widths = .{};
    var it = w.map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (!std.mem.endsWith(u8, name, ".trellis") or e.value_ptr.rank != 3) continue;
        try out.dense.put(a, name[0 .. name.len - ".trellis".len], @intCast(e.value_ptr.shape[2] / 8));
    }
    for (p.layers) |*l| {
        var r: [2][2]u32 = .{ .{ 99, 0 }, .{ 99, 0 } };
        for ([_]usize{ 0, 2, 1 }, [_]usize{ 0, 0, 1 }) |proj, slot| {
            for (l.experts[proj].layout.words) |words| {
                r[slot][0] = @min(r[slot][0], words / 8);
                r[slot][1] = @max(r[slot][1], words / 8);
            }
            const sh = out.dense.get(try std.fmt.allocPrint(a, "L{d}.moe.shared.0.{s}", .{ l.index, ([_][]const u8{ "w1", "w2", "w3" })[proj] })) orelse return error.NoSharedExpert;
            r[slot][0] = @min(r[slot][0], sh);
            r[slot][1] = @max(r[slot][1], sh);
        }
        try out.experts.put(a, l.index, r);
        // x3gm's widths present: the routed experts' gate (w1) and down (w2) K2s, the shared expert not among them
        var gm: [2]u32 = .{ 0, 0 };
        for (0..2) |proj| for (l.experts[proj].layout.words) |words| {
            gm[proj] |= @as(u32, 1) << @intCast(words / 8);
        };
        try out.gm.put(a, l.index, gm);
    }
    return out;
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, ref: []const u8, out_dir: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out_dir);
    const meta = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "digests.json" }), a, .limited(1 << 28)), .{});
    const ls_json = meta.object.get("layers").?.array.items;
    const layers = try a.alloc(u32, ls_json.len);
    for (ls_json, layers) |x, *y| y.* = @intCast(x.integer);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var kernels = try dk.Kernels.load(&ctx);
    defer kernels.deinit();

    // our loader: the reference's layers + the vocabulary rows, norm and head, rank 0 of 2
    const t0 = std.Io.Clock.awake.now(io);
    const cfg = try Config.read(gpa, io, pack_dir);
    var pack = try Pack.open(gpa, io, pack_dir);
    defer pack.deinit();
    var pl = try plan.build(gpa, &cfg, &pack, .{ .rank = 0, .world = 2, .blocks = layers, .top = true });
    defer pl.deinit();
    var w: load.Weights = .{ .gpa = gpa };
    defer @import("close_or_exit.zig").finish(&w);
    for (pl.layers) |*l| {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.layer(l, cfg.o_groups, 0, 2);
        try w.upload(&driver, &b);
    }
    {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = &pack };
        defer b.deinit();
        try b.top(&pl);
        try w.upload(&driver, &b);
    }
    std.debug.print("loaded {d} layers, {d:.2} GB\n", .{ layers.len, @as(f64, @floatFromInt(w.bytes)) / 1e9 });

    var aot = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ ref, "aot" }));
    defer aot.deinit();
    try @import("aot_trace.zig").attach(gpa, io, &aot, @import("aot_trace.zig").pathFromEnv());
    const wd = try widths(a, &w, &pl);
    var runner: run.Runner = .{ .gpa = gpa, .d = &driver, .stream = stream, .kernels = &kernels, .triton = &aot, .weights = &w };
    defer runner.deinit();
    var zp: fwd.ZeroPeer = .{ .d = &driver };
    var f = fwd.Forward.init(gpa, &cfg, &wd, .{ .trace = true }, &runner, zp.collective());
    defer @import("close_or_exit.zig").finish(&f);
    try @import("engine_hooks.zig").open(&f);
    f.layers = layers;
    const buckets = [_]u32{ 1, 3, 16 };
    var bp = try f.plan(&buckets);
    try runner.bind(&bp);

    // the persistent roles' initial bytes: the weight forms, the rope tables; the rest (pools, scratch) stays zero
    const src: forms.Source = .{ .d = &driver, .weights = &w, .plan = &pl };
    var formed: usize = 0;
    for (bp.sizes.keys()) |role| {
        if (buffers.scopeOf(role) != .persistent) continue;
        var ra = std.heap.ArenaAllocator.init(gpa);
        defer ra.deinit();
        const bytes = (try forms.build(ra.allocator(), &src, role)) orelse continue;
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = runner.addressOf(role).?, .len = bytes.len }, 0, bytes);
        formed += 1;
    }
    const rope = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "rope.bin" }), a, .limited(1 << 28));
    if (!std.mem.eql(u8, rope[0..8], "DSV41RP1")) return error.BadRope;
    const half = (rope.len - 16) / 2;
    for ([_][]const u8{ "s.rope.main", "s.rope.comp" }, 0..) |role, i| if (runner.addressOf(role)) |p|
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = p, .len = half }, 0, rope[16 + i * half ..][0..half]);
    const ehb = try cwd.readFileAllocOptions(io, try std.fs.path.join(a, &.{ ref, "engram-token-map.bin" }), a, .limited(1 << 26), .@"4", null);
    var lbuf: [eh.max_layers]u32 = undefined;
    const host = try eh.Host.load(ehb, cfg.engram_heads, cfg.engram_pad, cfg.engram_vocab, &lbuf);
    f.engram = &host;
    try driver.synchronize();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    std.debug.print("bound {d} roles ({d} weight forms), load {d:.1} s\n", .{ bp.sizes.count(), formed, load_s });

    // the slot after the reference's prompt (its own prefill): the stores into our roles, the position and tail
    const ids_json = meta.object.get("ids").?.array.items;
    const ids = try a.alloc(u32, ids_json.len);
    for (ids_json, ids) |x, *y| y.* = @intCast(x.integer);
    const st = try std.json.parseFromSliceLeaky(Value, a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "state", "state.json" }), a, .limited(1 << 24)), .{});
    var sit = st.object.get("roles").?.object.iterator();
    var loaded: usize = 0;
    while (sit.next()) |e| {
        const bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, "state", e.value_ptr.string }), a, .limited(1 << 32));
        const at = runner.addressOf(e.key_ptr.*) orelse return error.StateRoleNotInPlan;
        if (bp.sizes.get(e.key_ptr.*).? < bytes.len) return error.StateTooLarge;
        try cuda.DeviceBuffer.upload(.{ .d = &driver, .ptr = at, .len = bytes.len }, 0, bytes);
        loaded += 1;
    }
    f.slot.pos = @intCast(st.object.get("pos").?.integer);
    const tail = st.object.get("tail").?.array.items;
    for (tail, 0..) |t, i| f.slot.tail[i] = @intCast(t.integer);
    f.slot.tail_len = tail.len;
    try driver.synchronize();
    std.debug.print("state: {d} stores from the reference's prompt, position {d}\n", .{ loaded, f.slot.pos });
    var pos: usize = @intCast(meta.object.get("prompt").?.integer);
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    var bad: usize = 0;
    var trace: std.AutoArrayHashMapUnmanaged(u32, [64]u8) = .empty;
    defer trace.deinit(gpa);
    f.trace = &trace;
    for (meta.object.get("windows").?.array.items) |wv| {
        const wo = wv.object;
        const n: usize = @intCast(wo.get("n").?.integer);
        trace.clearRetainingCapacity();
        try f.window(ids[pos .. pos + n]);
        try stream.synchronize();
        var equal: usize = 0;
        var first: ?u32 = null;
        var it = wo.get("layers").?.object.iterator();
        while (it.next()) |e| {
            const L = try std.fmt.parseInt(u32, e.key_ptr.*, 10);
            const got = trace.get(L) orelse return error.TraceMissing;
            if (std.mem.eql(u8, &got, e.value_ptr.string)) equal += 1 else if (first == null) {
                first = L;
            }
        }
        // the logits: this rank's [n, V/2] fp32, element by element
        const lg = wo.get("logits").?.object;
        const want = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ ref, lg.get("file").?.string }), a, .limited(1 << 30));
        const got = try a.alloc(u8, want.len);
        try cuda.DeviceBuffer.download(.{ .d = &driver, .ptr = runner.addressOf("w.logits").?, .len = got.len }, 0, got);
        var diff: usize = 0;
        for (std.mem.bytesAsSlice(u32, got), std.mem.bytesAsSlice(u32, want)) |x, y| diff += @intFromBool(x != y);
        const layers_n = wo.get("layers").?.object.count();
        const ok = equal == layers_n and diff == 0;
        if (!ok) bad += 1;
        try log.writer.print("{{\"n\": {d}, \"layers_equal\": {d}, \"layers\": {d}, \"first_differing_layer\": {?d}, \"logits_differ\": {d}, \"logits\": {d}}}\n", .{ n, equal, layers_n, first, diff, want.len / 4 });
        std.debug.print("window {d} rows: {d}/{d} blocks equal (first differing {?d}), logits {d} of {d} differ\n", .{ n, equal, layers_n, first, diff, want.len / 4 });
        try f.keep(@intCast(n - 1));
        pos += n;
    }
    f.trace = null;
    try cwd.writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "forward-digests.jsonl" }), .data = log.written() });
    std.debug.print("{s} forward digests (layers {d}-{d}, one GPU, zero peer)\n", .{ if (bad == 0) "PASS" else "FAIL", layers[0], layers[layers.len - 1] });
    return if (bad == 0) 0 else 1;
}
