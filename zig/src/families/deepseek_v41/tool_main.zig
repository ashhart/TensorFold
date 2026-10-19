//! `tf-dsv41`: DeepSeek-V4.1's tool: captured blocks replayed bit for bit, host checks, GPU checks, the served engine.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const tri = @import("triton_capture.zig");
const gr = @import("gate_replay.zig");
const check = @import("calls_check.zig");
const fwd = @import("gate_layers.zig");
const run = @import("run.zig");

// the forward's executor is type-checked with this tool
comptime {
    _ = &run.Runner.bind;
    _ = &run.Runner.window;
    _ = &@import("forward.zig").Forward.window;
    _ = &@import("forward.zig").Forward.keep;
    _ = &@import("forward.zig").Forward.plan;
    _ = &@import("forward.zig").ZeroPeer.collective;
    _ = &@import("target.zig").GpuTarget.target;
    _ = &named.Builder.dspark;
}

const usage =
    \\usage: tf-dsv41 replay|forward|both PACK CAPTURE OUT
    \\       tf-dsv41 check CAPTURE   (host only: block.zig's calls against the capture's launches)
    \\       tf-dsv41 plan            (host only: the whole model emitted from the K2 width listings, its buffer plan)
    \\       tf-dsv41 kv-emit CAPTURE (host only: the pool / split programs' invariants for every captured window)
    \\       tf-dsv41 engram-probe DIR RANK WORLD (the Engram read path on a node's shards: device latency at depth
    \\                   1 / 8 / 32 / 64, the serving pool's batch times, idle and with spinning cores; no GPU)
    \\       tf-dsv41 aot-needs SIGS OUT (host only: every Triton variant the served config (env) can launch, for
    \\                   recipe/triton_fill.py compile; SIGS from its `sigs`)
    \\       tf-dsv41 kv paged|split|compact|park PACK REF OUT (a rank of TP=2 on the paged pool vs the pool reference)
    \\       tf-dsv41 forward-digests PACK REF OUT (our forward vs recipe/dsv41_forward_digests.py's run, one GPU)
    \\       tf-dsv41 decode-tp PACK REF OUT (a rank of TP=2 over NCCL vs recipe/dsv41_decode_ref.py; TF_TP_RANK ...)
    \\       tf-dsv41 follow PACK ASSETS (a served model's rank > 0: rank 0's forward operations until it stops)
    \\       tf-dsv41 generate PACK ASSETS TRACE (rank 0: TRACE's prompts through the served engine, ids == recorded)
    \\       tf-dsv41 slot-drafts PACK ASSETS TRACE... (rank 0: drafts over several live slots, batched == alone == recorded)
    \\  PACK     the EXL3 pack (config.json, index, shards)
    \\  CAPTURE  recipe/dsv41_triton_capture.py's output (ops.jsonl, blobs/, weights.json, aot/, meta.json)
    \\  recipe/  the reference tools, tools/tensorfold-dsv41/ of github.com/jayleaton/deepseek-v41-tensorfold-spark
    \\  OUT      weights.jsonl, replay.jsonl, forward.jsonl, summary.json
    \\replay: every captured launch re-issued and compared, alone and chained; forward: block.zig's calls for every
    \\captured decode window on our own buffers (gate_layers.zig); both: one load, the replay then the forward
    \\exit 0: every weight equal and every replayed op equal alone (skipped ops are listed, not failures)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 5 and std.mem.eql(u8, args[1], "decode-tp")) {
        return @import("gate_decode_tp.zig").main(init.gpa, init.io, args[2], args[3], args[4]) catch |e| {
            std.debug.print("FAIL tf-dsv41 decode-tp: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len == 6 and std.mem.eql(u8, args[1], "kv")) {
        return @import("gate_kv.zig").main(init.gpa, init.io, args[2], args[3], args[4], args[5]) catch |e| {
            std.debug.print("FAIL tf-dsv41 kv: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len == 5 and std.mem.eql(u8, args[1], "forward-digests")) {
        return @import("gate_forward.zig").main(init.gpa, init.io, args[2], args[3], args[4]) catch |e| {
            std.debug.print("FAIL tf-dsv41 forward-digests: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len == 2 and std.mem.eql(u8, args[1], "plan")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &buf);
        check.planAll(init.gpa, @embedFile("fixtures/dense-k2.txt"), @embedFile("fixtures/expert-k2.txt"), &out.interface) catch |e| {
            std.debug.print("FAIL tf-dsv41 plan: {t}\n", .{e});
            return 1;
        };
        out.interface.flush() catch {};
        return 0;
    }
    if (args.len == 4 and std.mem.eql(u8, args[1], "follow"))
        return @import("serve_cli.zig").follow(init.gpa, init.io, args[2], args[3]) catch |e| {
            std.debug.print("FAIL tf-dsv41 follow: {t}\n", .{e});
            return 1;
        };
    if (args.len == 5 and std.mem.eql(u8, args[1], "slots"))
        return @import("slots_cli.zig").main(init.gpa, init.io, args[2], args[3], args[4]) catch |e| {
            std.debug.print("FAIL tf-dsv41 slots: {t}\n", .{e});
            return 1;
        };
    if (args.len >= 6 and std.mem.eql(u8, args[1], "session-slots")) {
        // OUT REF...: sessions over several live slots (recipe/dsv41_session_slots/golden.py's references)
        const refs = try init.arena.allocator().alloc([]const u8, args.len - 5);
        for (refs, args[5..]) |*t, x| t.* = x;
        return @import("session_slots_cli.zig").main(init.gpa, init.io, args[2], args[3], args[4], refs) catch |e| {
            std.debug.print("FAIL tf-dsv41 session-slots: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len >= 5 and std.mem.eql(u8, args[1], "slot-drafts")) {
        // TRACE...: the several-slot drafting check
        const traces = try init.arena.allocator().alloc([]const u8, args.len - 4);
        for (traces, args[4..]) |*t, x| t.* = x;
        return @import("slot_drafts_cli.zig").main(init.gpa, init.io, args[2], args[3], traces) catch |e| {
            std.debug.print("FAIL tf-dsv41 slot-drafts: {t}\n", .{e});
            return 1;
        };
    }
    if (args.len == 5 and std.mem.eql(u8, args[1], "generate"))
        return @import("serve_cli.zig").generate(init.gpa, init.io, args[2], args[3], args[4]) catch |e| {
            std.debug.print("FAIL tf-dsv41 generate: {t}\n", .{e});
            return 1;
        };
    if (args.len == 3 and std.mem.eql(u8, args[1], "dcheck")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &buf);
        const bad = @import("dspark_emit.zig").checkDraft(init.gpa, init.io, args[2], @embedFile("fixtures/expert-k2.txt"), &out.interface, 12) catch |e| {
            std.debug.print("FAIL tf-dsv41 dcheck: {t}\n", .{e});
            return 1;
        };
        out.interface.print("dcheck: {d} mismatches\n", .{bad}) catch {};
        out.interface.flush() catch {};
        return if (bad == 0) 0 else 1;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "kv-emit")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &buf);
        const bad = @import("kv_emit_check.zig").checkAll(init.gpa, init.io, args[2], @embedFile("fixtures/expert-k2.txt"), &out.interface) catch |e| {
            std.debug.print("FAIL tf-dsv41 kv-emit: {t}\n", .{e});
            return 1;
        };
        out.interface.flush() catch {};
        return if (bad == 0) 0 else 1;
    }
    if (args.len == 5 and std.mem.eql(u8, args[1], "engram-probe")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &buf);
        const rank = std.fmt.parseInt(u32, args[3], 10) catch return 2;
        const world = std.fmt.parseInt(u32, args[4], 10) catch return 2;
        @import("engram_probe.zig").main(init.gpa, args[2], rank, world, &out.interface) catch |e| {
            std.debug.print("FAIL tf-dsv41 engram-probe: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (args.len == 4 and std.mem.eql(u8, args[1], "aot-needs")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stderr().writer(init.io, &buf);
        @import("aot_needs.zig").main(init.gpa, init.io, args[2], args[3], &out.interface) catch |e| {
            std.debug.print("FAIL tf-dsv41 aot-needs: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "check")) {
        var buf: [4096]u8 = undefined;
        var out = std.Io.File.stdout().writer(init.io, &buf);
        const bad = check.checkAll(init.gpa, init.io, args[2], @embedFile("fixtures/expert-k2.txt"), &out.interface, 12) catch |e| {
            std.debug.print("FAIL tf-dsv41 check: {t}\n", .{e});
            return 1;
        };
        out.interface.print("check: {d} mismatches\n", .{bad}) catch {};
        out.interface.flush() catch {};
        return if (bad == 0) 0 else 1;
    }
    const mode = std.meta.stringToEnum(Mode, if (args.len > 1) args[1] else "") orelse .none;
    if (args.len != 5 or mode == .none) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    return replay(init.gpa, init.io, args[2], args[3], args[4], mode) catch |e| {
        std.debug.print("FAIL tf-dsv41: {t}\n", .{e});
        return 1;
    };
}

/// The captured blocks' layer indices (meta.json "sets"), ascending.
fn layers(a: std.mem.Allocator, io: std.Io, capture: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ capture, "meta.json" }), a, .limited(1 << 24));
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    var out: std.ArrayList(u32) = .empty;
    var it = root.object.get("sets").?.object.iterator();
    while (it.next()) |e| for (e.value_ptr.object.get("layers").?.array.items) |l| try out.append(a, @intCast(l.integer));
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    // sets may share a block (P and B both hold layer 20): each loads once
    var k: usize = 0;
    for (out.items) |l| {
        if (k > 0 and out.items[k - 1] == l) continue;
        out.items[k] = l;
        k += 1;
    }
    return out.items[0..k];
}

const Mode = enum { none, replay, forward, both };

fn replay(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, capture: []const u8, out_dir: []const u8, mode: Mode) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var kernels: ?dk.Kernels = dk.Kernels.load(&ctx) catch |e| blk: {
        std.debug.print("note: extension kernels unavailable ({t}): extension ops are skipped\n", .{e});
        break :blk null;
    };
    defer if (kernels) |*k| k.deinit();

    // our loader: the captured blocks plus the vocabulary rows, norm and head, rank 0 of 2
    const t0 = std.Io.Clock.awake.now(io);
    const cfg = try Config.read(gpa, io, pack_dir);
    var pack = try Pack.open(gpa, io, pack_dir);
    defer pack.deinit();
    const blocks = try layers(a, io, capture);
    // a drafting pass in the capture (set P: blocks past the backbone): DSpark's heads load too
    const drafting = blocks.len > 0 and blocks[blocks.len - 1] >= cfg.layers;
    var pl = try plan.build(gpa, &cfg, &pack, .{ .rank = 0, .world = 2, .blocks = blocks, .top = true, .dspark = drafting });
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
        try b.dspark(&pl);
        try w.upload(&driver, &b);
    }
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9;
    std.debug.print("loaded {d} layers, {d} tensors, {d:.2} GB in {d:.1} s\n", .{ blocks.len, w.map.count(), @as(f64, @floatFromInt(w.bytes)) / 1e9, load_s });

    var wlog: std.Io.Writer.Allocating = .init(gpa);
    defer wlog.deinit();
    const wn = try gr.checkWeights(gpa, io, &w, capture, &wlog.writer);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "weights.jsonl" }), .data = wlog.written() });
    std.debug.print("weights: {d} equal, {d} differ, {d} missing\n", .{ wn[0], wn[1], wn[2] });

    var set = try tri.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ capture, "aot" }));
    defer set.deinit();
    std.debug.print("triton: {d} kernels loaded, {d} failed\n", .{ set.kernels.count(), set.failed.items.len });
    const diff_dir = try std.fs.path.join(a, &.{ out_dir, "diff" });
    try std.Io.Dir.cwd().createDirPath(io, diff_dir);
    var r: gr.Replay = .{ .gpa = gpa, .io = io, .d = &driver, .stream = stream, .kernels = if (kernels) |*k| k else null, .weights = &w, .triton = &set, .dir = capture, .diff_dir = diff_dir };
    defer r.deinit();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ capture, "ops.jsonl" }), gpa, .limited(1 << 32));
    defer gpa.free(text);
    var log: std.Io.Writer.Allocating = .init(gpa);
    defer log.deinit();
    // replay.jsonl grows op by op, so a run cut short still leaves what it checked
    const log_file = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ out_dir, "replay.jsonl" }), .{});
    defer log_file.close(io);
    var log_at: u64 = 0;
    // [mode][status] counts, mode 0 alone, 1 chained
    var counts: [2][4]usize = @splat(@splat(0));
    var chained_in: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, if (mode == .forward) "" else text, '\n');
    while (lines.next()) |line| {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
        defer parsed.deinit();
        const op = parsed.value.object;
        const alone = r.run(op, false);
        defer gpa.free(alone.detail);
        const chain = r.run(op, true);
        defer gpa.free(chain.detail);
        counts[0][@backingInt(alone.status)] += 1;
        counts[1][@backingInt(chain.status)] += 1;
        chained_in += chain.chained_in;
        try log.writer.print("{{\"seq\": {d}, \"set\": \"{s}\", \"phase\": \"{s}\", \"kind\": \"{s}\", \"name\": \"{s}\", " ++
            "\"alone\": \"{t}\", \"why\": \"{s}\", \"differ\": {d}, \"diff_bytes\": {d}, \"first\": {d}, " ++
            "\"chained\": \"{t}\", \"chained_why\": \"{s}\", \"chained_in\": {d}, \"alone_bufs\": \"{s}\", \"chained_bufs\": \"{s}\"}}\n", .{
            op.get("seq").?.integer, op.get("set").?.string, op.get("phase").?.string, op.get("kind").?.string,
            op.get("name").?.string, alone.status,           alone.why,                alone.differ,
            alone.diff_bytes,        alone.first,            chain.status,             chain.why,
            chain.chained_in,        alone.detail,           chain.detail,
        });
        const fresh = log.written()[log_at..];
        try log_file.writePositionalAll(io, fresh, log_at);
        log_at += fresh.len;
    }

    const summary = try std.fmt.allocPrint(a,
        \\{{"weights": {{"equal": {d}, "differ": {d}, "missing": {d}}}, "triton_loaded": {d}, "triton_failed": {d},
        \\ "alone": {{"equal": {d}, "differ": {d}, "skipped": {d}, "failed": {d}}},
        \\ "chained": {{"equal": {d}, "differ": {d}, "skipped": {d}, "failed": {d}, "inputs_from_our_outputs": {d}}}, "load_s": {d:.1}}}
        \\
    , .{ wn[0], wn[1], wn[2], set.kernels.count(), set.failed.items.len, counts[0][0], counts[0][1], counts[0][2], counts[0][3], counts[1][0], counts[1][1], counts[1][2], counts[1][3], chained_in, load_s });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "summary.json" }), .data = summary });
    std.debug.print("{s}", .{summary});
    var ok = wn[1] == 0 and wn[2] == 0 and counts[0][1] == 0 and counts[0][3] == 0;
    if (mode != .forward) std.debug.print("{s} capture replay\n", .{if (ok) "PASS" else "FAIL"});
    if (mode == .replay) return if (ok) 0 else 1;
    // the forward gate: our calls on our buffers, the replay's chain freed first
    r.deinit();
    r.live = .empty;
    var f: fwd.Forward = .{ .gpa = gpa, .io = io, .d = &driver, .stream = stream, .kernels = if (kernels) |*k| k else return error.NoKernels, .weights = &w, .triton = &set, .plan = &pl, .dir = capture };
    defer @import("close_or_exit.zig").finish(&f);
    _ = try fwd.runAll(&f, out_dir, @embedFile("fixtures/expert-k2.txt"));
    const c = try fwd.runDraft(&f, out_dir, @embedFile("fixtures/expert-k2.txt"));
    var glue: std.Io.Writer.Allocating = .init(gpa);
    defer glue.deinit();
    for (f.glue.keys(), f.glue.values()) |k, v| try glue.writer.print("\"{s}\": {d}, ", .{ k, v });
    const fsum = try std.fmt.allocPrint(a,
        \\{{"calls": {d}, "equal": {d}, "differ": {d}, "skipped_prefetch": {d}, "failed": {d}, "seeded": {d},
        \\ "reseeded_glue": {d}, "glue_roles": {{{s}}}, "prepared": {{"equal": {d}, "differ": {d}}}}}
        \\
    , .{ c.calls, c.equal, c.differ, c.skipped, c.failed, c.seeded, c.reseeded, if (glue.written().len > 2) glue.written()[0 .. glue.written().len - 2] else "", c.prepared_equal, c.prepared_differ });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ out_dir, "forward-summary.json" }), .data = fsum });
    std.debug.print("{s}", .{fsum});
    const fok = c.differ == 0 and c.failed == 0 and c.prepared_differ == 0 and c.equal > 0;
    std.debug.print("{s} block forward\n", .{if (fok) "PASS" else "FAIL"});
    if (mode == .forward) ok = true;
    return if (ok and fok) 0 else 1;
}
