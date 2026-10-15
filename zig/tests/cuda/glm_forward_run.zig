//! tf-cuda-test glm-tf and glm-prof: teacher-forced agreement with the Python engine, and a step's time by launch.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const gw = @import("glm_weights");
const fwd = gw.forward;
const dec = gw.decode;
const Gpu = check.Gpu;
const ms = @import("glm_forward.zig").ms;

/// glm-tf <rank> <rank 0 address> <port> <dir> <reference.json>: top-token agreement with Python's greedy reply.
pub fn teacher(gpu: Gpu, args: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    if (args.len < 5) return error.Usage;
    const rank = try std.fmt.parseInt(u8, args[0], 10);
    const port = try std.fmt.parseInt(u16, args[2], 10);
    var link = if (rank == 0) try cuda.tp_link.Link.lead(port, 600_000) else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(args[1], port), 600_000);
    defer link.close();
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    const comm = try cuda.tp_link.communicator(&nccl, link);
    // the communicator's threads stop before the library unloads (an unloaded NCCL under live threads crashes the exit)
    defer _ = nccl.api.ncclCommAbort(comm);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    {
        const warm = try cuda.DeviceBuffer.alloc(gpu.d, 4096);
        try nccl.check(nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, stream.handle), "warm gather");
        try stream.synchronize();
    }
    var kernels = try gw.api.Kernels.load(gpu.d);
    var ctx: gw.api.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = &nccl, .comm = comm, .rank = rank };
    var store = try gw.load(gpa, io, &ctx, args[3], rank);
    defer store.deinit();
    const e = try fwd.init(gpa, io, &ctx, &kernels, &store, .{ .context = 65536 });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[4], gpa, .limited(1 << 26));
    const ref = (try std.json.parseFromSlice(std.json.Value, gpa, text, .{})).value.object;
    const ints = struct {
        fn f(a: std.mem.Allocator, v: std.json.Value) ![]u32 {
            const it = v.array.items;
            const out = try a.alloc(u32, it.len);
            for (it, 0..) |x, i| out[i] = @intCast(x.integer);
            return out;
        }
    }.f;
    const prompt = try ints(gpa, ref.get("prompt").?);
    const serial = try ints(gpa, ref.get("serial").?);
    const s = try fwd.newSeq(e, 8192);
    const first = try dec.prefill(e, s, prompt, 0, 0);
    var hits: usize = @intFromBool(first == serial[0]);
    var picks: [8]dec.Pick = undefined;
    var j: usize = 0;
    while (j + 1 < serial.len) {
        const w = @min(8, serial.len - 1 - j);
        try dec.verify(e, &.{.{ .s = s, .ids = serial[j .. j + w] }}, &picks);
        try dec.keep(e, s, @intCast(w), @intCast(w));
        for (0..w) |r| hits += @intFromBool(picks[r].tok == serial[j + r + 1]);
        j += w;
    }
    std.debug.print("rank {d}: teacher-forced on {d} prompt tokens: the top token equals Python's next token at {d} of {d} positions\n", .{ rank, prompt.len, hits, serial.len });
    std.process.exit(0); // as glm-batch
}

/// glm-prof <rank> <rank 0 address> <port> <dir> [streams]: 1 and 4-row verify windows, each launch timed, and a graph.
pub fn prof(gpu: Gpu, args: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    if (args.len < 4) return error.Usage;
    const rank = try std.fmt.parseInt(u8, args[0], 10);
    const port = try std.fmt.parseInt(u16, args[2], 10);
    const nstreams: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 1;
    const graphs_only = args.len > 5 and std.mem.eql(u8, args[5], "graphs"); // for nsys: 50 one-row steps, no eager pass
    var link = if (rank == 0) try cuda.tp_link.Link.lead(port, 600_000) else try cuda.tp_link.Link.follow(io, try std.Io.net.IpAddress.parse(args[1], port), 600_000);
    defer link.close();
    var nccl = try cuda.nccl.Library.open();
    defer nccl.close();
    const comm = try cuda.tp_link.communicator(&nccl, link);
    // the communicator's threads stop before the library unloads (an unloaded NCCL under live threads crashes the exit)
    defer _ = nccl.api.ncclCommAbort(comm);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    {
        const warm = try cuda.DeviceBuffer.alloc(gpu.d, 4096);
        try nccl.check(nccl.api.ncclAllGather(warm.ptr, warm.ptr + 2048, 256, .i32, comm, stream.handle), "warm gather");
        try stream.synchronize();
    }
    var kernels = try gw.api.Kernels.load(gpu.d);
    var ctx: gw.api.Ctx = .{ .d = gpu.d, .ctx = gpu.ctx, .stream = stream, .nccl = &nccl, .comm = comm, .rank = rank };
    const no_roce = if (std.c.getenv("TF_ROCE")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
    ctx.roce = try cuda.roce.Gather.init(gpa, io, gpu.d, link, !no_roce, 512 << 10);
    defer if (ctx.roce) |g| g.deinit();
    var store = try gw.load(gpa, io, &ctx, args[3], rank);
    defer store.deinit();
    const e = try fwd.init(gpa, io, &ctx, &kernels, &store, .{ .context = 65536 });
    if (std.c.getenv("TF_PROF_PROMPT")) |v| { // a prompt of this many tokens: its wall time, then its kernels (eager chunks)
        const n = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        const p = try gpa.alloc(u32, n);
        defer gpa.free(p);
        for (p, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 60000);
        const s0 = try fwd.newSeq(e, @intCast(n + 1024));
        const t0 = std.Io.Timestamp.now(io, .awake);
        _ = try dec.prefill(e, s0, p, 0, 0);
        const wall = ms(io, t0);
        if (rank == 0) std.debug.print("prompt of {d} tokens: {d:.0} ms, {d:.0} tok/s\n", .{ n, wall, @as(f64, @floatFromInt(n)) / wall * 1000 });
        var m = std.StringHashMap(f64).init(gpa);
        e.k.prof = &m;
        e.k.io = io;
        const s1 = try fwd.newSeq(e, @intCast(n + 1024));
        _ = try dec.prefill(e, s1, p, 0, 0);
        e.k.prof = null;
        var total: f64 = 0;
        var it = m.iterator();
        const Row = struct { name: []const u8, ms: f64 };
        var rows: std.ArrayList(Row) = .empty;
        while (it.next()) |x| {
            total += x.value_ptr.*;
            try rows.append(gpa, .{ .name = x.key_ptr.*, .ms = x.value_ptr.* });
        }
        std.mem.sort(Row, rows.items, {}, struct {
            fn lt(_: void, a: Row, b: Row) bool {
                return a.ms > b.ms;
            }
        }.lt);
        if (rank == 0) {
            std.debug.print("prompt of {d} tokens: eager launches sum {d:.0} ms (synchronized: proportions only)\n", .{ n, total });
            for (rows.items[0..@min(rows.items.len, 30)]) |x| std.debug.print("  {d:8.1} ms {d:5.1}%  {s}\n", .{ x.ms, 100 * x.ms / total, x.name[0..@min(x.name.len, 90)] });
        }
        return;
    }
    var seqs: [fwd.MAX_STREAMS]*fwd.Seq = undefined;
    var prompt: [512]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + (i * 7919) % 60000);
    var tok: [fwd.MAX_STREAMS]u32 = undefined;
    for (0..nstreams) |i| {
        seqs[i] = try fwd.newSeq(e, 8192);
        prompt[0] = @intCast(2000 + i);
        tok[i] = try dec.prefill(e, seqs[i], &prompt, 0, 0);
    }
    var picks: [fwd.STEP_ROWS]dec.Pick = undefined;
    var ids: [fwd.MAX_STREAMS][8]u32 = undefined;
    var wins: [fwd.MAX_STREAMS]dec.Win = undefined;
    for ([_]usize{ 1, 4, 8 }) |width| {
        if (graphs_only and width != 1) break;
        const step = struct {
            fn f(ee: *fwd.Engine, ss: []*fwd.Seq, tk: []u32, idv: *[fwd.MAX_STREAMS][8]u32, wv: *[fwd.MAX_STREAMS]dec.Win, pk: []dec.Pick, w: usize) !void {
                for (ss, 0..) |s, i| {
                    for (0..w) |r| idv[i][r] = if (r == 0) tk[i] else @intCast(3000 + r * 101 + i);
                    wv[i] = .{ .s = s, .ids = idv[i][0..w] };
                }
                try dec.verify(ee, wv[0..ss.len], pk);
                for (ss, 0..) |s, i| {
                    try dec.keep(ee, s, @intCast(w), 1);
                    tk[i] = pk[i * w].tok;
                }
            }
        }.f;
        for (0..8) |_| try step(e, seqs[0..nstreams], &tok, &ids, &wins, &picks, width); // graphs captured, clocks up
        var best: f64 = std.math.inf(f64);
        for (0..10) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            try step(e, seqs[0..nstreams], &tok, &ids, &wins, &picks, width);
            best = @min(best, ms(io, t0));
        }
        if (graphs_only) {
            for (0..40) |_| try step(e, seqs[0..nstreams], &tok, &ids, &wins, &picks, width);
            if (rank == 0) std.debug.print("{d} stream(s) x {d} rows: graph step {d:.2} ms (graphs only)\n", .{ nstreams, width, best });
            continue;
        }
        var m = std.StringHashMap(f64).init(gpa);
        e.k.prof = &m;
        e.k.io = io;
        e.use_graphs = false;
        for (0..4) |_| try step(e, seqs[0..nstreams], &tok, &ids, &wins, &picks, width);
        e.k.prof = null;
        e.use_graphs = true;
        var total: f64 = 0;
        var it = m.iterator();
        const Row = struct { name: []const u8, ms: f64 };
        var rows: std.ArrayList(Row) = .empty;
        while (it.next()) |x| {
            total += x.value_ptr.* / 4;
            try rows.append(gpa, .{ .name = x.key_ptr.*, .ms = x.value_ptr.* / 4 });
        }
        std.mem.sort(Row, rows.items, {}, struct {
            fn lt(_: void, a: Row, b: Row) bool {
                return a.ms > b.ms;
            }
        }.lt);
        if (rank == 0) {
            std.debug.print("{d} stream(s) x {d} rows: graph step {d:.2} ms; eager launches sum {d:.2} ms\n", .{ nstreams, width, best, total });
            for (rows.items[0..@min(rows.items.len, 22)]) |x| std.debug.print("  {d:7.3} ms {d:5.1}%  {s}\n", .{ x.ms, 100 * x.ms / total, x.name[0..@min(x.name.len, 90)] });
        }
    }
    std.process.exit(0); // as glm-batch
}
