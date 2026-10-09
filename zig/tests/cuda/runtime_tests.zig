//! Driver-level GPU tests: copies, fills, launches, argument packing, module globals and graphs.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

const ProbeView = extern struct { src: u64, n: c_int, scale: f32 };

pub fn smoke(gpu: Gpu) !void {
    const d = gpu.d;
    const n: usize = 1 << 20;
    const pattern = try gpu.gpa.alloc(u8, n);
    defer gpu.gpa.free(pattern);
    for (pattern, 0..) |*p, i| p.* = @truncate(i *% 2654435761 >> 7);

    var a = try cuda.DeviceBuffer.fromHost(d, pattern);
    defer a.free();
    var b = try cuda.DeviceBuffer.alloc(d, n);
    defer b.free();
    try b.copyFrom(0, a.ptr, n, null);
    const back = try check.download(gpu, b);
    defer gpu.gpa.free(back);
    try check.sameBytes("H2D, D2D, D2H round trip (1 MiB)", back, pattern);
    check.pass("blocking copies: 1 MiB host -> device -> device -> host equal", .{});

    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var pinned = try cuda.HostBuffer.alloc(d, n);
    defer pinned.free();
    @memcpy(pinned.bytes, pattern);
    var c = try cuda.DeviceBuffer.alloc(d, n);
    defer c.free();
    try c.uploadAsync(0, pinned.bytes, stream.handle);
    @memset(back, 0);
    try c.downloadAsync(0, back, stream.handle);
    try stream.synchronize();
    try check.sameBytes("async copies through pinned memory", back, pattern);
    try c.fill8(0x5a, stream.handle);
    try stream.synchronize();
    try c.download(0, back);
    try expect(std.mem.allEqual(u8, back, 0x5a), "memset8", .{});
    try c.fill32(0xdeadbeef, null);
    try c.download(0, back);
    for (std.mem.bytesAsSlice(u32, back)) |w| try expect(w == 0xdeadbeef, "memset32 word {x}", .{w});
    check.pass("async copies, memset8 and memset32", .{});

    var probe = try cuda.Module.load(d, cuda.kernels.probe);
    defer probe.unload();
    const fill = try probe.function("tf_probe_fill");
    const axpy = try probe.function("tf_probe_axpy");
    const count: u32 = 100_000;
    var y = try cuda.DeviceBuffer.alloc(d, count * 4);
    defer y.free();
    var x = try cuda.DeviceBuffer.alloc(d, count * 4);
    defer x.free();
    const blocks = (count + 255) / 256;
    var args: cuda.Args = .{};
    args.add(x.ptr);
    args.add(@as(f32, 0));
    args.add(count);
    try cuda.launch.launch(fill, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    args = .{};
    args.add(y.ptr);
    args.add(@as(f32, 1));
    args.add(count);
    try cuda.launch.launch(fill, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    args = .{};
    args.add(y.ptr);
    args.add(x.ptr);
    args.add(@as(f32, 2));
    args.add(count);
    try cuda.launch.launch(axpy, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    const ys = try check.download(gpu, y);
    defer gpu.gpa.free(ys);
    for (std.mem.bytesAsSlice(f32, ys), 0..) |v, i| {
        const want: f32 = 2 * @as(f32, @floatFromInt(i)) + (1 + @as(f32, @floatFromInt(i)));
        try expect(v == want, "fill+axpy element {d}: {d} != {d}", .{ i, v, want });
    }
    check.pass("launches: fill, fill, axpy over {d} elements exact", .{count});

    // the serial round's token history: a kernel writes mapped host memory, over PCIe on a discrete card
    var mapped = try cuda.HostBuffer.allocMapped(d, 1024 * 4);
    defer mapped.free();
    @memset(mapped.bytes, 0);
    args = .{};
    args.add(try mapped.device());
    args.add(@as(f32, 7));
    args.add(@as(u32, 1024));
    try cuda.launch.launch(fill, .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    for (mapped.slice(f32), 0..) |v, i| try expect(v == 7 + @as(f32, @floatFromInt(i)), "mapped host word {d}: {d}", .{ i, v });
    check.pass("mapped host memory: a kernel's 1024 writes read on the host after the stream syncs", .{});

    const table = try probe.global("tf_probe_table");
    try expect(table.len == 16, "module global size {d}", .{table.len});
    const vals = [4]i32{ 1, 2, 3, 4 };
    try d.check(d.api.cuMemcpyHtoD_v2(table.ptr, &vals, 16), "write global");
    var out = try cuda.DeviceBuffer.alloc(d, 64);
    defer out.free();
    args = .{};
    args.add(out.ptr);
    try cuda.launch.launch(try probe.function("tf_probe_read_table"), .{ .grid = .{}, .block = .{ .x = 4 } }, stream, &args);
    try stream.synchronize();
    var got: [4]i32 = undefined;
    try out.download(0, std.mem.asBytes(&got));
    try expect(std.mem.eql(i32, &got, &.{ 2, 4, 6, 8 }), "module global read back {any}", .{got});
    check.pass("module global: cuModuleGetGlobal write, kernel read", .{});

    try argPacking(gpu, probe, stream);

    var bad: cuda.Args = .{};
    bad.add(out.ptr);
    const refused = cuda.launch.launch(fill, .{ .grid = .{}, .block = .{ .x = 2048 } }, stream, &bad);
    try expect(refused == error.Invalid, "a 2048-thread block is refused before the driver", .{});
    const missing = probe.function("tf_probe_missing");
    try expect(missing == error.NotFound, "an absent symbol is NOT_FOUND", .{});
    check.pass("refusals: oversized block, absent kernel symbol", .{});
}

/// A by-value struct, bool, int8, double and int64 reach the kernel exactly as packed.
fn argPacking(gpu: Gpu, probe: cuda.Module, stream: cuda.Stream) !void {
    const d = gpu.d;
    const src = [8]f32{ 1, -2, 3.5, 4, 5, 6.25, -7, 8 };
    var s = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&src));
    defer s.free();
    var out = try cuda.DeviceBuffer.alloc(d, 12 * 4);
    defer out.free();
    var args: cuda.Args = .{};
    args.add(out.ptr);
    args.add(ProbeView{ .src = s.ptr, .n = 8, .scale = 0.5 });
    args.add(true);
    args.add(@as(i8, -5));
    args.add(@as(f64, 3.25));
    args.add(@as(i64, -123456789));
    try cuda.launch.launch(try probe.function("tf_probe_args"), .{ .grid = .{}, .block = .{ .x = 32 } }, stream, &args);
    try stream.synchronize();
    var got: [12]f32 = undefined;
    try out.download(0, std.mem.asBytes(&got));
    for (0..8) |i| try expect(got[i] == src[i] * 0.5, "struct arg element {d}", .{i});
    const big: f32 = @floatFromInt(@as(i64, -123456789));
    try expect(got[8] == 1 and got[9] == -5 and got[10] == 3.25 and got[11] == big, "scalar args {any}", .{got[8..]});
    check.pass("argument packing: by-value struct, bool, int8, double, int64", .{});
}

pub fn graphs(gpu: Gpu) !void {
    const d = gpu.d;
    var probe = try cuda.Module.load(d, cuda.kernels.probe);
    defer probe.unload();
    const step = try probe.function("tf_probe_step");
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var counter = try cuda.DeviceBuffer.alloc(d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    const one: cuda.Config = .{ .grid = .{}, .block = .{} };

    try cuda.graph.beginCapture(stream, .thread_local);
    try expect(try cuda.graph.captureStatus(stream) == .active, "stream is capturing", .{});
    for (1..9) |i| {
        var args: cuda.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, i));
        try cuda.launch.launch(step, one, stream, &args);
    }
    var captured = try cuda.graph.endCapture(stream);
    defer captured.deinit();
    try expect(try captured.nodeCount() == 8, "captured graph has 8 nodes", .{});
    try expect(try readCounter(counter) == 0, "capture runs nothing", .{});
    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(stream);
    for (0..3) |_| try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try readCounter(counter) == 108, "three replays add 3 * 36", .{});
    check.pass("stream capture: 8 launches captured, nothing run, 3 replays = 108", .{});

    var nodes_buf: [8]cuda.graph.Node = undefined;
    const nodes = try captured.nodes(&nodes_buf);
    for (nodes) |node| {
        var args: cuda.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, 10));
        try exec.setKernel(node, step, one, &args);
    }
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try readCounter(counter) == 188, "updated nodes add 8 * 10", .{});
    check.pass("exec kernel-node update: new arguments in place, replay adds 80", .{});

    try explicitGraphs(gpu, probe, stream);
}

fn readCounter(counter: cuda.DeviceBuffer) !u64 {
    var v: u64 = 0;
    try counter.download(0, std.mem.asBytes(&v));
    return v;
}

fn fillAxpy(g: cuda.graph.Graph, fill: cuda.Function, axpy: cuda.Function, y: u64, x: u64, base: f32, a: f32, n: u32) !void {
    const cfg: cuda.Config = .{ .grid = .{ .x = (n + 255) / 256 }, .block = .{ .x = 256 } };
    var fa: cuda.Args = .{};
    fa.add(y);
    fa.add(base);
    fa.add(n);
    const first = try g.addKernel(&.{}, fill, cfg, &fa);
    var aa: cuda.Args = .{};
    aa.add(y);
    aa.add(x);
    aa.add(a);
    aa.add(n);
    _ = try g.addKernel(&.{first}, axpy, cfg, &aa);
}

/// Explicit nodes with an edge, then a whole-exec update from a same-topology graph, and a refused topology change.
fn explicitGraphs(gpu: Gpu, probe: cuda.Module, stream: cuda.Stream) !void {
    const d = gpu.d;
    const n: u32 = 4096;
    const fill = try probe.function("tf_probe_fill");
    const axpy = try probe.function("tf_probe_axpy");
    var x = try cuda.DeviceBuffer.alloc(d, n * 4);
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(d, n * 4);
    defer y.free();
    var args: cuda.Args = .{};
    args.add(x.ptr);
    args.add(@as(f32, 0));
    args.add(n);
    try cuda.launch.launch(fill, .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 } }, stream, &args);

    var g1 = try cuda.graph.Graph.init(d);
    defer g1.deinit();
    try fillAxpy(g1, fill, axpy, y.ptr, x.ptr, 1, 2, n);
    var exec = try g1.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 1, 2, n);

    var g2 = try cuda.graph.Graph.init(d);
    defer g2.deinit();
    try fillAxpy(g2, fill, axpy, y.ptr, x.ptr, 5, 3, n);
    const r2 = try exec.update(g2);
    try expect(r2 == .success, "same-topology update result {t}", .{r2});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 5, 3, n);

    var g3 = try cuda.graph.Graph.init(d);
    defer g3.deinit();
    try fillAxpy(g3, fill, axpy, y.ptr, x.ptr, 7, 4, n);
    var extra: cuda.Args = .{};
    extra.add(x.ptr);
    extra.add(@as(f32, 0));
    extra.add(n);
    _ = try g3.addKernel(&.{}, fill, .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 } }, &extra);
    const r3 = try exec.update(g3);
    try expect(r3 != .success, "a topology change must be refused", .{});
    try exec.launchOn(stream);
    try stream.synchronize();
    try expectAxpy(gpu, y, 5, 3, n);
    check.pass("explicit graph: fill->axpy edge exact; exec update from a same-topology graph; topology change refused ({t}) and the exec kept", .{r3});
}

fn expectAxpy(gpu: Gpu, y: cuda.DeviceBuffer, base: f32, a: f32, n: u32) !void {
    const got = try check.download(gpu, y);
    defer gpu.gpa.free(got);
    for (std.mem.bytesAsSlice(f32, got)[0..n], 0..) |v, i| {
        const fi: f32 = @floatFromInt(i);
        try expect(v == a * fi + (base + fi), "graph axpy element {d}: {d}", .{ i, v });
    }
}

/// cuLaunchKernelEx: cluster dimensions (plain and spread), a cooperative grid, and a PDL chain on a stream and in a graph.
pub fn launchEx(gpu: Gpu) !void {
    const d = gpu.d;
    var probe = try cuda.Module.load(d, cuda.kernels.probe);
    defer probe.unload();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const clusters = try gpu.ctx.capability() >= 90; // thread-block clusters and PDL start at sm_90
    const ranks = try probe.function("tf_probe_cluster_rank");
    var out = try cuda.DeviceBuffer.alloc(d, 8 * 4);
    defer out.free();
    if (!clusters) std.debug.print("SKIP cuLaunchKernelEx clusters and PDL: sm_{d} has neither\n", .{try gpu.ctx.capability()});
    for ([_]u32{ 2, 4 }, [_]bool{ false, true }) |size, spread| {
        if (!clusters) break;
        try out.fill32(0xffffffff, null);
        var args: cuda.Args = .{};
        args.add(out.ptr);
        try cuda.launch.launch(ranks, .{ .grid = .{ .x = 8 }, .block = .{ .x = 32 }, .cluster = .{ .x = size }, .cluster_spread = spread }, stream, &args);
        try stream.synchronize();
        var got: [8]u32 = undefined;
        try out.download(0, std.mem.asBytes(&got));
        for (got, 0..) |rank, i| try expect(rank == i % size, "cluster {d} block {d} rank {d}", .{ size, i, rank });
    }
    if (clusters) check.pass("cuLaunchKernelEx clusters: ranks 0..1 and 0..3 in clusters of 2 and 4 (spread policy on the second)", .{});

    const fill = try probe.function("tf_probe_fill");
    const sms: u32 = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var y = try cuda.DeviceBuffer.alloc(d, sms * 256 * 4);
    defer y.free();
    var fa: cuda.Args = .{};
    fa.add(y.ptr);
    fa.add(@as(f32, 3));
    fa.add(sms * 256);
    try cuda.launch.launch(fill, .{ .grid = .{ .x = sms }, .block = .{ .x = 256 }, .cooperative = true }, stream, &fa);
    try stream.synchronize();
    const ys = try check.download(gpu, y);
    defer gpu.gpa.free(ys);
    for (std.mem.bytesAsSlice(f32, ys), 0..) |v, i| try expect(v == 3 + @as(f32, @floatFromInt(i)), "cooperative fill {d}", .{i});
    check.pass("cuLaunchKernelEx cooperative grid of {d} blocks", .{sms});

    if (!clusters) return;
    const step = try probe.function("tf_probe_pdl_step");
    var counter = try cuda.DeviceBuffer.alloc(d, 8);
    defer counter.free();
    try counter.fill8(0, null);
    const pdl: cuda.Config = .{ .grid = .{}, .block = .{}, .pdl = true };
    for (1..101) |i| {
        var args: cuda.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, i));
        try cuda.launch.launch(step, pdl, stream, &args);
    }
    try stream.synchronize();
    try expect(try readCounter(counter) == 5050, "PDL chain on a stream", .{});
    try cuda.graph.beginCapture(stream, .thread_local);
    for (1..101) |i| {
        var args: cuda.Args = .{};
        args.add(counter.ptr);
        args.add(@as(u64, i));
        try cuda.launch.launch(step, pdl, stream, &args);
    }
    var g = try cuda.graph.endCapture(stream);
    defer g.deinit();
    var exec = try g.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try exec.launchOn(stream);
    try stream.synchronize();
    try expect(try readCounter(counter) == 3 * 5050, "PDL chain captured in a graph", .{});
    check.pass("cuLaunchKernelEx PDL: 100 dependent steps exact on a stream and twice from a captured graph", .{});
}

const add_one_ptx =
    \\.version 8.0
    \\.target sm_80
    \\.address_size 64
    \\
    \\.visible .entry tf_ptx_add_one(
    \\    .param .u64 p_out,
    \\    .param .u64 p_src,
    \\    .param .u32 p_n
    \\)
    \\{
    \\    .reg .pred %p1;
    \\    .reg .b32 %r<5>;
    \\    .reg .f32 %f<2>;
    \\    .reg .b64 %d<6>;
    \\    ld.param.u64 %d0, [p_out];
    \\    ld.param.u64 %d1, [p_src];
    \\    ld.param.u32 %r0, [p_n];
    \\    mov.u32 %r1, %ctaid.x;
    \\    mov.u32 %r2, %ntid.x;
    \\    mov.u32 %r3, %tid.x;
    \\    mad.lo.u32 %r4, %r1, %r2, %r3;
    \\    setp.ge.u32 %p1, %r4, %r0;
    \\    @%p1 bra done;
    \\    cvta.to.global.u64 %d2, %d0;
    \\    cvta.to.global.u64 %d3, %d1;
    \\    mul.wide.u32 %d4, %r4, 4;
    \\    add.u64 %d5, %d3, %d4;
    \\    ld.global.f32 %f0, [%d5];
    \\    add.f32 %f1, %f0, 0f3F800000;
    \\    add.u64 %d5, %d2, %d4;
    \\    st.global.f32 [%d5], %f1;
    \\done:
    \\    ret;
    \\}
    \\
;

/// A PTX image JIT-compiled by the driver, and a broken one refused with the driver's error log.
pub fn ptx(gpu: Gpu) !void {
    const d = gpu.d;
    var log: [4096]u8 = undefined;
    var m = try cuda.Module.loadPtx(d, add_one_ptx, &log);
    defer m.unload();
    const f = try m.function("tf_ptx_add_one");
    const n: u32 = 1000;
    var src = try cuda.DeviceBuffer.alloc(d, n * 4);
    defer src.free();
    var out = try cuda.DeviceBuffer.alloc(d, n * 4);
    defer out.free();
    try src.fill32(@bitCast(@as(f32, 2.5)), null);
    var args: cuda.Args = .{};
    args.add(out.ptr);
    args.add(src.ptr);
    args.add(n);
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try cuda.launch.launch(f, .{ .grid = .{ .x = (n + 127) / 128 }, .block = .{ .x = 128 } }, stream, &args);
    try stream.synchronize();
    const got = try check.download(gpu, out);
    defer gpu.gpa.free(got);
    for (std.mem.bytesAsSlice(f32, got)) |v| try expect(v == 3.5, "PTX kernel result {d}", .{v});
    const refused = cuda.Module.loadPtx(d, ".version 8.0\n.target sm_80\nnot ptx at all\n", &log);
    try expect(refused == error.CudaFailed, "broken PTX must be refused", .{});
    const text = std.mem.sliceTo(&log, 0);
    try expect(text.len > 0, "the driver's PTX error log is empty", .{});
    check.pass("PTX: hand-written sm_80 PTX JIT-compiled and exact; broken PTX refused with log \"{s}\"", .{text[0..@min(text.len, 80)]});
}

/// Every gdn instantiation resolves by its listed symbol, with its register and thread limits for the record.
pub fn symbols(gpu: Gpu) !void {
    var m = try cuda.Module.load(gpu.d, cuda.kernels.gdn);
    defer m.unload();
    const replays = [_][:0]const u8{ cuda.kernels.gdn_symbols.replay_bf16, cuda.kernels.gdn_symbols.replay_f32 };
    for (replays) |name| _ = try m.function(name);
    for (cuda.kernels.gdn_variants) |v| {
        const f = try m.function(v.symbol);
        std.debug.print("RESULT gdn tree<{d},{d},{d},{}>: {d} registers, max {d} threads\n", .{ v.slots, v.r, v.warps, v.chain, try f.attribute(.num_regs), try f.attribute(.max_threads_per_block) });
    }
    check.pass("symbols: 2 replay and {d} tree instantiations resolve in the gdn fatbin", .{cuda.kernels.gdn_variants.len});
}
