//! Bit-exact checks against the Python engine: our GDN .cu kernels and Triton cubins on the oracle's inputs.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;
const expect = check.expect;

const Pending = extern struct {
    k: u64 = 0,
    v: u64 = 0,
    g: u64 = 0,
    beta: u64 = 0,
    rows: u64 = 0,
    row_stride: c_int = 0,
    counts: u64 = 0,
    count_stride: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(Pending) == 64 and @offsetOf(Pending, "counts") == 48);
}

fn upload(gpu: Gpu, fx: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const host = try fx.bytes(name);
    defer gpu.gpa.free(host);
    return cuda.DeviceBuffer.fromHost(gpu.d, host);
}

fn int(fx: Fixture, name: []const u8) !i32 {
    return @intCast(try fx.int(name));
}

/// gdn_replay_cuda: one launch commits every (stream, layer); out of place, in place, and replayed from a graph.
pub fn gdnReplay(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const d = gpu.d;
    const layers = try int(fx, "layers");
    const streams = try int(fx, "streams");
    const hk = try int(fx, "hk");
    const hv = try int(fx, "hv");
    const dv = try int(fx, "dv");
    const width = try int(fx, "width");
    const row_stride = try int(fx, "row_stride");
    const L: usize = @intCast(layers);
    const S: usize = @intCast(streams);

    var k = try upload(gpu, fx, "k");
    defer k.free();
    var v = try upload(gpu, fx, "v");
    defer v.free();
    var g = try upload(gpu, fx, "g");
    defer g.free();
    var beta = try upload(gpu, fx, "beta");
    defer beta.free();
    var states = try upload(gpu, fx, "states");
    defer states.free();
    var rows = try upload(gpu, fx, "rows");
    defer rows.free();
    var counts = try upload(gpu, fx, "counts");
    defer counts.free();

    const w: usize = @intCast(width);
    const k_layer = w * @as(usize, @intCast(hk)) * 128 * 2;
    const v_layer = w * @as(usize, @intCast(hv * dv)) * 2;
    const gate_layer = w * @as(usize, @intCast(hv)) * 4;
    const state_bytes = @as(usize, @intCast(hv * dv)) * 128 * 4;
    var table_host = try gpu.gpa.alloc(u64, 4 * L + L * S);
    defer gpu.gpa.free(table_host);
    for (0..L) |l| {
        table_host[4 * l] = k.ptr + l * k_layer;
        table_host[4 * l + 1] = v.ptr + l * v_layer;
        table_host[4 * l + 2] = g.ptr + l * gate_layer;
        table_host[4 * l + 3] = beta.ptr + l * gate_layer;
    }
    for (0..S) |s| for (0..L) |l| {
        table_host[4 * L + s * L + l] = states.ptr + (s * L + l) * state_bytes;
    };
    var table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(table_host));
    defer table.free();
    var out = try cuda.DeviceBuffer.alloc(d, S * L * state_bytes);
    defer out.free();

    var gdn = try cuda.Module.load(d, cuda.kernels.gdn);
    defer gdn.unload();
    const kernel = try gdn.function(cuda.kernels.gdn_symbols.replay_bf16);
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(@divTrunc(dv + 31, 32)), .y = @intCast(hv), .z = @intCast(layers * streams) },
        .block = .{ .x = 128 },
    };
    const Replay = struct {
        fn args(t: u64, nl: i32, r: u64, rs: i32, c: u64, o: u64, a: i32, b: i32, e: i32) cuda.Args {
            var x: cuda.Args = .{};
            x.add(t);
            x.add(nl);
            x.add(r);
            x.add(rs);
            x.add(c);
            x.add(@as(i32, 1));
            x.add(o);
            x.add(a);
            x.add(b);
            x.add(e);
            return x;
        }
    };

    var args = Replay.args(table.ptr, layers, rows.ptr, row_stride, counts.ptr, out.ptr, hk, hv, dv);
    try cuda.launch.launch(kernel, cfg, stream, &args);
    try stream.synchronize();
    const want_out = try fx.bytes("out");
    defer gpu.gpa.free(want_out);
    const got_out = try check.download(gpu, out);
    defer gpu.gpa.free(got_out);
    try check.sameBytes("gdn replay out of place", got_out, want_out);
    check.pass("BITEXACT gdn replay_kernel<bf16,8,4> out of place: {d} bytes equal to Python's gdn.replay ({d} layers, {d} streams)", .{ got_out.len, L, S });

    try out.fill8(0, null);
    try cuda.graph.beginCapture(stream, .thread_local);
    args = Replay.args(table.ptr, layers, rows.ptr, row_stride, counts.ptr, out.ptr, hk, hv, dv);
    try cuda.launch.launch(kernel, cfg, stream, &args);
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.launchOn(stream);
    try stream.synchronize();
    try out.download(0, got_out);
    try check.sameBytes("gdn replay from a graph", got_out, want_out);
    check.pass("BITEXACT gdn replay_kernel replayed from a captured graph: same bits", .{});

    args = Replay.args(table.ptr, layers, rows.ptr, row_stride, counts.ptr, 0, hk, hv, dv);
    try cuda.launch.launch(kernel, cfg, stream, &args);
    try stream.synchronize();
    const want_states = try fx.bytes("states_after");
    defer gpu.gpa.free(want_states);
    const got_states = try check.download(gpu, states);
    defer gpu.gpa.free(got_states);
    try check.sameBytes("gdn replay in place", got_states, want_states);
    check.pass("BITEXACT gdn replay_kernel in place: {d} state bytes equal to Python's", .{got_states.len});
}

/// gdn_tree_cuda for one stream's window from its own state, with the instantiation dispatch_tree picks on this GPU.
pub fn gdnTree(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const d = gpu.d;
    const hk = try int(fx, "hk");
    const hv = try int(fx, "hv");
    const dv = try int(fx, "dv");
    const width = try int(fx, "width");
    const slots: u32 = @intCast(try fx.int("slots"));
    const max_rows: u32 = @intCast(try fx.int("max_rows"));
    const major = try gpu.ctx.attribute(.compute_capability_major);
    const minor = try gpu.ctx.attribute(.compute_capability_minor);
    const sms = try gpu.ctx.attribute(.multiprocessor_count);
    const variant = cuda.kernels.treeVariant(slots, 1, major, minor, sms);

    var q = try upload(gpu, fx, "q");
    defer q.free();
    var k = try upload(gpu, fx, "k");
    defer k.free();
    var v = try upload(gpu, fx, "v");
    defer v.free();
    var g = try upload(gpu, fx, "g");
    defer g.free();
    var beta = try upload(gpu, fx, "beta");
    defer beta.free();
    var state = try upload(gpu, fx, "state");
    defer state.free();
    var plan = try upload(gpu, fx, "plan");
    defer plan.free();
    const w: usize = @intCast(width);
    var y = try cuda.DeviceBuffer.alloc(d, w * @as(usize, @intCast(hv * dv)) * 2);
    defer y.free();

    var gdn = try cuda.Module.load(d, cuda.kernels.gdn);
    defer gdn.unload();
    const kernel = try gdn.function(variant.symbol);
    const per_block = variant.r * variant.warps;
    const shared: u32 = 16 * variant.warps * variant.slots * variant.r * 32 + (if (variant.chain) 0 else 4 * 3 * max_rows);
    if (shared > 48 * 1024) try kernel.allowDynamicShared(shared);
    const r8: u64 = @min(variant.r, 8);
    const vec = @mod(dv, @as(i32, @intCast(variant.r))) == 0 and v.ptr % (2 * r8) == 0;
    var args: cuda.Args = .{};
    for ([_]u64{ q.ptr, k.ptr, v.ptr, g.ptr, beta.ptr, state.ptr, 0, 0, plan.ptr }) |p| args.add(p);
    args.add(width);
    args.add(y.ptr);
    args.add(hk);
    args.add(hv);
    args.add(dv);
    args.add(Pending{});
    args.add(@as(u64, 0));
    args.add(@as(u64, 0));
    args.add(vec);
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(@divTrunc(@as(u32, @intCast(dv)) + per_block - 1, per_block)), .y = @intCast(hv) },
        .block = .{ .x = 32 * variant.warps },
        .shared = shared,
    };
    try cuda.launch.launch(kernel, cfg, stream, &args);
    try stream.synchronize();
    const want = try fx.bytes("y");
    defer gpu.gpa.free(want);
    const got = try check.download(gpu, y);
    defer gpu.gpa.free(got);
    try check.sameBytes("gdn tree", got, want);
    check.pass("BITEXACT gdn tree_kernel<bf16,{d},{d},{d},{}> ({d} rows, {d} slots, vec {}): {d} output bytes equal to Python's", .{ variant.slots, variant.r, variant.warps, variant.chain, width, slots, vec, got.len });
}

/// A Triton cubin launched from its metadata on the oracle's arguments; outputs compared byte for byte.
pub fn tritonKernel(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const d = gpu.d;
    const meta_text = try fx.readFile("kernel.json");
    defer gpu.gpa.free(meta_text);
    const meta = try cuda.triton.parseMeta(gpu.gpa, meta_text);
    defer meta.deinit();
    const cubin_raw = try fx.readFile("kernel.cubin");
    defer gpu.gpa.free(cubin_raw);
    const cubin = try gpu.gpa.alignedAlloc(u8, .@"16", cubin_raw.len);
    defer gpu.gpa.free(cubin);
    @memcpy(cubin, cubin_raw);
    const name_z = try gpu.gpa.dupeSentinel(u8, meta.value.name, 0);
    defer gpu.gpa.free(name_z);
    var kernel = try cuda.triton.Kernel.load(d, gpu.ctx.device, cubin, meta.value, name_z);
    defer kernel.unload();

    const list = fx.parsed.value.object.get("args").?.array;
    var buffers: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (buffers.items) |*b| b.free();
        buffers.deinit(gpu.gpa);
    }
    var outputs: std.ArrayList(struct { name: []const u8, index: usize }) = .empty;
    defer outputs.deinit(gpu.gpa);
    var args: cuda.Args = .{};
    for (list.items) |item| {
        const o = item.object;
        const kind = o.get("kind").?.string;
        if (std.mem.eql(u8, kind, "ptr")) {
            const name = o.get("array").?.string;
            const b = try upload(gpu, fx, name);
            if (o.get("output")) |flag| if (flag.bool) try outputs.append(gpu.gpa, .{ .name = name, .index = buffers.items.len });
            try buffers.append(gpu.gpa, b);
            args.add(b.ptr);
        } else if (std.mem.eql(u8, kind, "i32")) {
            args.add(@as(i32, @intCast(o.get("value").?.integer)));
        } else if (std.mem.eql(u8, kind, "i64")) {
            args.add(@as(i64, o.get("value").?.integer));
        } else if (std.mem.eql(u8, kind, "f32")) {
            args.add(@as(f32, @bitCast(@as(u32, @intCast(o.get("bits").?.integer)))));
        } else {
            return check.expect(false, "unknown Triton argument kind {s}", .{kind});
        }
    }
    const grid = fx.parsed.value.object.get("grid").?.array.items;
    const dims: cuda.Dim3 = .{ .x = @intCast(grid[0].integer), .y = @intCast(grid[1].integer), .z = @intCast(grid[2].integer) };
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    var divisible: std.ArrayList(u32) = .empty;
    defer divisible.deinit(gpu.gpa);
    for (fx.parsed.value.object.get("divisible16").?.array.items) |i| try divisible.append(gpu.gpa, @intCast(i.integer));
    if (divisible.items.len > 0) {
        var misaligned = args;
        misaligned.storage[misaligned.offsets[divisible.items[0]]] +%= 2;
        const refused = kernel.launchOn(dims, stream, &misaligned, .{}, divisible.items);
        try expect(refused == error.Invalid, "a pointer off its 16-byte specialization must be refused", .{});
    }
    try kernel.launchOn(dims, stream, &args, .{}, divisible.items);
    try stream.synchronize();
    for (outputs.items) |out| {
        const want_name = try std.fmt.allocPrint(gpu.gpa, "expected_{s}", .{out.name});
        defer gpu.gpa.free(want_name);
        const want = try fx.bytes(want_name);
        defer gpu.gpa.free(want);
        const got = try check.download(gpu, buffers.items[out.index]);
        defer gpu.gpa.free(got);
        try check.sameBytes(out.name, got, want);
    }
    const cfg = kernel.config(dims);
    check.pass("BITEXACT triton {s}: {d} outputs equal to Python's; {d} params (+2 scratch), grid ({d},{d},{d}), block {d}, shared {d}, divisible-by-16 args {any}", .{ meta.value.name, outputs.items.len, args.count - 2, cfg.grid.x, cfg.grid.y, cfg.grid.z, cfg.block.x, cfg.shared, divisible.items });
}
