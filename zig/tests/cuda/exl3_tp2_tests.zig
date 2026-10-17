//! EXL3 tensor parallel: each rank's shard of real RED-SNOW projections and experts vs the engine's, all-gathered.
const std = @import("std");
const posix = std.posix;
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3 = cuda.exl3;
const exl3e = cuda.exl3_experts;
const exl3t = @import("exl3_tests.zig");
const Gpu = check.Gpu;
const Rig = quant_ref_tests.RigT;

/// NCCL's 128-byte unique id across the cable, rank 0 serving it on host:port (the benchmark starts rank 1 first).
pub fn run(gpu: Gpu, rank_arg: []const u8, addr: []const u8, dir: []const u8) !void {
    const rank = try std.fmt.parseInt(u8, rank_arg, 10);
    if (rank > 1) {
        std.debug.print("usage: tf-cuda-test exl3-tp2 <rank 0|1> <host:port> <fixture_dir|none>\n", .{});
        return error.BadArguments;
    }
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    const id_bytes = std.mem.asBytes(&id);
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try serve(addr, id_bytes);
    } else {
        try receive(gpu, addr, id_bytes);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, @intCast(rank)), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);

    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    var exp = try exl3e.Experts.load(gpu.d);
    defer exp.unload();

    var failed = false;
    {
        var ok = true;
        commSanity(&r, &lib, comm) catch |e| {
            fail0(rank, "exl3.tp2.comm", e);
            ok = false;
        };
        if (!ok) failed = true;
    }
    if (std.mem.eql(u8, dir, "none")) {
        if (rank == 0) {
            std.debug.print("SKIP exl3.tp2.linear\n", .{});
            std.debug.print("SKIP exl3.tp2.experts\n", .{});
        }
    } else {
        linearChecks(&r, &lib, comm, &lin, dir, rank, &failed);
        expertsCheck(&r, &lib, comm, &exp, dir, rank, &failed);
    }
    if (failed) return error.TestFailed;
    std.debug.print("RESULT exl3.tp2 rank {d}: every collective and comparison agreed\n", .{rank});
}

/// One f32 all-gather: the communicator moves bits before anything else is trusted.
fn commSanity(r: *Rig, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm) !void {
    const one = [_]u8{ 0, 0, 0x80, 0x3f };
    const send = try r.dev(&one);
    const recv = try r.zeros(8);
    try lib.check(lib.api.ncclAllGather(send, recv, 1, .f32, comm, r.s.handle), "ncclAllGather");
    try r.s.synchronize();
    const got = try r.back(recv, 8);
    try same(0, "exl3.tp2.comm all-gather", got, &.{ 0, 0, 0x80, 0x3f, 0, 0, 0x80, 0x3f });
}

fn fail0(rank: u8, id: []const u8, e: anyerror) void {
    if (rank == 0) std.debug.print("FAIL {s}: {t}\n", .{ id, e });
}

/// Equal bytes, or the first difference; the FAIL line is rank 0's (the contract's one printer).
fn same(rank: u8, what: []const u8, got: []const u8, want: []const u8) !void {
    if (std.mem.eql(u8, got, want)) return;
    if (rank == 0) {
        if (got.len != want.len) {
            std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ what, got.len, want.len });
        } else {
            std.debug.print("FAIL {s}: first difference at byte {d}\n", .{ what, std.mem.indexOfDiff(u8, got, want).? });
        }
    }
    return error.TestFailed;
}

fn bitsOf(v: std.json.Value) u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => unreachable,
    };
}

fn codebook(name: []const u8) exl3.Codebook {
    if (std.mem.eql(u8, name, "mcg")) return .mcg;
    if (std.mem.eql(u8, name, "inst3")) return .inst3;
    return .mul1;
}

fn tp2Section(r: *Rig, dir: []const u8) !std.json.ObjectMap {
    const text = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, "manifest.json" }), r.a, .limited(1 << 24));
    var parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    return parsed.value.object.get("tp2").?.object;
}

/// The two fixture linears, the out-split ('dim1') and the in-split ('row') one: each reports its own line.
fn linearChecks(r: *Rig, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, lin: *const exl3.Linear, dir: []const u8, rank: u8, failed: *bool) void {
    const tp2 = tp2Section(r, dir) catch |e| {
        fail0(rank, "exl3.tp2.linear", e);
        failed.* = true;
        return;
    };
    var all = true;
    for (tp2.get("linears").?.array.items) |item| {
        const o = item.object;
        const id = std.fmt.allocPrint(r.a, "exl3.tp2.linear.{s}", .{o.get("id").?.string}) catch unreachable;
        var ok = true;
        linearCheck(r, lib, comm, lin, dir, rank, o, id) catch |e| {
            if (e != error.TestFailed) fail0(rank, id, e);
            ok = false;
        };
        if (ok and rank == 0) check.pass("{s}", .{id});
        if (!ok) all = false;
    }
    if (all and rank == 0) check.pass("exl3.tp2.linear", .{});
    if (!all) failed.* = true;
}

/// One projection: the rank's trellis/scale shard of the EXL3_RULES cut, its output, and the all-gather.
fn linearCheck(r: *Rig, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, lin: *const exl3.Linear, dir: []const u8, rank: u8, o: std.json.ObjectMap, id: []const u8) !void {
    const dim1 = std.mem.eql(u8, o.get("split").?.string, "dim1");
    const k: usize = @intCast(o.get("k").?.integer);
    const n: usize = @intCast(o.get("n").?.integer);
    const bits: u32 = bitsOf(o.get("bits").?);
    const halves: usize = 2 * bits;
    const kt = k / 16;
    const nt = n / 16;
    const tw = halves * 16; // bytes a trellis tile (8 * halves i16 values)
    const full = try exl3t.readFile(r, dir, o.get("trellis").?.string);
    const suh_full = try exl3t.readFile(r, dir, o.get("suh").?.string);
    const svh_full = try exl3t.readFile(r, dir, o.get("svh").?.string);
    var k_use = k;
    var n_use = n;
    // dim1: trellis tile columns and svh rows in half, suh whole; row: tile rows, suh rows and x's columns
    const trellis = if (dim1) try cutDim1(r, full, kt, nt, tw, rank) else full[rank * (kt / 2) * nt * tw ..][0 .. (kt / 2) * nt * tw];
    const suh = if (dim1) suh_full else suh_full[rank * (k / 2) * 2 ..][0 .. (k / 2) * 2];
    const svh = if (dim1) svh_full[rank * (n / 2) * 2 ..][0 .. (n / 2) * 2] else svh_full;
    if (dim1) n_use = n / 2 else k_use = k / 2;
    const biasv = o.get("bias").?;
    var bias: u64 = 0;
    if (std.meta.activeTag(biasv) == .string) {
        const bfull = try exl3t.readFile(r, dir, biasv.string);
        bias = try r.dev(if (dim1) bfull[rank * (n / 2) * 2 ..][0 .. (n / 2) * 2] else bfull);
    }
    const st = exl3.strides(k_use, n_use, halves, false);
    const w: exl3.Weight = .{
        .trellis = try r.dev(trellis),
        .suh = try r.dev(suh),
        .svh = try r.dev(svh),
        .bias = bias,
        .k = @intCast(k_use),
        .n = @intCast(n_use),
        .halves = @intCast(halves),
        .codebook = codebook(o.get("codebook").?.string),
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    const counters = try exl3t.zeroed(r, exl3.Linear.counterBytes(w));
    const ydt: exl3.DType = if (dim1) .bf16 else .f32;
    const esz: usize = if (dim1) 2 else 4;
    const dt: cuda.nccl.DataType = if (dim1) .bf16 else .f32;
    const xmap = o.get("x").?.object;

    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const what = try std.fmt.allocPrint(r.a, "{s} m={d}", .{ id, m });
        const xf = try exl3t.readFile(r, dir, xmap.get(key).?.string);
        const x_use = if (dim1) xf else try cutDim1(r, xf, m, k, 2, rank);
        const count = m * n_use; // elements this rank all-gathers
        const send = try r.zeros(count * esz);
        const xh = try r.zeros(m * k_use * 2);
        const z: u64 = if (exl3.plan(k_use, n_use, 192, 8).sk > 1) try r.zeros(exl3.Linear.zBytes(w, m)) else 0;
        try lin.matmul(r.s, w, try r.dev(x_use), .bf16, m, send, ydt, xh, z, counters);
        const recv = try r.zeros(2 * count * esz);
        try lib.check(lib.api.ncclAllGather(send, recv, count, dt, comm, r.s.handle), "ncclAllGather");
        try r.s.synchronize();
        const got = try r.back(recv, 2 * count * esz);
        if (dim1) {
            // the two ranks' bf16 halves concatenated are the full output
            const it = o.get("shards").?.object.get(key).?.array.items;
            const want = try std.mem.concat(r.a, u8, &.{
                try exl3t.readFile(r, dir, it[0].string),
                try exl3t.readFile(r, dir, it[1].string),
            });
            try same(rank, what, got, want);
        } else {
            // fp32 partials bit for bit, then bf16(sum rank 0 first) as forward.py's hc_post rounds
            const it = o.get("partials").?.object.get(key).?.array.items;
            try same(rank, what, got[0 .. m * n * 4], try exl3t.readFile(r, dir, it[0].string));
            try same(rank, what, got[m * n * 4 ..], try exl3t.readFile(r, dir, it[1].string));
            const want = try r.a.alloc(u8, m * n * 2);
            for (0..m * n) |i| {
                const g0 = std.mem.bytesToValue(f32, got[4 * i ..][0..4]);
                const g1 = std.mem.bytesToValue(f32, got[4 * (m * n + i) ..][0..4]);
                std.mem.writeInt(u16, want[2 * i ..][0..2], bf16r(g0 + g1), .little);
            }
            try same(rank, what, want, try exl3t.readFile(r, dir, o.get("y").?.object.get(key).?.string));
        }
    }
}

/// The routed experts of one MoE layer: each rank's gate/up/down shards, its fp32 partials, the all-gathered sum.
fn expertsCheck(r: *Rig, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, exp: *const exl3e.Experts, dir: []const u8, rank: u8, failed: *bool) void {
    const id = "exl3.tp2.experts";
    const o = blk: {
        const tp2 = tp2Section(r, dir) catch |e| {
            fail0(rank, id, e);
            failed.* = true;
            return;
        };
        break :blk tp2.get("experts").?.object;
    };
    var ok = true;
    expertsRun(r, lib, comm, exp, dir, rank, o, id) catch |e| {
        if (e != error.TestFailed) fail0(rank, id, e);
        ok = false;
    };
    if (ok and rank == 0) check.pass("{s}", .{id});
    if (!ok) failed.* = true;
}

fn expertsRun(r: *Rig, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, exp: *const exl3e.Experts, dir: []const u8, rank: u8, o: std.json.ObjectMap, id: []const u8) !void {
    const e_count: usize = @intCast(o.get("count").?.integer);
    const dims: usize = @intCast(o.get("dims").?.integer);
    const width: usize = @intCast(o.get("width").?.integer); // the full layer's; each rank holds width / 2
    const slots: usize = @intCast(o.get("slots").?.integer);
    const trellis = o.get("trellis").?.object;
    var gate_p = try r.a.alloc(u64, e_count);
    var up_p = try r.a.alloc(u64, e_count);
    var down_p = try r.a.alloc(u64, e_count);
    var gk2 = try r.a.alloc(i32, e_count);
    var uk2 = try r.a.alloc(i32, e_count);
    var dk2 = try r.a.alloc(i32, e_count);
    for (o.get("gate_k2").?.array.items, 0..) |v, i| gk2[i] = @intCast(v.integer);
    for (o.get("up_k2").?.array.items, 0..) |v, i| uk2[i] = @intCast(v.integer);
    for (o.get("down_k2").?.array.items, 0..) |v, i| dk2[i] = @intCast(v.integer);
    // gate/up: trellis tile columns and svh rows in half; down: tile rows and suh rows in half
    for (0..e_count) |i| {
        const gw = @as(usize, @intCast(gk2[i])) * 16; // bytes a gate/up tile
        const dw = @as(usize, @intCast(dk2[i])) * 16;
        gate_p[i] = try r.dev(try cutDim1(r, try exl3t.readFile(r, dir, trellis.get("gate").?.array.items[i].string), dims / 16, width / 16, gw, rank));
        up_p[i] = try r.dev(try cutDim1(r, try exl3t.readFile(r, dir, trellis.get("up").?.array.items[i].string), dims / 16, width / 16, gw, rank));
        const dfull = try exl3t.readFile(r, dir, trellis.get("down").?.array.items[i].string);
        down_p[i] = try r.dev(dfull[rank * (width / 32) * (dims / 16) * dw ..][0 .. (width / 32) * (dims / 16) * dw]);
    }
    const layer: exl3e.Layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p)),
        .up = try r.dev(std.mem.sliceAsBytes(up_p)),
        .down = try r.dev(std.mem.sliceAsBytes(down_p)),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2)),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2)),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2)),
        .suh_g = try r.dev(try exl3t.readFile(r, dir, o.get("suh_g").?.string)),
        .suh_u = try r.dev(try exl3t.readFile(r, dir, o.get("suh_u").?.string)),
        .svh_g = try r.dev(try cutDim1(r, try exl3t.readFile(r, dir, o.get("svh_g").?.string), e_count, width, 2, rank)),
        .svh_u = try r.dev(try cutDim1(r, try exl3t.readFile(r, dir, o.get("svh_u").?.string), e_count, width, 2, rank)),
        .suh_d = try r.dev(try cutDim1(r, try exl3t.readFile(r, dir, o.get("suh_d").?.string), e_count, width, 2, rank)),
        .svh_d = try r.dev(try exl3t.readFile(r, dir, o.get("svh_d").?.string)),
        .experts = e_count,
        .dims = dims,
        .width = width / 2,
        .codebook = codebook(o.get("codebook").?.string),
        .limit = @floatCast(o.get("limit").?.float),
        .act_mode = @intCast(o.get("act_mode").?.integer),
    };
    const xmap = o.get("x").?.object;
    const pickmap = o.get("pick").?.object;
    const wtsmap = o.get("wts").?.object;
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const what = try std.fmt.allocPrint(r.a, "{s} m={d}", .{ id, m });
        const dx = try r.dev(try exl3t.readFile(r, dir, xmap.get(key).?.string));
        const dpick = try r.dev(try exl3t.readFile(r, dir, pickmap.get(key).?.string));
        const dwts = try r.dev(try exl3t.readFile(r, dir, wtsmap.get(key).?.string));
        const sb = try r.zeros(try exl3e.scratchBytes(layer, m, slots));
        const sc = try exl3e.scratch(sb, layer, m, slots);
        const out = try r.zeros(m * dims * 4);
        try exp.run(r.s, layer, dx, dims, .bf16, dpick, dwts, m, out, sc);
        const recv = try r.zeros(2 * m * dims * 4);
        try lib.check(lib.api.ncclAllGather(out, recv, m * dims, .f32, comm, r.s.handle), "ncclAllGather");
        try r.s.synchronize();
        const got = try r.back(recv, 2 * m * dims * 4);
        const it = o.get("partials").?.object.get(key).?.array.items;
        try same(rank, what, got[0 .. m * dims * 4], try exl3t.readFile(r, dir, it[0].string));
        try same(rank, what, got[m * dims * 4 ..], try exl3t.readFile(r, dir, it[1].string));
        const want = try r.a.alloc(u8, m * dims * 2);
        for (0..m * dims) |i| {
            const g0 = std.mem.bytesToValue(f32, got[4 * i ..][0..4]);
            const g1 = std.mem.bytesToValue(f32, got[4 * (m * dims + i) ..][0..4]);
            std.mem.writeInt(u16, want[2 * i ..][0..2], bf16r(g0 + g1), .little);
        }
        try same(rank, what, want, try exl3t.readFile(r, dir, o.get("y").?.object.get(key).?.string));
    }
}

/// The rank's half of an even 2-D row-major tensor's second axis, repacked contiguous (split.py's dim1 cut).
fn cutDim1(r: *Rig, src: []const u8, rows: usize, cols: usize, item: usize, rank: u8) ![]u8 {
    const per = cols / 2 * item;
    const out = try r.a.alloc(u8, rows * per);
    for (0..rows) |i| {
        @memcpy(out[i * per ..][0..per], src[i * cols * item + @as(usize, rank) * per ..][0..per]);
    }
    return out;
}

/// bf16 round-to-nearest-even of an fp32 sum (hc_post's `acc.to(torch.bfloat16)`).
fn bf16r(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// Rank 0 binds host:port and sends its unique id to the one peer (70 s for rank 1's retries).
fn serve(addr_text: []const u8, bytes: []const u8) !void {
    const a = try parseAddr(addr_text);
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = try posixLen(a.host, a.port, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    const one: c_int = 1;
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
    switch (posix.errno(posix.system.bind(fd, &storage.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.BindFailed,
    }
    if (posix.errno(posix.system.listen(fd, 1)) != .SUCCESS) return error.ListenFailed;
    var waited: u64 = 0;
    while (true) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.AcceptFailed;
        if (ready > 0) break;
        waited += 500;
        if (waited > 70_000) return error.PeerTimeout;
    }
    var peer: std.Io.Threaded.PosixAddress = undefined;
    var peer_len: posix.socklen_t = @sizeOf(std.Io.Threaded.PosixAddress);
    const rc2 = posix.system.accept(fd, &peer.any, &peer_len);
    if (posix.errno(rc2) != .SUCCESS) return error.AcceptFailed;
    const cfd: posix.socket_t = @intCast(rc2);
    defer _ = posix.system.close(cfd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = posix.system.write(cfd, bytes.ptr + off, bytes.len - off);
        if (posix.errno(n) != .SUCCESS) return error.WriteFailed;
        off += @intCast(n);
    }
}

/// Rank 1 connects to host:port, retrying for 60 s (rank 0 may not have bound yet), and reads the id.
fn receive(gpu: Gpu, addr_text: []const u8, bytes: []u8) !void {
    const a = try parseAddr(addr_text);
    const ip = try std.Io.net.IpAddress.parseIp4(a.host, a.port);
    var fd: posix.socket_t = -1;
    defer {
        if (fd >= 0) _ = posix.system.close(fd);
    }
    var waited: u64 = 0;
    while (fd < 0) {
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&ip, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const c: posix.socket_t = @intCast(rc);
        if (posix.errno(posix.system.connect(c, &storage.any, len)) == .SUCCESS) {
            fd = c;
        } else {
            _ = posix.system.close(c);
            waited += 200;
            if (waited > 60_000) return error.PeerTimeout;
            std.Io.sleep(gpu.io, .fromMilliseconds(200), .awake) catch {};
        }
    }
    var off: usize = 0;
    var read_waited: u64 = 0;
    while (off < bytes.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.ReadFailed;
        if (ready == 0) {
            read_waited += 500;
            if (read_waited > 60_000) return error.PeerTimeout;
            continue;
        }
        const n = posix.system.read(fd, bytes.ptr + off, bytes.len - off);
        switch (posix.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerClosed; // the peer died before its id arrived
                off += @intCast(n);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}

fn posixLen(host: []const u8, port: u16, storage: *std.Io.Threaded.PosixAddress) !posix.socklen_t {
    const ip = try std.Io.net.IpAddress.parseIp4(host, port);
    return std.Io.Threaded.addressToPosix(&ip, storage);
}

fn parseAddr(text: []const u8) !struct { host: []const u8, port: u16 } {
    const at = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadAddress;
    return .{ .host = text[0..at], .port = try std.fmt.parseInt(u16, text[at + 1 ..], 10) };
}
