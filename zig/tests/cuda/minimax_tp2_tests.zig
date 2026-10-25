//! MiniMax M3.1's TP2 weight plumbing over the REAL EXL3 pack (Stage-5, part 1 — receipts only):
//! both ranks read the same checkpoint through admission + the whole Stage-2 plan, shard.zig lays
//! the load out (rank 0: layers 0..N/2 plus the embed/final-norm/head; rank 1: the rest), and every
//! owned EXL3 group and passthrough tensor uploads to device buffers through the mainline
//! exl3_upload.Uploader — the trellis words and scales the exl3 kernels read in place, no decode on
//! upload. The ranks then allgather their checkpoint-byte shares over NCCL: their sum must equal the
//! whole plan's bytes exactly. Prints per-class bytes, upload seconds, device-buffer totals and free
//! RAM per rank — the Stage-5 memory receipts. No forward here: the launch graph is the remaining
//! attach (cuda_native.zig's boundary words it), so `open()` still refuses and must.
//!
//!   tf-cuda-test minimax-m3-tp2 <rank 0|1> <host:port> <pack dir>
//!
//! rank 0 serves the NCCL id on host:port (the exl3-tp2 precedent); the harness passes rank/addr
//! explicitly — the serving form reads the TF_MINIMAXM3_RANK/TF_MINIMAXM3_WORLD env (shard.zig).

const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const cuda = @import("cuda");
const check = @import("check.zig");
const m3 = @import("minimax_m3_vl_cuda");
const exl3_load = @import("core").exl3_load;

const exl3_upload = cuda.exl3_upload;

/// The swigluoai contract's clamp (serve.zig's `limit`), staged into the Layer metadata; this stage
/// never executes the experts kernel — the family's tf_m3_swigluoai epilogue drives it when attached.
const limit: f32 = 7.0;
/// fp32 epilogue accumulation (M3's contract; 0 would be GLM's bf16-rounding route).
const act_mode: i32 = 1;

pub fn run(gpu: check.Gpu, rank_arg: []const u8, addr_arg: []const u8, dir: []const u8) !void {
    if (dir.len == 0 or std.mem.eql(u8, dir, "none")) {
        if (std.mem.eql(u8, rank_arg, "0")) std.debug.print("SKIP minimax.m3.tp2: give the real EXL3 pack dir (rank 1's dir arg is ignored)\n", .{});
        return;
    }
    const rank = std.fmt.parseInt(u8, rank_arg, 10) catch {
        std.debug.print("usage: tf-cuda-test minimax-m3-tp2 <rank 0|1> <host:port> <pack dir>\n", .{});
        return error.BadArguments;
    };
    if (rank > 1 or addr_arg.len == 0) {
        std.debug.print("usage: tf-cuda-test minimax-m3-tp2 <rank 0|1> <host:port> <pack dir>\n", .{});
        return error.BadArguments;
    }

    // -- the TP2 boot: the GLM lane's recipe, harness form (rank 0 serves the NCCL id on host:port) --
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    const id_bytes = std.mem.asBytes(&id);
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try serve(addr_arg, id_bytes);
    } else {
        try receive(gpu, addr_arg, id_bytes);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, @intCast(rank)), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    // -- the whole plan, from the family's own admission (config + trellis widths + every tensor) --
    const t0 = check.now(gpu.io); // monotonic nanoseconds
    var admitted = try m3.admitDir(gpu.gpa, gpu.io, dir);
    defer admitted.deinit();
    const admit_s = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    const c = &admitted.c;
    const load = &admitted.load;
    const groups = admitted.groups.groups;

    const device_before = cuda.usage(false).device;
    const ram_before = hostFree(gpu);

    // -- the reader: one checkpoint scan, group prefix -> core group index (parts come read-only) --
    var scan = try exl3_load.scan(gpu.gpa, gpu.io, dir);
    defer scan.deinit();
    var idx = std.StringHashMapUnmanaged(usize).empty;
    defer idx.deinit(gpu.gpa);
    for (scan.groups, 0..) |g, i| try idx.put(gpu.gpa, g.prefix, i);

    // -- the upload, item by item, through the mainline Uploader (weights the exl3 kernels read) --
    var arena_state = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    // routed experts collect per owned MoE layer, then upload as one layer() call each
    const Projs = std.ArrayListUnmanaged(exl3_upload.Proj);
    const LayerProjs = struct { gate: Projs, up: Projs, down: Projs };
    var exps = try a.alloc(?LayerProjs, c.layers);
    for (exps) |*e| e.* = null;

    var owned_items: usize = 0;
    var owned_bytes: u64 = 0;
    var counts = [_]Stat{ .{}, .{}, .{}, .{}, .{}, .{} }; // attn, dense_mlp, shared, routed, head, passthrough
    var max_buffers: usize = 64;
    for (load.items) |*item| {
        if (!m3.rank_plan.owns(item, c.layers, 2, @intCast(rank))) continue;
        owned_items += 1;
        owned_bytes += item.bytes;
        max_buffers += item.bytes / (64 << 20) + @as(usize, switch (item.kind) {
            .group => 4, // trellis + two scales + bias
            .passthrough => 1,
        });
    }
    var up: exl3_upload.Uploader = .{ .d = gpu.d, .gpa = gpu.gpa, .max_buffers = max_buffers };
    defer up.deinit();

    const t1 = check.now(gpu.io);
    for (load.items) |*item| {
        if (!m3.rank_plan.owns(item, c.layers, 2, @intCast(rank))) continue;
        switch (item.kind) {
            .passthrough => {
                const t = try scan.ck.get(item.name);
                _ = try up.dev(t.bytes);
                counts[5].add(item.bytes);
            },
            .group => |gi| {
                const fg = groups[gi]; // the family's classified group
                const core_gi = idx.get(item.name) orelse {
                    std.debug.print("FAIL minimax.m3.tp2 rank {d}: no core group for {s}\n", .{ rank, item.name });
                    return error.MissingTensor;
                };
                const parts = try scan.parts(a, core_gi);
                if (fg.class == .routed) {
                    const slot = &exps[fg.layer];
                    if (slot.* == null) slot.* = .{ .gate = .empty, .up = .empty, .down = .empty };
                    const proj: *Projs = switch (fg.proj) {
                        0 => &slot.*.?.gate,
                        1 => &slot.*.?.up,
                        else => &slot.*.?.down,
                    };
                    try proj.append(a, .{
                        .trellis = parts.trellis.bytes,
                        .suh = parts.suh.bytes,
                        .svh = parts.svh.bytes,
                        .k2 = fg.k2(),
                        .signs = parts.suh_packed,
                    });
                    counts[3].add(item.bytes);
                } else {
                    _ = try up.dense(parts, fg);
                    counts[switch (fg.class) {
                        .attn => 0,
                        .dense_mlp => 1,
                        .shared => 2,
                        .head => 4,
                        else => unreachable, // .routed handled above
                    }].add(item.bytes);
                }
            },
        }
    }
    // each owned MoE layer's collected experts upload as one grouped layer (the k2 widths are the
    // headers' half-bit counts, packed su/sv scales expand in the uploader)
    var moe_layers: usize = 0;
    const dims: usize = c.hidden;
    const width: usize = firstOwnedWidth(groups, c.layers, rank) orelse {
        std.debug.print("FAIL minimax.m3.tp2 rank {d}: no owned routed gate group to read the act width from\n", .{rank});
        return error.BadExperts;
    };
    for (exps, 0..) |*e, d| {
        if (e.*) |*lp| {
            if (lp.gate.items.len != c.experts or lp.up.items.len != c.experts or lp.down.items.len != c.experts) {
                std.debug.print("FAIL minimax.m3.tp2 rank {d}: layer {d} collected gate {d} up {d} down {d}, expected {d} experts each\n", .{ rank, d, lp.gate.items.len, lp.up.items.len, lp.down.items.len, c.experts });
                return error.BadExperts;
            }
            if (d == c.dense_layers) { // the pack's first MoE layer proves the projections' roles
                const gate = lp.gate.items[0];
                const down = lp.down.items[0];
                if (gate.suh.len != dims * 2 or down.suh.len == 0 or down.svh.len != dims * 2) {
                    std.debug.print("FAIL minimax.m3.tp2 rank {d}: layer {d} expert scales do not spell hidden {d}\n", .{ rank, d, dims });
                    return error.BadExperts;
                }
            }
            _ = try exl3_upload.layer(&up, a, dims, width, .mul1, limit, act_mode, lp.gate.items, lp.up.items, lp.down.items);
            moe_layers += 1;
        }
    }
    const upload_s = @as(f64, @floatFromInt(check.now(gpu.io) - t1)) / 1e9;

    // -- the collective receipt: the shares allgather, their sum must be the whole plan's bytes --
    const mine: u64 = owned_bytes;
    var send = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&[1]u64{mine}));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(gpu.d, 16);
    defer recv.free();
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, 1, .u64, comm, s.handle), "ncclAllGather");
    try s.synchronize();
    const got = try a.alloc(u8, 16);
    try recv.download(0, got);
    // the allgather's slots are ordered by rank: slot r holds rank r's share, so the peer's share
    // sits at slot (1 - rank) — rank 0 reads slot 1, rank 1 reads slot 0
    const peer_slot: usize = if (rank == 0) 8 else 0;
    var view: [8]u8 = undefined;
    @memcpy(&view, got[peer_slot..][0..8]);
    const other = std.mem.readInt(u64, &view, .little);
    const owned_moe_expected: usize = blk: {
        const r = m3.rank_plan.range(c.layers, 2, @intCast(rank));
        var n: usize = 0;
        for (r.first..r.first + r.count) |l| {
            if (l >= c.dense_layers) n += 1;
        }
        break :blk n;
    };
    const whole = load.bytes();

    const device_after = cuda.usage(false).device;
    const ram_after = hostFree(gpu);
    const gib = struct {
        fn f(v: u64) f64 {
            return @as(f64, @floatFromInt(v)) / (1024.0 * 1024.0 * 1024.0);
        }
    }.f;
    std.debug.print("M3.TP2 rank {d}: admission {d:.1}s, groups {d}, plan items {d}, whole ckpt {d} bytes ({d:.2} GiB)\n", .{ rank, admit_s, groups.len, load.items.len, whole, gib(whole) });
    std.debug.print("M3.TP2 rank {d}: owns {d} items, {d} ckpt bytes ({d:.2} GiB), layers [{d}, {d})\n", .{ rank, owned_items, mine, gib(mine), m3.rank_plan.range(c.layers, 2, @intCast(rank)).first, m3.rank_plan.range(c.layers, 2, @intCast(rank)).first + m3.rank_plan.range(c.layers, 2, @intCast(rank)).count });
    for ([_][]const u8{ "attn", "dense_mlp", "shared", "routed", "head", "passthrough" }, &counts) |name, *st| {
        std.debug.print("M3.TP2 rank {d}:   {s:>11} items {d:>5} ckpt bytes {d:>14} ({d:.2} GiB)\n", .{ rank, name, st.items, st.bytes, gib(st.bytes) });
    }
    std.debug.print("M3.TP2 rank {d}:   moe layers grouped {d} of {d} owned expected ({d} across both ranks); experts per layer {d}\n", .{ rank, moe_layers, owned_moe_expected, c.layers - c.dense_layers, c.experts });
    std.debug.print("M3.TP2 rank {d}:   layer() width {d}; Uploader device buffers {d} bytes ({d:.2} GiB)\n", .{ rank, width, up.bytes, gib(up.bytes) });
    std.debug.print("M3.TP2 rank {d}: upload took {d:.1}s; cuda device usage delta {d} bytes ({d:.2} GiB); host MemAvailable {d:.2} -> {d:.2} GiB\n", .{ rank, upload_s, device_after - device_before, gib(device_after - device_before), gib(ram_before), gib(ram_after) });
    std.debug.print("M3.TP2 rank {d}: shares {d} + {d} = {d} vs whole {d}\n", .{ rank, mine, other, mine + other, whole });
    if (mine + other != whole) {
        if (rank == 0) std.debug.print("FAIL minimax.m3.tp2: the ranks' shares do not tile the whole plan\n", .{});
        return error.TestFailed;
    }
    if (moe_layers != owned_moe_expected) {
        if (rank == 0) std.debug.print("FAIL minimax.m3.tp2: not every owned MoE layer grouped its experts\n", .{});
        return error.TestFailed;
    }
    if (rank == 0) std.debug.print("RESULT minimax.m3.tp2: both ranks' shares tile the plan; every owned group and passthrough tensor is on device buffers\n", .{});
}

/// The grouped experts' act width: the first owned layer's routed gate group's n (w1/w2's outputs,
/// w3's inputs — the same number the grouped kernel's epilogue holds).
fn firstOwnedWidth(groups: anytype, layers: u32, rank: u8) ?u32 {
    const r = m3.rank_plan.range(layers, 2, @intCast(rank));
    for (groups) |g| {
        if (g.class != .routed or g.proj != 0) continue;
        if (g.layer >= r.first and g.layer < r.first + r.count) return g.n;
    }
    return null;
}

const Stat = struct {
    items: u64 = 0,
    bytes: u64 = 0,

    fn add(st: *Stat, b: u64) void {
        st.items += 1;
        st.bytes += b;
    }
};

/// Free MemAvailable from /proc/meminfo (0 elsewhere — the receipt prints GB10 nodes' figures).
fn hostFree(gpu: check.Gpu) u64 {
    var f = std.Io.Dir.openFileAbsolute(gpu.io, "/proc/meminfo", .{}) catch return 0;
    defer f.close(gpu.io);
    var buf: [8192]u8 = undefined;
    const n = f.readPositionalAll(gpu.io, &buf, 0) catch return 0;
    const line = std.mem.indexOf(u8, buf[0..n], "MemAvailable:") orelse return 0;
    var it = std.mem.tokenizeScalar(u8, buf[line + "MemAvailable:".len .. n], ' ');
    const kb = it.next() orelse return 0;
    return (std.fmt.parseInt(u64, kb, 10) catch 0) * 1024; // /proc/meminfo reports kB, not bytes
}

// --- the NCCL id exchange, verbatim the exl3-tp2 precedent (rank 0 serves on host:port, rank 1 reads) ---

fn serve(addr_text: []const u8, bytes: []const u8) !void {
    if (builtin.os.tag == .windows) return error.SocketFailed;
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = std.Io.Threaded.addressToPosix(&ip, &storage);
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

fn receive(gpu: check.Gpu, addr_text: []const u8, bytes: []u8) !void {
    if (builtin.os.tag == .windows) return error.SocketFailed;
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var fd: posix.socket_t = -1;
    defer {
        if (fd >= 0) _ = posix.system.close(fd);
    }
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
                if (n == 0) return error.PeerClosed;
                off += @intCast(n);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}