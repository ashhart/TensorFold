//! A group's small all-gathers as one-shot RoCE writes after b12x's RoCEnante (local-inference-lab/b12x, Apache-2.0):
//! written for two ranks, since generalised to any world whose ranks all reach each other (a switch; a ring's far ranks
//! do not), with one RC queue pair a peer an HCA, a receive slot a peer and a flag a peer an HCA.

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");
const tp_group = @import("tp_group.zig");
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");
const nap = @import("roce_nap.zig");

const max_hcas = 2;
pub const max_world = 8; // ranks a group may have on this path (GB10 kits are 2 or 4)
const max_peers = max_world - 1;
const slots = 2;
const flag_stride = 128;
const send_depth = 256;
const port_num: u8 = 1;
const threads = 512;
const max_blocks = 8; // the largest grid; smaller shards launch fewer (a power of two)
const classes = 4; // grid size classes 1, 2, 4, 8 (each its own arrival counters)
const idle_spins: u64 = 20_000_000; // doorbell polls before the proxy naps between them (a model's gaps are short)
const spin_limit: u32 = 100_000_000; // the kernel's flag polls before it gives up and poisons the exchange
const probe_naps: u32 = 250_000; // 20 us naps a connect probe waits for every peer's token (about 5 s) before refusing

// -- libibverbs (the ABI of rdma-core's verbs.h; offsets checked against its headers) ---------------------------------
const Device = opaque {};
const Context = opaque {};
const Pd = opaque {};
const Cq = extern struct { context: *Context };
const Mr = extern struct { context: *Context, pd: *Pd, addr: ?*anyopaque, length: usize, handle: u32, lkey: u32, rkey: u32 };
const Qp = extern struct { context: *Context, qp_context: ?*anyopaque, pd: *Pd, send_cq: *Cq, recv_cq: *Cq, srq: ?*anyopaque, handle: u32, qp_num: u32 };
const QpCap = extern struct { max_send_wr: u32 = 0, max_recv_wr: u32 = 0, max_send_sge: u32 = 0, max_recv_sge: u32 = 0, max_inline_data: u32 = 0 };
const QpInitAttr = extern struct { qp_context: ?*anyopaque = null, send_cq: ?*Cq = null, recv_cq: ?*Cq = null, srq: ?*anyopaque = null, cap: QpCap = .{}, qp_type: c_int = 0, sq_sig_all: c_int = 0 };
const GlobalRoute = extern struct { dgid: [16]u8 = @splat(0), flow_label: u32 = 0, sgid_index: u8 = 0, hop_limit: u8 = 0, traffic_class: u8 = 0 };
const AhAttr = extern struct { grh: GlobalRoute = .{}, dlid: u16 = 0, sl: u8 = 0, src_path_bits: u8 = 0, static_rate: u8 = 0, is_global: u8 = 0, port_num: u8 = 0 };
const QpAttr = extern struct {
    qp_state: u32 = 0,
    cur_qp_state: u32 = 0,
    path_mtu: u32 = 0,
    path_mig_state: u32 = 0,
    qkey: u32 = 0,
    rq_psn: u32 = 0,
    sq_psn: u32 = 0,
    dest_qp_num: u32 = 0,
    qp_access_flags: u32 = 0,
    cap: QpCap = .{},
    ah_attr: AhAttr = .{},
    alt_ah_attr: AhAttr = .{},
    pkey_index: u16 = 0,
    alt_pkey_index: u16 = 0,
    en_sqd_async_notify: u8 = 0,
    sq_draining: u8 = 0,
    max_rd_atomic: u8 = 0,
    max_dest_rd_atomic: u8 = 0,
    min_rnr_timer: u8 = 0,
    port_num: u8 = 0,
    timeout: u8 = 0,
    retry_cnt: u8 = 0,
    rnr_retry: u8 = 0,
    alt_port_num: u8 = 0,
    alt_timeout: u8 = 0,
    rate_limit: u32 = 0,
    _tail: u32 = 0,
};
const Sge = extern struct { addr: u64, length: u32, lkey: u32 };
const SendWr = extern struct {
    wr_id: u64 = 0,
    next: ?*SendWr = null,
    sg_list: ?*Sge = null,
    num_sge: c_int = 0,
    opcode: u32 = 0,
    send_flags: u32 = 0,
    imm_data: u32 = 0,
    remote_addr: u64 = 0,
    rkey: u32 = 0,
    _tail: [128 - 52]u8 = @splat(0),
};
const Wc = extern struct { wr_id: u64, status: u32, opcode: u32, vendor_err: u32, _rest: [48 - 20]u8 };
const PortAttr = extern struct { bytes: [56]u8 align(8) };

comptime {
    std.debug.assert(@sizeOf(QpInitAttr) == 64 and @offsetOf(QpInitAttr, "cap") == 32 and @offsetOf(QpInitAttr, "qp_type") == 52);
    std.debug.assert(@sizeOf(AhAttr) == 32 and @offsetOf(AhAttr, "dlid") == 24 and @offsetOf(AhAttr, "port_num") == 30);
    std.debug.assert(@sizeOf(QpAttr) == 144 and @offsetOf(QpAttr, "ah_attr") == 56 and @offsetOf(QpAttr, "pkey_index") == 120);
    std.debug.assert(@offsetOf(QpAttr, "rnr_retry") == 132 and @offsetOf(QpAttr, "qp_access_flags") == 32);
    std.debug.assert(@sizeOf(SendWr) == 128 and @offsetOf(SendWr, "remote_addr") == 40 and @offsetOf(SendWr, "rkey") == 48);
    std.debug.assert(@sizeOf(Wc) == 48 and @offsetOf(Mr, "lkey") == 36 and @offsetOf(Qp, "qp_num") == 52);
}

// struct ibv_context's ops: the two calls verbs.h inlines
const ops_poll_cq = 96;
const ops_post_send = 208;
const PollCq = *const fn (*Cq, c_int, [*]Wc) callconv(.c) c_int;
const PostSend = *const fn (*Qp, *SendWr, *?*SendWr) callconv(.c) c_int;

const wr_rdma_write = 0;
const send_signaled = 2;
const send_inline = 8;
const qpt_rc = 2;
const qps_init = 1;
const qps_rtr = 2;
const qps_rts = 3;
const port_active = 4;
const access_local_write = 1;
const access_remote_write = 2;
const qp_state = 1;
const qp_access_flags = 8;
const qp_pkey_index = 16;
const qp_port = 32;
const qp_av = 128;
const qp_path_mtu = 256;
const qp_timeout = 512;
const qp_retry_cnt = 1024;
const qp_rnr_retry = 2048;
const qp_rq_psn = 4096;
const qp_max_qp_rd_atomic = 8192;
const qp_min_rnr_timer = 32768;
const qp_sq_psn = 65536;
const qp_max_dest_rd_atomic = 131072;
const qp_dest_qpn = 1048576;

const Api = struct {
    ibv_get_device_list: *const fn (?*c_int) callconv(.c) ?[*]?*Device,
    ibv_free_device_list: *const fn ([*]?*Device) callconv(.c) void,
    ibv_get_device_name: *const fn (*Device) callconv(.c) ?[*:0]const u8,
    ibv_open_device: *const fn (*Device) callconv(.c) ?*Context,
    ibv_close_device: *const fn (*Context) callconv(.c) c_int,
    ibv_query_port: *const fn (*Context, u8, *PortAttr) callconv(.c) c_int,
    ibv_query_gid: *const fn (*Context, u8, c_int, *[16]u8) callconv(.c) c_int,
    ibv_alloc_pd: *const fn (*Context) callconv(.c) ?*Pd,
    ibv_dealloc_pd: *const fn (*Pd) callconv(.c) c_int,
    ibv_reg_mr: *const fn (*Pd, ?*anyopaque, usize, c_int) callconv(.c) ?*Mr,
    ibv_dereg_mr: *const fn (*Mr) callconv(.c) c_int,
    ibv_create_cq: *const fn (*Context, c_int, ?*anyopaque, ?*anyopaque, c_int) callconv(.c) ?*Cq,
    ibv_destroy_cq: *const fn (*Cq) callconv(.c) c_int,
    ibv_create_qp: *const fn (*Pd, *QpInitAttr) callconv(.c) ?*Qp,
    ibv_destroy_qp: *const fn (*Qp) callconv(.c) c_int,
    ibv_modify_qp: *const fn (*Qp, *QpAttr, c_int) callconv(.c) c_int,
    ibv_wc_status_str: *const fn (u32) callconv(.c) ?[*:0]const u8,
};

fn opsCall(comptime T: type, ctx: *Context, offset: usize) T {
    const slot: *const T = @ptrFromInt(@intFromPtr(ctx) + offset);
    return slot.*;
}

pub const Error = error{ VerbsUnavailable, NoHca, HcaFailed, PeerRefused, PeerUnreachable, ProxyFailed, ExchangeTimedOut, Misaligned };

/// TF_ROCE=0 keeps every gather on NCCL; unset or any other value asks for this path (every rank must ask).
pub fn wanted() bool {
    const v = std.c.getenv("TF_ROCE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Every rank's `mine` in rank order: the values cross the group's communicator (an NCCL all-gather).
fn exchange(comptime T: type, group: *const tp_group.Group, d: *const Driver, s: Stream, mine: T, all: *[max_world]T) ![]T {
    const out = all[0..group.world];
    try group.gather(T, d, s, mine, out);
    return out;
}

/// A rank's peers in rank order, without itself: rank `peer` is this rank's peer number `peerIndex`.
pub fn peerIndex(rank: usize, peer: usize) usize {
    return if (peer < rank) peer else peer - 1;
}

/// The rank behind peer number `p` of `rank` (the inverse of peerIndex).
pub fn senderOf(rank: usize, p: usize) usize {
    return if (p < rank) p else p + 1;
}

/// The first rank `rank` cannot reach on a fabric described by `reaches` (every rank must: a ring of four leaves its
/// far rank out), or null when every peer answers. The device probe in `Gather.probe` establishes this with one RDMA
/// write each way; this is its contract for the fabric tests.
pub fn firstUnreachable(rank: usize, world: usize, ctx: anytype, reaches: *const fn (@TypeOf(ctx), usize, usize) bool) ?usize {
    for (0..world) |peer| if (peer != rank and !reaches(ctx, rank, peer)) return peer;
    return null;
}

// -- the region's layout (the same on every rank) ---------------------------------------------------------------------
//   recv[slot][peer][slot_bytes]       each peer's shard, written by that peer's NIC (peer: the sender's peerIndex here)
//   flags[slot][peer][hca]             the sequence a peer's NIC writes after its stripe on that HCA (128 bytes apart)
//   send[slot][slot_bytes]             this rank's shard, staged for its NICs
//   ctrl                               the doorbell, sizes, error words and the connect probe's tokens
// At world 2 (one peer) every offset is the two-rank layout's, byte for byte.
const Layout = struct {
    slot_bytes: usize,
    peers: usize,
    recv: usize,
    flags: usize,
    send: usize,
    ctrl: usize,
    total: usize,

    fn of(max_bytes: usize, world: usize) Layout {
        const peers = world - 1;
        const s = std.mem.alignForward(usize, max_bytes, 4096);
        const flags = slots * peers * s;
        const send = flags + slots * peers * max_hcas * flag_stride;
        const ctrl = send + slots * s;
        return .{ .slot_bytes = s, .peers = peers, .recv = 0, .flags = flags, .send = send, .ctrl = ctrl, .total = ctrl + flag_stride };
    }

    /// Where peer number `p`'s shard of slot `slot` lands.
    fn recvAt(l: Layout, slot: usize, p: usize) usize {
        return l.recv + (slot * l.peers + p) * l.slot_bytes;
    }

    /// Where peer number `p`'s flag for HCA `h` of slot `slot` lands (`n_hca` HCAs in use: the kernel's stride).
    fn flagAt(l: Layout, slot: usize, p: usize, h: usize, n_hca: usize) usize {
        return l.flags + ((slot * l.peers + p) * n_hca + h) * flag_stride;
    }
};
// ctrl words (32 of them in flag_stride bytes)
const c_seq = 0;
const c_nbytes = 1; // + slot
const c_error = 3;
const c_hca = 4;
const c_peer = 5; // the rank whose stripe a timed-out wait missed
const c_probe = 8; // + peer * max_hcas + hca: the connect probe's token (rank + 1) from that peer on that HCA

comptime {
    std.debug.assert(c_probe + max_peers * max_hcas <= flag_stride / 4);
}

/// What a rank tells the others to connect: its region, per HCA its key, address and port, and per HCA and peer the
/// queue pair it opened toward that peer (indexed by the peer's number in this rank's list, peerIndex).
const Blob = extern struct {
    ok: u32,
    n_hca: u32,
    slot_bytes: u64,
    max_bytes: u64, // the largest shard this rank would send this way (every rank uses the smallest)
    addr: u64,
    rkey: [max_hcas]u32,
    mtu: [max_hcas]u32,
    lid: [max_hcas]u16,
    gid: [max_hcas][16]u8,
    qpn: [max_hcas][max_peers]u32,
};

const Hca = struct {
    ctx: ?*Context = null,
    pd: ?*Pd = null,
    mr: ?*Mr = null,
    cq: ?*Cq = null,
    qp: [max_peers]?*Qp = @splat(null), // one RC queue pair a peer, by peerIndex
    gid: [16]u8 = @splat(0),
    gid_index: c_int = 0,
    lid: u16 = 0,
    mtu: u32 = 0,
    outstanding: u32 = 0, // signalled writes posted on this HCA's queue pairs and not yet drained
};

pub const Gather = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    lib: platform.DynLib,
    api: Api,
    rank: u8,
    world: u8,
    hcas: [max_hcas]Hca = @splat(.{}),
    n_hca: u32 = 0,
    layout: Layout,
    region: memory.HostBuffer,
    region_dev: abi.DevicePtr,
    counters: memory.DeviceBuffer,
    module: Module,
    f: Function,
    max_bytes: usize,
    peer_addr: [max_peers]u64 = @splat(0), // by peerIndex
    peer_rkey: [max_peers][max_hcas]u32 = @splat(@splat(0)),
    last_seq: u32 = 0,
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    err: [256]u8 = @splat(0),
    gathers: u64 = 0,

    /// Every rank's exchange or null on every rank; setup crosses ``group``'s communicator on ``s``. ``max_bytes``: a
    /// shard. A rank with no RDMA path to another on the same HCA (a ring's far rank) fails the connect probe (one
    /// write each way with each peer), and every rank returns null: NCCL carries every gather, with a warning naming
    /// the rank (TF_ROCE=0 silences it).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, s: Stream, want: bool, max_bytes: usize) !?*Gather {
        if (group.world < 2) return error.PeerRefused;
        if (group.world > max_world) {
            std.log.err("RoCE all-gather: {d} ranks; this path carries at most {d}", .{ group.world, max_world });
            return error.PeerRefused;
        }
        const g = try gpa.create(Gather);
        var mine: Blob = std.mem.zeroes(Blob);
        const ok = want and blk: {
            g.setup(gpa, io, d, @intCast(group.rank), @intCast(group.world), max_bytes) catch |e| {
                std.log.warn("RoCE all-gather: {s} ({s}); NCCL carries every gather", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        if (ok) g.blob(&mine);
        var blobs: [max_world]Blob = undefined;
        const all = exchange(Blob, group, d, s, mine, &blobs) catch |e| {
            if (ok) g.teardown();
            gpa.destroy(g);
            return e;
        };
        var every = ok;
        var caps: [max_world]u64 = undefined;
        for (all, 0..) |b, r| {
            if (b.ok == 0 or b.n_hca != mine.n_hca or b.slot_bytes != mine.slot_bytes) every = false;
            caps[r] = b.max_bytes;
        }
        if (!every) {
            if (ok) g.teardown();
            gpa.destroy(g);
            return null;
        }
        g.max_bytes = agreedCap(caps[0..group.world]); // one cap for fits on every rank
        const connected = blk: {
            g.connect(all) catch |e| {
                std.log.warn("RoCE all-gather: {s} ({s}); NCCL carries every gather", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        // every rank's queue pairs are ready to receive before any rank's probe write leaves
        var words: [max_world]u32 = undefined;
        const ready = exchange(u32, group, d, s, @intFromBool(connected), &words) catch |e| {
            g.teardown();
            gpa.destroy(g);
            return e;
        };
        if (!connected or std.mem.indexOfScalar(u32, ready, 0) != null) {
            g.teardown();
            gpa.destroy(g);
            return null;
        }
        // the connect probe: a token to every peer on every HCA, our writes completed and every peer's token in
        const reached: u32 = blk: {
            g.probe() catch |e| {
                std.log.err("RoCE all-gather: {s} ({s})", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk 0;
            };
            break :blk 1;
        };
        const verdicts = exchange(u32, group, d, s, reached, &words) catch |e| {
            g.teardown();
            gpa.destroy(g);
            return e;
        };
        if (std.mem.indexOfScalar(u32, verdicts, 0)) |r| { // every rank or none, as for every other failure above
            g.teardown();
            gpa.destroy(g);
            std.log.warn("RoCE all-gather: rank {d} has no RDMA path to every other rank (its log names the peer and the HCA); NCCL carries every gather. Every rank must reach every other on the same HCA index: two on a cable, or a switch; a ring of four has none. TF_ROCE=0 keeps this quiet", .{r});
            return null;
        }
        g.startProxy() catch |e| {
            g.teardown();
            gpa.destroy(g);
            return e;
        };
        std.log.info("RoCE all-gather: {d} ranks, {d} HCA(s), shards up to {d} KiB by RDMA writes", .{ g.world, g.n_hca, g.max_bytes >> 10 });
        return g;
    }

    fn fail(g: *Gather, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(g.err[0 .. g.err.len - 1], fmt, args) catch {};
    }

    fn setup(g: *Gather, gpa: std.mem.Allocator, io: std.Io, d: *const Driver, rank: u8, world: u8, max_bytes: usize) !void {
        g.* = .{ .gpa = gpa, .d = d, .lib = undefined, .api = undefined, .rank = rank, .world = world, .layout = .of(max_bytes, world), .region = undefined, .region_dev = 0, .counters = undefined, .module = undefined, .f = undefined, .max_bytes = max_bytes };
        g.lib = platform.DynLib.open("libibverbs.so.1") catch return platform.optional_libraries.unavailable(.verbs);
        errdefer g.lib.close();
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(g.api, name) = g.lib.lookup(T, name) orelse return @import("sys").optional_libraries.unavailable(.verbs);
        }
        var names: [max_hcas][64]u8 = undefined;
        var lens: [max_hcas]usize = undefined;
        g.n_hca = try hcaNames(io, gpa, &names, &lens);
        g.region = try memory.HostBuffer.allocMapped(d, g.layout.total);
        errdefer g.region.free();
        @memset(g.region.bytes, 0);
        g.region_dev = try g.region.device();
        g.counters = try memory.DeviceBuffer.alloc(d, 64 * 4);
        errdefer g.counters.free();
        const zeros: [64 * 4]u8 = @splat(0);
        try g.counters.upload(0, &zeros);
        g.module = try Module.load(d, kernels.roce_gather);
        errdefer g.module.unload();
        g.f = try g.module.function("roce_gather");
        errdefer g.closeHcas();
        for (0..g.n_hca) |h| try g.openHca(io, gpa, h, names[h][0..lens[h]]);
    }

    fn closeHcas(g: *Gather) void {
        for (&g.hcas) |*x| {
            for (x.qp) |maybe| if (maybe) |q| {
                _ = g.api.ibv_destroy_qp(q);
            };
            if (x.cq) |q| _ = g.api.ibv_destroy_cq(q);
            if (x.mr) |m| _ = g.api.ibv_dereg_mr(m);
            if (x.pd) |p| _ = g.api.ibv_dealloc_pd(p);
            if (x.ctx) |c| _ = g.api.ibv_close_device(c);
            x.* = .{};
        }
    }

    fn openHca(g: *Gather, io: std.Io, gpa: std.mem.Allocator, h: usize, name: []const u8) !void {
        const a = &g.api;
        var num: c_int = 0;
        const list = a.ibv_get_device_list(&num) orelse return error.NoHca;
        defer a.ibv_free_device_list(list);
        var dev: ?*Device = null;
        for (0..@intCast(num)) |i| {
            const n = a.ibv_get_device_name(list[i].?) orelse continue;
            if (std.mem.eql(u8, std.mem.span(n), name)) dev = list[i];
        }
        const hca = &g.hcas[h];
        hca.ctx = a.ibv_open_device(dev orelse {
            g.fail("RDMA device {s} not found", .{name});
            return error.NoHca;
        }) orelse return error.HcaFailed;
        var port: PortAttr = std.mem.zeroes(PortAttr);
        if (a.ibv_query_port(hca.ctx.?, port_num, &port) != 0) return error.HcaFailed;
        if (std.mem.readInt(u32, port.bytes[0..4], .little) != port_active) {
            g.fail("{s} port 1 is not active", .{name});
            return error.NoHca;
        }
        hca.mtu = std.mem.readInt(u32, port.bytes[8..12], .little);
        hca.lid = std.mem.readInt(u16, port.bytes[34..36], .little);
        hca.gid_index = try gidIndex(io, gpa, name);
        if (a.ibv_query_gid(hca.ctx.?, port_num, hca.gid_index, &hca.gid) != 0) return error.HcaFailed;
        hca.pd = a.ibv_alloc_pd(hca.ctx.?) orelse return error.HcaFailed;
        hca.mr = a.ibv_reg_mr(hca.pd.?, g.region.bytes.ptr, g.region.bytes.len, access_local_write | access_remote_write) orelse {
            g.fail("{s}: registering the pinned region failed", .{name});
            return error.HcaFailed;
        };
        // one completion queue an HCA: post() keeps at most send_depth / 4 signalled writes outstanding across its
        // queue pairs (each with its unsignalled stripe ahead of it), the probe at most one a peer
        hca.cq = a.ibv_create_cq(hca.ctx.?, send_depth, null, null, 0) orelse return error.HcaFailed;
        for (0..g.world - 1) |p| { // one RC queue pair a peer, every peer's writes on their own connection
            var attr: QpInitAttr = .{ .send_cq = hca.cq, .recv_cq = hca.cq, .qp_type = qpt_rc, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 } };
            hca.qp[p] = a.ibv_create_qp(hca.pd.?, &attr) orelse return error.HcaFailed;
            var init_attr: QpAttr = .{ .qp_state = qps_init, .port_num = port_num, .qp_access_flags = access_remote_write };
            if (a.ibv_modify_qp(hca.qp[p].?, &init_attr, qp_state | qp_pkey_index | qp_port | qp_access_flags) != 0) return error.HcaFailed;
        }
    }

    fn blob(g: *Gather, b: *Blob) void {
        b.ok = 1;
        b.n_hca = g.n_hca;
        b.slot_bytes = g.layout.slot_bytes;
        b.max_bytes = g.max_bytes;
        b.addr = @intFromPtr(g.region.bytes.ptr);
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            b.rkey[h] = x.mr.?.rkey;
            b.mtu[h] = x.mtu;
            b.lid[h] = x.lid;
            b.gid[h] = x.gid;
            for (0..g.world - 1) |p| b.qpn[h][p] = x.qp[p].?.qp_num;
        }
    }

    /// Every queue pair to its peer's (the one that peer opened toward this rank, on the same HCA): RTR, then RTS.
    fn connect(g: *Gather, all: []const Blob) !void {
        for (0..g.world) |r| if (r != g.rank) {
            const p = peerIndex(g.rank, r);
            const me = peerIndex(r, g.rank); // this rank's number in that peer's list
            const peer = &all[r];
            g.peer_addr[p] = peer.addr;
            for (0..g.n_hca) |h| {
                const x = &g.hcas[h];
                g.peer_rkey[p][h] = peer.rkey[h];
                var rtr: QpAttr = .{ .qp_state = qps_rtr, .path_mtu = @min(peer.mtu[h], x.mtu), .dest_qp_num = peer.qpn[h][me], .max_dest_rd_atomic = 1, .min_rnr_timer = 12, .ah_attr = .{ .is_global = 1, .dlid = peer.lid[h], .port_num = port_num, .grh = .{ .dgid = peer.gid[h], .sgid_index = @intCast(x.gid_index), .hop_limit = 64 } } };
                if (g.api.ibv_modify_qp(x.qp[p].?, &rtr, qp_state | qp_av | qp_path_mtu | qp_dest_qpn | qp_rq_psn | qp_max_dest_rd_atomic | qp_min_rnr_timer) != 0) {
                    g.fail("HCA {d}: the queue pair toward rank {d} refused RTR", .{ h, r });
                    return error.PeerRefused;
                }
                var rts: QpAttr = .{ .qp_state = qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = 1 };
                if (g.api.ibv_modify_qp(x.qp[p].?, &rts, qp_state | qp_timeout | qp_retry_cnt | qp_rnr_retry | qp_sq_psn | qp_max_qp_rd_atomic) != 0) {
                    g.fail("HCA {d}: the queue pair toward rank {d} refused RTS", .{ h, r });
                    return error.PeerRefused;
                }
            }
        };
    }

    /// The connect probe, on the host before the proxy runs: this rank's token (rank + 1) RDMA-written into every
    /// peer's ctrl on every HCA, then every write's completion (a cable no path follows ends in a retry-exhausted
    /// completion naming the rank) and every peer's token here. Both directions of every pair are thus proven
    /// before the first gather; a ring's far rank fails here, not in a kernel's spin.
    fn probe(g: *Gather) !void {
        const c = g.ctrl();
        var token: u32 = @as(u32, g.rank) + 1;
        for (0..g.world) |r| if (r != g.rank) {
            const p = peerIndex(g.rank, r);
            const me = peerIndex(r, g.rank);
            for (0..g.n_hca) |h| {
                const x = &g.hcas[h];
                var sge: Sge = .{ .addr = @intFromPtr(&token), .length = 4, .lkey = 0 };
                var wr: SendWr = .{ .wr_id = r, .sg_list = &sge, .num_sge = 1, .opcode = wr_rdma_write, .send_flags = send_signaled | send_inline, .remote_addr = g.peer_addr[p] + g.layout.ctrl + 4 * (c_probe + me * max_hcas + h), .rkey = g.peer_rkey[p][h] };
                var bad: ?*SendWr = null;
                const rc = opsCall(PostSend, x.ctx.?, ops_post_send)(x.qp[p].?, &wr, &bad);
                if (rc != 0) {
                    g.fail("ibv_post_send (probe to rank {d}): {d}", .{ r, rc });
                    return error.ProxyFailed;
                }
                x.outstanding += 1;
            }
        };
        var naps: u32 = 0;
        while (true) {
            var pending: u32 = 0;
            for (0..g.n_hca) |h| {
                g.drain(h) catch return error.PeerUnreachable; // g.err names the HCA and the write's rank (wr_id)
                pending += g.hcas[h].outstanding;
            }
            var missing: ?usize = null;
            var missing_hca: usize = 0;
            for (0..g.world) |r| if (r != g.rank) {
                const p = peerIndex(g.rank, r);
                for (0..g.n_hca) |h| if (@atomicLoad(u32, &c[c_probe + p * max_hcas + h], .acquire) != r + 1) {
                    missing = r;
                    missing_hca = h;
                };
            };
            if (pending == 0 and missing == null) return;
            if (naps >= probe_naps) {
                if (missing) |r| g.fail("rank {d}: no token from rank {d} on HCA {d} after about 5 s (no RDMA path between them?)", .{ g.rank, r, missing_hca }) else g.fail("rank {d}: {d} probe write(s) never completed", .{ g.rank, pending });
                return error.PeerUnreachable;
            }
            nap.pause();
            naps += 1;
        }
    }

    fn startProxy(g: *Gather) !void {
        g.running.store(true, .release);
        g.thread = try std.Thread.spawn(.{}, proxy, .{g});
    }

    fn ctrl(g: *Gather) [*]u32 {
        return @ptrCast(@alignCast(g.region.bytes.ptr + g.layout.ctrl));
    }

    fn drain(g: *Gather, h: usize) !void {
        const x = &g.hcas[h];
        var wc: [32]Wc = undefined;
        const n = opsCall(PollCq, x.ctx.?, ops_poll_cq)(x.cq.?, wc.len, &wc);
        if (n < 0) return error.ProxyFailed;
        for (wc[0..@intCast(n)]) |w| {
            if (w.status != 0) {
                const why = g.api.ibv_wc_status_str(w.status) orelse "?";
                g.fail("RDMA write on HCA {d} to rank {d} failed: {s} (vendor 0x{x}, seq {d})", .{ h, w.wr_id, std.mem.span(why), w.vendor_err, g.last_seq });
                return error.ProxyFailed;
            }
            x.outstanding -= 1;
        }
    }

    /// One sequence's shard to every peer: per HCA a stripe, then its flag (inline, signalled) on the same queue pair,
    /// so a flag never lands before its stripe; each peer's copy goes on this rank's queue pair toward it.
    fn post(g: *Gather, seq: u32, nbytes: u32) !void {
        if (nbytes == 0 or nbytes % 16 != 0 or nbytes > g.layout.slot_bytes) {
            g.fail("bad shard size {d} at seq {d}", .{ nbytes, seq });
            return error.ProxyFailed;
        }
        const slot: usize = seq & 1;
        const peers: u32 = @as(u32, g.world) - 1;
        const base = @intFromPtr(g.region.bytes.ptr);
        const packs = nbytes / 16;
        var seq_copy: u32 = seq;
        var offset: u64 = 0;
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            while (x.outstanding + peers > send_depth / 4) {
                try g.drain(h);
                if (!g.running.load(.monotonic)) return error.ProxyFailed;
            }
            var n = packs / g.n_hca;
            if (h < packs % g.n_hca) n += 1;
            const bytes: u32 = n * 16;
            for (0..g.world) |r| if (r != g.rank) {
                const p = peerIndex(g.rank, r);
                const me = peerIndex(r, g.rank);
                var flag_sge: Sge = .{ .addr = @intFromPtr(&seq_copy), .length = 4, .lkey = 0 };
                var flag: SendWr = .{ .wr_id = r, .sg_list = &flag_sge, .num_sge = 1, .opcode = wr_rdma_write, .send_flags = send_signaled | send_inline, .remote_addr = g.peer_addr[p] + g.layout.flagAt(slot, me, h, g.n_hca), .rkey = g.peer_rkey[p][h] };
                var data_sge: Sge = .{ .addr = base + g.layout.send + slot * g.layout.slot_bytes + offset, .length = bytes, .lkey = x.mr.?.lkey };
                var data: SendWr = .{ .wr_id = r, .next = &flag, .sg_list = &data_sge, .num_sge = 1, .opcode = wr_rdma_write, .remote_addr = g.peer_addr[p] + g.layout.recvAt(slot, me) + offset, .rkey = g.peer_rkey[p][h] };
                var bad: ?*SendWr = null;
                const first: *SendWr = if (bytes != 0) &data else &flag;
                const rc = opsCall(PostSend, x.ctx.?, ops_post_send)(x.qp[p].?, first, &bad);
                if (rc != 0) {
                    g.fail("ibv_post_send (to rank {d}): {d}", .{ r, rc });
                    return error.ProxyFailed;
                }
                x.outstanding += 1;
            };
            offset += bytes;
        }
        for (0..g.n_hca) |h| try g.drain(h);
    }

    /// The proxy: spins on the doorbell and posts each sequence in order (a missed one from its slot's size).
    fn proxy(g: *Gather) void {
        const c = g.ctrl();
        var idle: u64 = 0;
        while (g.running.load(.monotonic)) {
            const seq = @atomicLoad(u32, &c[c_seq], .acquire);
            if (seq == g.last_seq) {
                idle += 1;
                if (idle % 64 == 0) for (0..g.n_hca) |h| g.drain(h) catch {
                    g.failed.store(true, .release);
                    return;
                };
                if (idle >= idle_spins) nap.pause();
                continue;
            }
            idle = 0;
            // a rank's next gather can ring before this thread saw the last one; peers stay within a slot's distance
            var pending = seq -% g.last_seq;
            if (pending > slots) {
                g.fail("the doorbell skipped {d} sequences (last {d}, now {d})", .{ pending, g.last_seq, seq });
                g.failed.store(true, .release);
                return;
            }
            var s = g.last_seq +% 1;
            while (pending > 0) : ({
                s +%= 1;
                pending -= 1;
            }) {
                const nbytes = @atomicLoad(u32, &c[c_nbytes + (s & 1)], .acquire);
                g.post(s, nbytes) catch {
                    g.failed.store(true, .release);
                    return;
                };
                g.last_seq = s;
            }
        }
    }

    /// The cap every rank tests in ``fits``: the smallest of theirs, which each rank's slot holds.
    fn agreedCap(caps: []const u64) usize {
        var cap = caps[0];
        for (caps[1..]) |c| cap = @min(cap, c);
        return @intCast(cap);
    }

    /// Whether a gather of ``bytes`` a rank goes this way: the size alone decides, so every rank picks the same path.
    pub fn fits(g: *const Gather, bytes: usize) bool {
        return bytes > 0 and bytes <= g.max_bytes and bytes % 16 == 0;
    }

    /// Every rank's ``bytes`` from ``send`` into ``recv`` in rank order (as ncclAllGather); misaligned is an error.
    pub fn gather(g: *Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) !void {
        if (send % 16 != 0 or recv % 16 != 0) return error.Misaligned;
        const packs = bytes / 16;
        const need = @max(1, std.math.divCeil(usize, packs, 2 * threads) catch unreachable);
        const grid: u32 = @intCast(@min(std.math.ceilPowerOfTwo(usize, need) catch max_blocks, max_blocks));
        const klass: i32 = @intCast(std.math.log2_int(u32, grid));
        const base = g.region_dev;
        var a: launch.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(i32, @intCast(packs)));
        a.add(@as(i32, g.rank));
        a.add(base + g.layout.recv);
        a.add(base + g.layout.flags);
        a.add(base + g.layout.send);
        a.add(base + g.layout.ctrl);
        a.add(@as(i64, @intCast(g.layout.slot_bytes)));
        a.add(g.counters.ptr);
        a.add(klass);
        a.add(spin_limit);
        a.add(@as(i32, @intCast(g.n_hca)));
        a.add(@as(i32, g.world));
        try launch.launch(g.f, .{ .grid = .{ .x = grid }, .block = .{ .x = threads } }, .{ .handle = stream, .d = g.d }, &a);
        g.gathers += 1;
    }

    /// An error if a wait timed out on the GPU or the proxy failed (the host checks after a synchronize).
    pub fn check(g: *Gather) !void {
        const c = g.ctrl();
        const bad = @atomicLoad(u32, &c[c_error], .acquire);
        if (bad == 0 and !g.failed.load(.acquire)) return;
        std.log.err("RoCE all-gather failed (sequence {d}, HCA {d}, rank {d}): {s}", .{ bad, c[c_hca], c[c_peer], std.mem.sliceTo(&g.err, 0) });
        return error.ProxyFailed;
    }

    fn teardown(g: *Gather) void {
        if (g.running.swap(false, .acq_rel)) if (g.thread) |t| t.join();
        g.closeHcas();
        g.module.unload();
        g.counters.free();
        g.region.free();
        g.lib.close();
    }

    pub fn deinit(g: *Gather) void {
        g.teardown();
        g.gpa.destroy(g);
    }
};

/// The HCAs to stripe over, at most two: TF_ROCE_HCA, else NCCL_IB_HCA without "=^" prefixes or ":port" suffixes.
fn hcaNames(io: std.Io, gpa: std.mem.Allocator, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) !u32 {
    _ = io;
    _ = gpa;
    const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse return error.NoHca;
    var n: u32 = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |w| {
        if (n == max_hcas) break;
        var s = std.mem.trimStart(u8, std.mem.trim(u8, w, " "), "=^");
        if (std.mem.indexOfScalar(u8, s, ':')) |c| s = s[0..c];
        if (s.len == 0 or s.len > 63) continue;
        @memcpy(names[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
    }
    if (n == 0) return error.NoHca;
    return n;
}

/// The HCA's RoCE v2 IPv4 GID (TF_ROCE_GID_INDEX overrides): its index differs between hosts' tables.
fn gidIndex(io: std.Io, gpa: std.mem.Allocator, name: []const u8) !c_int {
    if (std.c.getenv("TF_ROCE_GID_INDEX")) |v| return std.fmt.parseInt(c_int, std.mem.span(v), 10) catch error.NoHca;
    const cwd = std.Io.Dir.cwd();
    var i: c_int = 0;
    while (i < 16) : (i += 1) {
        var pbuf: [160]u8 = undefined;
        const tpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, i });
        const t = cwd.readFileAlloc(io, tpath, gpa, .limited(64)) catch continue;
        defer gpa.free(t);
        if (!std.mem.startsWith(u8, t, "RoCE v2")) continue;
        const gpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ name, i });
        const gid = cwd.readFileAlloc(io, gpath, gpa, .limited(64)) catch continue;
        defer gpa.free(gid);
        if (std.mem.startsWith(u8, gid, "0000:0000:0000:0000:0000:ffff:")) return i;
    }
    return error.NoHca;
}

test "RoCE gather: every rank agrees one cap" {
    // 511 KiB and 512 KiB round to the same slot; a 512 KiB shard must take the same path on every rank
    try std.testing.expectEqual(Layout.of(511 << 10, 2).slot_bytes, Layout.of(512 << 10, 2).slot_bytes);
    const a = Gather.agreedCap(&.{ 511 << 10, 512 << 10 });
    const b = Gather.agreedCap(&.{ 512 << 10, 511 << 10 });
    try std.testing.expectEqual(a, b);
    try std.testing.expectEqual(@as(usize, 511 << 10), a);
    // four ranks: the smallest, wherever it sits
    try std.testing.expectEqual(@as(usize, 64 << 10), Gather.agreedCap(&.{ 4096 << 10, 512 << 10, 64 << 10, 4096 << 10 }));
}

test "RoCE gather: two ranks keep the two-rank region, byte for byte" {
    for ([_]usize{ 16, 4096, 64 << 10, 512 << 10, 4096 << 10 }) |max_bytes| {
        const l = Layout.of(max_bytes, 2);
        // the layout before any world: recv[2][s], flags[2][max_hcas], send[2][s], ctrl
        const s = std.mem.alignForward(usize, max_bytes, 4096);
        try std.testing.expectEqual(@as(usize, 0), l.recv);
        try std.testing.expectEqual(slots * s, l.flags);
        try std.testing.expectEqual(slots * s + slots * max_hcas * flag_stride, l.send);
        try std.testing.expectEqual(l.send + slots * s, l.ctrl);
        try std.testing.expectEqual(l.ctrl + flag_stride, l.total);
        for (0..slots) |slot| {
            try std.testing.expectEqual(slot * s, l.recvAt(slot, 0));
            for (0..max_hcas) |h| try std.testing.expectEqual(l.flags + (slot * 2 + h) * flag_stride, l.flagAt(slot, 0, h, 2));
        }
    }
}

test "RoCE gather: four ranks' slots and flags never overlap and sit in the sender's rank order" {
    const l = Layout.of(16 << 10, 4);
    try std.testing.expectEqual(@as(usize, 3), l.peers);
    try std.testing.expectEqual(slots * 3 * l.slot_bytes, l.flags);
    try std.testing.expect(l.flags + slots * 3 * max_hcas * flag_stride == l.send);
    for (0..4) |rank| {
        for (0..4) |r| if (r != rank) {
            const p = peerIndex(rank, r);
            try std.testing.expect(p < 3);
            try std.testing.expectEqual(r, senderOf(rank, p));
            // what rank r writes here lands where this rank reads peer p: the same number on both sides
            try std.testing.expectEqual(p, peerIndex(rank, senderOf(rank, p)));
        };
        // the three peers of a rank are numbered 0, 1, 2 without a gap
        var seen: [3]bool = @splat(false);
        for (0..4) |r| if (r != rank) {
            seen[peerIndex(rank, r)] = true;
        };
        for (seen) |x| try std.testing.expect(x);
    }
    for (0..slots) |slot| for (0..3) |p| {
        const at = l.recvAt(slot, p);
        try std.testing.expect(at + l.slot_bytes <= l.flags);
        for (0..3) |q| if (q != p) try std.testing.expect(l.recvAt(slot, q) >= at + l.slot_bytes or l.recvAt(slot, q) + l.slot_bytes <= at);
        for (0..2) |h| {
            const f = l.flagAt(slot, p, h, 2);
            try std.testing.expect(f >= l.flags and f + flag_stride <= l.send);
        }
    };
    try std.testing.expect(@sizeOf(Blob) % 8 == 0);
}

test "RoCE gather: a mesh reaches every rank, a ring of four leaves the far rank out, two on a cable are neighbours" {
    const Fabric = struct {
        fn ring(world: usize, rank: usize, peer: usize) bool {
            return (peer + 1) % world == rank or (rank + 1) % world == peer;
        }
        fn mesh(_: usize, _: usize, _: usize) bool {
            return true;
        }
    };
    for (0..4) |r| try std.testing.expectEqual(@as(?usize, null), firstUnreachable(r, 4, @as(usize, 4), &Fabric.mesh));
    try std.testing.expectEqual(@as(?usize, 2), firstUnreachable(0, 4, @as(usize, 4), &Fabric.ring));
    try std.testing.expectEqual(@as(?usize, 3), firstUnreachable(1, 4, @as(usize, 4), &Fabric.ring));
    try std.testing.expectEqual(@as(?usize, 0), firstUnreachable(2, 4, @as(usize, 4), &Fabric.ring));
    for (0..2) |r| try std.testing.expectEqual(@as(?usize, null), firstUnreachable(r, 2, @as(usize, 2), &Fabric.ring));
}
