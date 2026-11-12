//! libibverbs for the RoCE all-gather (rdma-core's verbs.h ABI, offsets checked) and the HCA and GID lookups.
const std = @import("std");

pub const max_hcas = 2;

// -- libibverbs (the ABI of rdma-core's verbs.h; offsets checked against its headers) ---------------------------------
pub const Device = opaque {};
pub const Context = opaque {};
pub const Pd = opaque {};
pub const Cq = extern struct { context: *Context };
pub const Mr = extern struct { context: *Context, pd: *Pd, addr: ?*anyopaque, length: usize, handle: u32, lkey: u32, rkey: u32 };
pub const Qp = extern struct { context: *Context, qp_context: ?*anyopaque, pd: *Pd, send_cq: *Cq, recv_cq: *Cq, srq: ?*anyopaque, handle: u32, qp_num: u32 };
pub const QpCap = extern struct { max_send_wr: u32 = 0, max_recv_wr: u32 = 0, max_send_sge: u32 = 0, max_recv_sge: u32 = 0, max_inline_data: u32 = 0 };
pub const QpInitAttr = extern struct { qp_context: ?*anyopaque = null, send_cq: ?*Cq = null, recv_cq: ?*Cq = null, srq: ?*anyopaque = null, cap: QpCap = .{}, qp_type: c_int = 0, sq_sig_all: c_int = 0 };
pub const GlobalRoute = extern struct { dgid: [16]u8 = @splat(0), flow_label: u32 = 0, sgid_index: u8 = 0, hop_limit: u8 = 0, traffic_class: u8 = 0 };
pub const AhAttr = extern struct { grh: GlobalRoute = .{}, dlid: u16 = 0, sl: u8 = 0, src_path_bits: u8 = 0, static_rate: u8 = 0, is_global: u8 = 0, port_num: u8 = 0 };
pub const QpAttr = extern struct {
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
pub const Sge = extern struct { addr: u64, length: u32, lkey: u32 };
pub const SendWr = extern struct {
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
pub const Wc = extern struct { wr_id: u64, status: u32, opcode: u32, vendor_err: u32, _rest: [48 - 20]u8 };
pub const PortAttr = extern struct { bytes: [56]u8 align(8) };

comptime {
    std.debug.assert(@sizeOf(QpInitAttr) == 64 and @offsetOf(QpInitAttr, "cap") == 32 and @offsetOf(QpInitAttr, "qp_type") == 52);
    std.debug.assert(@sizeOf(AhAttr) == 32 and @offsetOf(AhAttr, "dlid") == 24 and @offsetOf(AhAttr, "port_num") == 30);
    std.debug.assert(@sizeOf(QpAttr) == 144 and @offsetOf(QpAttr, "ah_attr") == 56 and @offsetOf(QpAttr, "pkey_index") == 120);
    std.debug.assert(@offsetOf(QpAttr, "rnr_retry") == 132 and @offsetOf(QpAttr, "qp_access_flags") == 32);
    std.debug.assert(@sizeOf(SendWr) == 128 and @offsetOf(SendWr, "remote_addr") == 40 and @offsetOf(SendWr, "rkey") == 48);
    std.debug.assert(@sizeOf(Wc) == 48 and @offsetOf(Mr, "lkey") == 36 and @offsetOf(Qp, "qp_num") == 52);
}

// struct ibv_context's ops: the two calls verbs.h inlines
pub const ops_poll_cq = 96;
pub const ops_post_send = 208;
pub const PollCq = *const fn (*Cq, c_int, [*]Wc) callconv(.c) c_int;
pub const PostSend = *const fn (*Qp, *SendWr, *?*SendWr) callconv(.c) c_int;

pub const wr_rdma_write = 0;
pub const send_signaled = 2;
pub const send_inline = 8;
pub const qpt_rc = 2;
pub const qps_init = 1;
pub const qps_rtr = 2;
pub const qps_rts = 3;
pub const port_active = 4;
pub const access_local_write = 1;
pub const access_remote_write = 2;
pub const qp_state = 1;
pub const qp_access_flags = 8;
pub const qp_pkey_index = 16;
pub const qp_port = 32;
pub const qp_av = 128;
pub const qp_path_mtu = 256;
pub const qp_timeout = 512;
pub const qp_retry_cnt = 1024;
pub const qp_rnr_retry = 2048;
pub const qp_rq_psn = 4096;
pub const qp_max_qp_rd_atomic = 8192;
pub const qp_min_rnr_timer = 32768;
pub const qp_sq_psn = 65536;
pub const qp_max_dest_rd_atomic = 131072;
pub const qp_dest_qpn = 1048576;

pub const Api = struct {
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

pub fn opsCall(comptime T: type, ctx: *Context, offset: usize) T {
    const slot: *const T = @ptrFromInt(@intFromPtr(ctx) + offset);
    return slot.*;
}

/// The HCAs to stripe over, at most two: TF_ROCE_HCA, else NCCL_IB_HCA without "=^" prefixes or ":port" suffixes.
pub fn hcaNames(given: ?[*:0]const u8, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) !u32 {
    const raw = given orelse return error.NoHca;
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
pub fn gidIndex(io: std.Io, gpa: std.mem.Allocator, name: []const u8, set: bool) !c_int {
    if (set) return @import("sys").environment.read(c_int, "TF_ROCE_GID_INDEX", 0);
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
