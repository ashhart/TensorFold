//! The verbs ABI (rdma-core's infiniband/verbs.h, which Apple's SDK ships for librdma): the structs and constants we use.
const std = @import("std");

pub const Device = extern struct {
    _ops: [2]?*anyopaque,
    node_type: c_int,
    transport_type: c_int,
    name: [64]u8,
    dev_name: [64]u8,
    dev_path: [256]u8,
    ibdev_path: [256]u8,
};

pub const Qp = extern struct {
    context: *Context,
    qp_context: ?*anyopaque,
    pd: *Pd,
    send_cq: *Cq,
    recv_cq: *Cq,
    srq: ?*anyopaque,
    handle: u32,
    qp_num: u32,
    state: c_int,
    qp_type: c_int,
    mutex: std.c.pthread_mutex_t,
    cond: std.c.pthread_cond_t,
    events_completed: u32,
};

pub const Cq = extern struct {
    context: *Context,
    channel: ?*anyopaque,
    cq_context: ?*anyopaque,
    handle: u32,
    cqe: c_int,
    mutex: std.c.pthread_mutex_t,
    cond: std.c.pthread_cond_t,
    comp_events_completed: u32,
    async_events_completed: u32,
};

pub const PostSend = *const fn (qp: *Qp, wr: *SendWr, bad: *?*SendWr) callconv(.c) c_int;
pub const PostRecv = *const fn (qp: *Qp, wr: *RecvWr, bad: *?*RecvWr) callconv(.c) c_int;
pub const PollCq = *const fn (cq: *Cq, n: c_int, wc: [*]Wc) callconv(.c) c_int;

/// The provider's data-path entry points; only these three are called through the table.
pub const ContextOps = extern struct {
    _before_poll: [11]?*anyopaque,
    poll_cq: PollCq,
    _before_post: [13]?*anyopaque,
    post_send: PostSend,
    post_recv: PostRecv,
    _after: [5]?*anyopaque,
};

pub const Context = extern struct {
    device: *Device,
    ops: ContextOps,
    cmd_fd: c_int,
    async_fd: c_int,
    num_comp_vectors: c_int,
    mutex: std.c.pthread_mutex_t,
    abi_compat: ?*anyopaque,
};

pub const Pd = extern struct { context: *Context, handle: u32 };

pub const Mr = extern struct {
    context: *Context,
    pd: *Pd,
    addr: ?*anyopaque,
    length: usize,
    handle: u32,
    lkey: u32,
    rkey: u32,
};

pub const Gid = extern union { raw: [16]u8, global: extern struct { subnet_prefix: u64, interface_id: u64 } };

pub const PortAttr = extern struct {
    state: c_int,
    max_mtu: c_int,
    active_mtu: c_int,
    gid_tbl_len: c_int,
    port_cap_flags: u32,
    max_msg_sz: u32,
    bad_pkey_cntr: u32,
    qkey_viol_cntr: u32,
    pkey_tbl_len: u16,
    lid: u16,
    sm_lid: u16,
    lmc: u8,
    max_vl_num: u8,
    sm_sl: u8,
    subnet_timeout: u8,
    init_type_reply: u8,
    active_width: u8,
    active_speed: u8,
    phys_state: u8,
    link_layer: u8,
    flags: u8,
    port_cap_flags2: u16,
};

pub const DeviceAttr = extern struct {
    fw_ver: [64]u8,
    node_guid: u64,
    sys_image_guid: u64,
    max_mr_size: u64,
    page_size_cap: u64,
    vendor_id: u32,
    vendor_part_id: u32,
    hw_ver: u32,
    max_qp: c_int,
    max_qp_wr: c_int,
    device_cap_flags: c_uint,
    max_sge: c_int,
    max_sge_rd: c_int,
    max_cq: c_int,
    max_cqe: c_int,
    max_mr: c_int,
    max_pd: c_int,
    max_qp_rd_atom: c_int,
    max_ee_rd_atom: c_int,
    max_res_rd_atom: c_int,
    max_qp_init_rd_atom: c_int,
    max_ee_init_rd_atom: c_int,
    atomic_cap: c_int,
    _counts: [12]c_int,
    max_srq_wr: c_int,
    max_srq_sge: c_int,
    max_pkeys: u16,
    local_ca_ack_delay: u8,
    phys_port_cnt: u8,
};

pub const QpCap = extern struct { max_send_wr: u32, max_recv_wr: u32, max_send_sge: u32, max_recv_sge: u32, max_inline_data: u32 };

pub const QpInitAttr = extern struct {
    qp_context: ?*anyopaque = null,
    send_cq: *Cq,
    recv_cq: *Cq,
    srq: ?*anyopaque = null,
    cap: QpCap,
    qp_type: c_int,
    sq_sig_all: c_int = 0,
};

pub const GlobalRoute = extern struct { dgid: Gid, flow_label: u32 = 0, sgid_index: u8 = 0, hop_limit: u8 = 0, traffic_class: u8 = 0 };

pub const AhAttr = extern struct {
    grh: GlobalRoute,
    dlid: u16 = 0,
    sl: u8 = 0,
    src_path_bits: u8 = 0,
    static_rate: u8 = 0,
    is_global: u8 = 0,
    port_num: u8 = 0,
};

pub const QpAttr = extern struct {
    qp_state: c_int = 0,
    cur_qp_state: c_int = 0,
    path_mtu: c_int = 0,
    path_mig_state: c_int = 0,
    qkey: u32 = 0,
    rq_psn: u32 = 0,
    sq_psn: u32 = 0,
    dest_qp_num: u32 = 0,
    qp_access_flags: c_uint = 0,
    cap: QpCap = std.mem.zeroes(QpCap),
    ah_attr: AhAttr = std.mem.zeroes(AhAttr),
    alt_ah_attr: AhAttr = std.mem.zeroes(AhAttr),
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
};

pub const Sge = extern struct { addr: u64, length: u32, lkey: u32 };

pub const SendWr = extern struct {
    wr_id: u64 = 0,
    next: ?*SendWr = null,
    sg_list: [*]Sge,
    num_sge: c_int,
    opcode: c_int,
    send_flags: c_uint,
    imm_data: u32 = 0,
    remote_addr: u64 = 0,
    rkey: u32 = 0,
    _wr_rest: [20]u8 = @splat(0),
    _qp_type: u32 = 0,
    _tail: [52]u8 = @splat(0),
};

pub const RecvWr = extern struct { wr_id: u64 = 0, next: ?*RecvWr = null, sg_list: ?[*]Sge = null, num_sge: c_int = 0 };

pub const Wc = extern struct {
    wr_id: u64,
    status: c_int,
    opcode: c_int,
    vendor_err: u32,
    byte_len: u32,
    imm_data: u32,
    qp_num: u32,
    src_qp: u32,
    wc_flags: c_uint,
    pkey_index: u16,
    slid: u16,
    sl: u8,
    dlid_path_bits: u8,
};

pub const GidEntry = extern struct { gid: Gid, gid_index: u32, port_num: u32, gid_type: u32, ndev_ifindex: u32 };

pub const port_active = 4;
pub const link_thunderbolt = 100;
pub const qpt_rc = 2;
pub const qpt_uc = 3;
pub const qps_init = 1;
pub const qps_rtr = 2;
pub const qps_rts = 3;
pub const qps_err = 6;
pub const access_local_write = 1;
pub const access_remote_write = 2;
pub const access_remote_read = 4;
pub const wr_rdma_write = 0;
pub const wr_rdma_write_imm = 1;
pub const wr_send = 2;
pub const send_signaled = 2;
pub const send_inline = 8;
pub const wc_success = 0;
pub const wc_with_imm = 2;

/// ibv_qp_attr_mask bits.
pub const mask = struct {
    pub const state = 1 << 0;
    pub const access_flags = 1 << 3;
    pub const pkey_index = 1 << 4;
    pub const port = 1 << 5;
    pub const av = 1 << 7;
    pub const path_mtu = 1 << 8;
    pub const timeout = 1 << 9;
    pub const retry_cnt = 1 << 10;
    pub const rnr_retry = 1 << 11;
    pub const rq_psn = 1 << 12;
    pub const max_qp_rd_atomic = 1 << 13;
    pub const min_rnr_timer = 1 << 15;
    pub const sq_psn = 1 << 16;
    pub const max_dest_rd_atomic = 1 << 17;
    pub const dest_qpn = 1 << 20;
};

// Sizes and offsets the SDK's verbs.h gives on arm64 macOS, as tools/zig/verbs_layout.c prints them.
test "the ABI matches verbs.h" {
    const t = std.testing;
    try t.expectEqual(664, @sizeOf(Device));
    try t.expectEqual(24, @offsetOf(Device, "name"));
    try t.expectEqual(256, @sizeOf(ContextOps));
    try t.expectEqual(88, @offsetOf(ContextOps, "poll_cq"));
    try t.expectEqual(200, @offsetOf(ContextOps, "post_send"));
    try t.expectEqual(208, @offsetOf(ContextOps, "post_recv"));
    try t.expectEqual(48, @sizeOf(Mr));
    try t.expectEqual(36, @offsetOf(Mr, "lkey"));
    try t.expectEqual(52, @offsetOf(Qp, "qp_num"));
    try t.expectEqual(184, @sizeOf(Qp));
    try t.expectEqual(152, @sizeOf(Cq));
    try t.expectEqual(352, @sizeOf(Context));
    try t.expectEqual(264, @offsetOf(Context, "cmd_fd"));
    try t.expectEqual(52, @sizeOf(PortAttr));
    try t.expectEqual(46, @offsetOf(PortAttr, "link_layer"));
    try t.expectEqual(232, @sizeOf(DeviceAttr));
    try t.expectEqual(80, @offsetOf(DeviceAttr, "max_mr_size"));
    try t.expectEqual(164, @offsetOf(DeviceAttr, "atomic_cap"));
    try t.expectEqual(227, @offsetOf(DeviceAttr, "phys_port_cnt"));
    try t.expectEqual(64, @sizeOf(QpInitAttr));
    try t.expectEqual(52, @offsetOf(QpInitAttr, "qp_type"));
    try t.expectEqual(32, @sizeOf(AhAttr));
    try t.expectEqual(29, @offsetOf(AhAttr, "is_global"));
    try t.expectEqual(144, @sizeOf(QpAttr));
    try t.expectEqual(56, @offsetOf(QpAttr, "ah_attr"));
    try t.expectEqual(129, @offsetOf(QpAttr, "port_num"));
    try t.expectEqual(136, @offsetOf(QpAttr, "rate_limit"));
    try t.expectEqual(128, @sizeOf(SendWr));
    try t.expectEqual(36, @offsetOf(SendWr, "imm_data"));
    try t.expectEqual(40, @offsetOf(SendWr, "remote_addr"));
    try t.expectEqual(48, @offsetOf(SendWr, "rkey"));
    try t.expectEqual(72, @offsetOf(SendWr, "_qp_type"));
    try t.expectEqual(32, @sizeOf(RecvWr));
    try t.expectEqual(48, @sizeOf(Wc));
    try t.expectEqual(36, @offsetOf(Wc, "wc_flags"));
    try t.expectEqual(32, @sizeOf(GidEntry));
}
