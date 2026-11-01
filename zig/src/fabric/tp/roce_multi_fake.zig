//! In-memory switched or ring fabric with distinct peer QPs and a shared CQ that completes peers out of order.
const std = @import("std");
const abi = @import("verbs").abi;
const rv = @import("roce_multi_verbs.zig");
const guard = @import("roce_multi_guard.zig");
const cap = 256;
const Entry = struct { id: u64, bytes: u32, order: u64 };
pub const Fabric = struct {
    world: u32,
    nh: u32,
    layout: guard.Layout,
    regions: [8][]align(std.heap.page_size_min) u8 = undefined,
    wires: [8]rv.VerbsWire = undefined,
    ctx: [8][2]abi.Context = undefined,
    cq: [8][2]abi.Cq = undefined,
    qp: [8][2][8]abi.Qp = undefined,
    mr: [8][2]abi.Mr = undefined,
    queue: [8][2][cap]?Entry = @splat(@splat(@splat(null))),
    flight: u64 = 0,
    peak: u64 = 0,
    order: u64 = 0,
    last_bytes: u64 = 0,
    wrong_offset: bool = false,
    bad_completion: bool = false,
    bad: usize = 0,
    peer_order: [8]u32 = undefined,
    peers_seen: usize = 0,
    /// lands[r][h][p]: the HCA of rank p that rank r's HCA h reaches (null: no cable to p); a switch reaches HCA h.
    lands: [8][2][8]?u32 = @splat(@splat(@splat(null))),
    /// dest[r][h][p]: the QP number that rank r's HCA h's QP to p was brought to RTR against (null: never).
    dest: [8][2][8]?u32 = @splat(@splat(@splat(null))),
    pub var active: *Fabric = undefined;

    pub fn init(f: *Fabric, world: u32, nh: u32, slot: usize) !void {
        f.* = .{ .world = world, .nh = nh, .layout = guard.Layout.of(world, slot) };
        active = f;
        for (0..world) |r| {
            f.regions[r] = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), f.layout.len);
            @memset(f.regions[r], 0);
        }
        for (0..world) |r| {
            f.wires[r] = .{ .v = undefined, .traffic_class = 0, .me = @intCast(r), .layout = f.layout, .region = f.regions[r], .n_hca = nh, .stripes = nh };
            f.wires[r].v.api.ibv_modify_qp = modify;
            for (0..nh) |h| {
                f.ctx[r][h].ops = .{ ._before_poll = @splat(null), .poll_cq = poll, ._before_post = @splat(null), .post_send = post, .post_recv = undefined, ._after = @splat(null) };
                f.cq[r][h].handle = @intCast(2 * r + h);
                f.mr[r][h].lkey = @intCast(100 + 2 * r + h);
                f.mr[r][h].rkey = @intCast(200 + 2 * r + h);
                f.wires[r].hcas[h] = .{ .ctx = &f.ctx[r][h], .pd = undefined, .mr = &f.mr[r][h], .cq = &f.cq[r][h], .gid_index = 0, .gid = @splat(0), .mtu = 0, .lid = 0 };
                for (0..world) |p| if (p != r) {
                    f.qp[r][h][p].qp_num = @intCast(r * 16 + h * 8 + p);
                    f.wires[r].hcas[h].qp[p] = &f.qp[r][h][p];
                    f.wires[r].peer_rkey[p][h] = @intCast(200 + 2 * p + h);
                    f.lands[r][h][p] = @intCast(h);
                };
            }
            for (0..world) |p| f.wires[r].peer_region[p] = @intFromPtr(f.regions[p].ptr);
        }
    }
    /// Rewires two HCAs a rank as a switchless ring: rank r's HCA 0 to rank r + 1's HCA 1, a /24 a link; unconnected.
    pub fn ring(f: *Fabric) void {
        std.debug.assert(f.nh == 2);
        f.lands = @splat(@splat(@splat(null)));
        for (0..f.world) |r| {
            const next = (r + 1) % f.world;
            f.lands[r][0][next] = 1;
            f.lands[next][1][r] = 0;
            f.wires[r].hcas[0].gid = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 41, @intCast(r), 1 };
            f.wires[next].hcas[1].gid = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 10, 41, @intCast(r), 2 };
            f.wires[r].stripes = 0;
            f.wires[r].peer_rkey = @splat(@splat(0));
        }
    }
    pub fn deinit(f: *Fabric) void {
        for (f.regions[0..f.world]) |r| std.testing.allocator.free(r);
    }
    /// `connect`'s verb: RTR must name the queue pair and GID of the HCA the cable lands on.
    fn modify(qp: *abi.Qp, a: *abi.QpAttr, _: c_int) callconv(.c) c_int {
        const f = active;
        const src = qp.qp_num / 16;
        const h = qp.qp_num / 8 % 2;
        const dst = qp.qp_num % 8;
        if (a.qp_state != abi.qps_rtr) return 0;
        const j = f.lands[src][h][dst] orelse {
            f.bad += 1;
            return 1;
        };
        if (!std.mem.eql(u8, &a.ah_attr.grh.dgid.raw, &f.wires[dst].hcas[j].gid)) {
            f.bad += 1;
            return 1;
        }
        f.dest[src][h][dst] = a.dest_qp_num;
        return 0;
    }
    fn post(qp: *abi.Qp, first: *abi.SendWr, _: *?*abi.SendWr) callconv(.c) c_int {
        const f = active;
        const src = qp.qp_num / 16;
        const h = qp.qp_num / 8 % 2;
        const dst = qp.qp_num % 8;
        const j = f.lands[src][h][dst] orelse {
            f.bad += 1; // this HCA has no cable to that rank
            return 1;
        };
        if (h == 0) {
            f.peer_order[f.peers_seen % 8] = dst;
            f.peers_seen += 1;
        }
        var it: ?*abi.SendWr = first;
        var bytes: u32 = 0;
        while (it) |wr| : (it = wr.next) {
            const sg = wr.sg_list[0];
            const lo = @intFromPtr(f.regions[dst].ptr);
            var remote = wr.remote_addr;
            const flag = wr.send_flags & abi.send_inline != 0;
            if (!flag and f.wrong_offset) remote += 16;
            if (remote < lo or sg.length > f.layout.len or remote - lo > f.layout.len - sg.length or wr.rkey != 200 + 2 * dst + j) {
                f.bad += 1;
                return 1;
            }
            if (flag) {
                if (sg.length != 4 or wr.next != null or wr.send_flags != abi.send_inline | abi.send_signaled) {
                    f.bad += 1;
                    return 1;
                }
                const target: *std.atomic.Value(u32) = @ptrFromInt(remote);
                const value: *const u32 = @ptrFromInt(sg.addr);
                target.store(value.*, .release);
            } else {
                const from = @intFromPtr(f.regions[src].ptr);
                if (sg.addr < from + f.layout.send or sg.addr + sg.length > from + f.layout.recv or sg.lkey != 100 + 2 * src + h) {
                    f.bad += 1;
                    return 1;
                }
                const target: [*]u8 = @ptrFromInt(remote);
                const source: [*]const u8 = @ptrFromInt(sg.addr);
                @memcpy(target[0..sg.length], source[0..sg.length]);
                bytes += sg.length;
            }
            if (wr.send_flags & abi.send_signaled != 0) {
                for (&f.queue[src][h]) |*q| if (q.* == null) {
                    q.* = .{ .id = wr.wr_id, .bytes = bytes, .order = f.order };
                    f.order += 1;
                    f.flight += bytes;
                    f.peak = @max(f.peak, f.flight);
                    return 0;
                };
                return 1;
            }
        }
        return 1;
    }
    fn poll(cq: *abi.Cq, _: c_int, wc: [*]abi.Wc) callconv(.c) c_int {
        const f = active;
        const src = cq.handle / 2;
        const h = cq.handle % 2;
        var at: ?usize = null;
        for (f.queue[src][h], 0..) |q, i| if (q) |e| {
            if (at == null or e.id >> 32 > f.queue[src][h][at.?].?.id >> 32 or
                (e.id >> 32 == f.queue[src][h][at.?].?.id >> 32 and e.order < f.queue[src][h][at.?].?.order)) at = i;
        };
        const i = at orelse return 0;
        const e = f.queue[src][h][i].?;
        f.flight -= e.bytes;
        f.last_bytes = e.bytes;
        f.queue[src][h][i] = null;
        wc[0] = std.mem.zeroes(abi.Wc);
        wc[0].wr_id = if (f.bad_completion) e.id ^ 1 else e.id;
        wc[0].status = abi.wc_success;
        return 1;
    }
    pub fn drain(f: *Fabric, rank: usize) !void {
        while (!f.wires[rank].idle()) try f.wires[rank].reap();
    }
};
