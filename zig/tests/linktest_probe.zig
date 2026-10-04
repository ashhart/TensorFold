//! What a verbs device reports and what it actually allows: registration sizes and counts, queue pair types and counts.
const std = @import("std");
const fabric = @import("fabric");
const abi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const out = @import("linktest_suite.zig").out;

const mib = 1 << 20;

fn errno() c_int {
    return std.c._errno().*;
}

pub fn device(v: *const Verbs, dev: []const u8) !void {
    const ctx = try v.openDevice(dev);
    defer _ = v.api.ibv_close_device(ctx);
    const d = try v.device(ctx);
    out("R device dev={s} max_mr_size={d} max_mr={d} max_qp={d} max_cq={d} max_pd={d} max_qp_wr={d} max_cqe={d} max_sge={d} max_qp_rd_atom={d} atomic_cap={d} page_size_cap={d}", .{ dev, d.max_mr_size, d.max_mr, d.max_qp, d.max_cq, d.max_pd, d.max_qp_wr, d.max_cqe, d.max_sge, d.max_qp_rd_atom, d.atomic_cap, d.page_size_cap });
    const p = try v.port(ctx, 1);
    out("R port dev={s} state={d} max_mtu={d} active_mtu={d} gid_tbl_len={d} max_msg_sz={d} lid={d} link_layer={d} width={d} speed={d}", .{ dev, p.state, p.max_mtu, p.active_mtu, p.gid_tbl_len, p.max_msg_sz, p.lid, p.link_layer, p.active_width, p.active_speed });
    for (0..8) |i| {
        const g = v.gid(ctx, 1, @intCast(i)) catch continue;
        if (std.mem.allEqual(u8, &g.raw, 0)) continue;
        out("R gid dev={s} index={d} gid={s} type={?d}", .{ dev, i, &std.fmt.bytesToHex(g.raw, .lower), v.gidType(ctx, 1, @intCast(i)) });
    }
    const pd = v.api.ibv_alloc_pd(ctx) orelse {
        out("R pd dev={s} ok=0 errno={d}", .{ dev, errno() });
        return;
    };
    defer _ = v.api.ibv_dealloc_pd(pd);
    const big = try std.posix.mmap(null, 1024 * mib, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(big);
    const access = abi.access_local_write | abi.access_remote_write;
    for ([_]usize{ 16384, mib, @intCast(d.max_mr_size), 16 * mib, 64 * mib, 256 * mib, 512 * mib, 1024 * mib }) |size| {
        const mr = v.api.ibv_reg_mr(pd, big.ptr, size, access);
        out("R mr_size dev={s} bytes={d} ok={d} errno={d}", .{ dev, size, @intFromBool(mr != null), if (mr == null) errno() else 0 });
        if (mr) |m| _ = v.api.ibv_dereg_mr(m);
    }
    var mrs: [1024]*abi.Mr = undefined;
    var n: usize = 0;
    var failed: c_int = 0;
    while (n < mrs.len) : (n += 1) {
        mrs[n] = v.api.ibv_reg_mr(pd, big.ptr + n * mib, mib, access) orelse {
            failed = errno();
            break;
        };
    }
    out("R mr_count dev={s} bytes_each={d} count={d} tried={d} failed_errno={d}", .{ dev, mib, n, mrs.len, failed });
    for (mrs[0..n]) |m| _ = v.api.ibv_dereg_mr(m);
    for ([_]c_int{ abi.qpt_rc, abi.qpt_uc }) |kind| {
        const cq = v.api.ibv_create_cq(ctx, 1024, null, null, 0) orelse return error.VerbsFailed;
        defer _ = v.api.ibv_destroy_cq(cq);
        var qps: [8]*abi.Qp = undefined;
        var k: usize = 0;
        var why: c_int = 0;
        while (k < qps.len) : (k += 1) {
            var init: abi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = 1024, .max_recv_wr = 1024, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = kind };
            qps[k] = v.api.ibv_create_qp(pd, &init) orelse {
                why = errno();
                break;
            };
        }
        out("R qp_count dev={s} type={s} created={d} tried={d} failed_errno={d}", .{ dev, if (kind == abi.qpt_rc) "rc" else "uc", k, qps.len, why });
        for (qps[0..k]) |q| _ = v.api.ibv_destroy_qp(q);
    }
}
