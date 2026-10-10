//! Tensor parallelism across processes, one GPU each: bootstrap, collectives, fail-fast; `Session.open` starts a rank.
const std = @import("std");
const log = @import("log.zig");
const cuda = @import("cuda");

pub const collective = @import("collective.zig");
pub const reference = @import("reference.zig");
pub const config = @import("config.zig");
pub const sock = @import("sock.zig");
pub const bootstrap = @import("bootstrap.zig");
pub const fate = @import("fate.zig");
pub const host = @import("host.zig");
pub const nccl = @import("nccl.zig");
pub const mailbox = @import("mailbox.zig");
pub const hybrid = @import("hybrid.zig");
pub const roce = @import("roce.zig");
pub const roce_verbs = @import("roce_verbs.zig");
pub const planlink = @import("planlink.zig");
pub const Verbs = @import("verbs").Verbs;

pub const Collective = collective.Collective;
pub const DType = collective.DType;
pub const Op = collective.Op;
pub const Config = config.Config;
pub const Bootstrap = bootstrap.Bootstrap;
pub const Fate = fate.Fate;

pub const Session = struct {
    cfg: Config,
    lib: cuda.nccl.Library,
    nccl: nccl.Nccl,
    mail: ?mailbox.Mailbox = null,
    roce: ?*roce.Roce = null,
    route: hybrid.Hybrid = undefined,
    /// The round plan's link (rank 0 sends, the others receive), on its own TCP connection.
    plan: planlink.PlanLink = undefined,
    fate: ?Fate = null,

    /// Brings this rank up in place; `kernel_image` is mailbox.cu's fatbin, empty sends those backends to NCCL.
    pub fn open(s: *Session, d: *const cuda.Driver, cfg: Config, kernel_image: []const u8) !void {
        var b = try Bootstrap.open(cfg);
        defer b.close();
        try b.agree(&cfg.digest(), "settings (TF_TP_WORLD, TF_COMM_BACKEND, TF_TP_MAILBOX_MAX_KB, TF_TP_FAILFAST, TF_TP_ROCE_WIRE / _FALLBACK)");
        var lib = try cuda.nccl.Library.openPath(cfg.nccl_lib);
        errdefer lib.close();
        const v = try lib.version();
        try b.agree(std.mem.asBytes(&v), "NCCL version");
        var id: cuda.nccl.UniqueId = undefined;
        if (cfg.rank == 0) try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try b.broadcast(&id.internal, &id.internal);
        s.* = .{ .cfg = cfg, .lib = lib, .nccl = undefined };
        s.nccl = try nccl.Nccl.init(&s.lib, d, id, cfg.rank, cfg.world);
        errdefer s.nccl.deinit();
        switch (cfg.backend) {
            .nccl => {},
            .mailbox => if (try mailboxPossible(&b, kernel_image)) {
                s.mail = try mailbox.Mailbox.init(d, &b, kernel_image, cfg.mailbox_max_bytes, cfg.mailbox_timeout_ns);
                s.route = .{ .small = .{ .mailbox = &s.mail.? }, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes };
            } else if (cfg.rank == 0) {
                log.warn("[tensorfold] tp: TF_COMM_BACKEND=mailbox needs two ranks on one host and the kernel; every rank uses NCCL", .{});
            },
            .roce => try s.openRoce(d, &b, kernel_image),
        }
        try s.agreeVar(&b);
        s.plan = planlink.PlanLink.init(cfg.rank, cfg.world, b.releasePlan());
        s.plan.spin_ns = cfg.plan_spin_ns;
        s.plan.pin = cfg.plan_pin;
        s.plan.avoid = cfg.roce_cpu;
        if (cfg.failfast) {
            s.fate = Fate.init(cfg, b.release());
            s.fate.?.onFail(.{ .ctx = s, .call = abortHook });
            try s.fate.?.start();
        }
        log.info("[tensorfold] tp: rank {d} of {d} up ({t}, NCCL {d}, fail-fast {s}, roce fast {s}: {d} poller(s), stagger {d} ns)", .{ cfg.rank, cfg.world, s.comm().kind(), v, if (cfg.failfast) "on" else "off", if (cfg.roce_fast & 1 != 0) "on" else "off", (cfg.roce_fast >> 8) & 0xff, cfg.roce_fast >> 16 });
    }

    /// RoCE for the small exchanges, probed against NCCL at load; on failure every rank obeys TF_TP_ROCE_FALLBACK.
    fn openRoce(s: *Session, d: *const cuda.Driver, b: *Bootstrap, image: []const u8) !void {
        const cfg = s.cfg;
        var specs: [roce_verbs.max_hcas]roce_verbs.HcaSpec = undefined;
        var n_specs: usize = 0;
        if (cfg.roce_wire == .verbs) {
            if (cfg.roce_hca) |text| {
                n_specs = try roce_verbs.parseSpecs(text, &specs);
            } else if (Verbs.open(null)) |vb| {
                var v = vb;
                defer v.close();
                n_specs = roce_verbs.autoSpecs(&v, &specs, cfg.roce_hcas);
            } else |_| {}
        }
        const opts: roce.Options = .{ .wire = if (cfg.roce_wire == .shm) .shm else .verbs, .hcas = specs[0..n_specs], .traffic_class = cfg.roce_tc, .cpu = cfg.roce_cpu, .max_bytes = cfg.mailbox_max_bytes, .timeout_ns = cfg.mailbox_timeout_ns, .fast = cfg.roce_fast, .blocks = cfg.roce_blocks };
        const r = roce.Roce.init(d, b, image, opts) catch |e| return s.roceFallback(e);
        s.roce = r;
        s.route = .{ .small = .{ .roce = r }, .nccl = &s.nccl, .max_bytes = cfg.mailbox_max_bytes };
        const good = probe(s, d) catch false;
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [bootstrap.max_ranks]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(good)}, &buf, &blobs);
        for (all) |x| if (x[0] != 1) {
            r.deinit();
            s.roce = null;
            return s.roceFallback(error.ProbeMismatch);
        };
    }

    /// `Collective.varAgreed`: whether every rank's transport moves device lengths, agreed once a session.
    fn agreeVar(s: *Session, b: *Bootstrap) !void {
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [bootstrap.max_ranks]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(s.comm().varFits(16))}, &buf, &blobs);
        var ok = true;
        for (all) |x| ok = ok and x[0] == 1;
        if (s.mail != null or s.roce != null) s.route.var_agreed = ok;
    }

    fn roceFallback(s: *Session, e: anyerror) !void {
        if (s.cfg.roce_fallback == .@"error") return e;
        if (s.cfg.rank == 0) log.warn("[tensorfold] tp: RoCE unavailable ({t}); every rank uses NCCL (TF_TP_ROCE_FALLBACK=error refuses instead)", .{e});
    }

    /// One 4 KiB all-gather and one bf16 sum through RoCE and through NCCL: the bytes must be equal.
    fn probe(s: *Session, d: *const cuda.Driver) !bool {
        const n = 4096;
        // input [0, n), gathers [n, 3n) and [3n, 5n), sums [5n, 6n) and [6n, 7n)
        var a = try cuda.DeviceBuffer.alloc(d, 7 * n);
        defer a.free();
        var stream = try cuda.Stream.init(d, true);
        defer stream.deinit();
        var bytes: [n]u8 = undefined;
        reference.fill(&bytes, .bf16, s.cfg.rank, 0x7072_6f62);
        try a.uploadAsync(0, &bytes, stream.handle);
        const fast = s.route.iface();
        const slow = s.nccl.iface();
        try fast.allGather(a.ptr, a.ptr + n, n, .u8, stream.handle);
        try slow.allGather(a.ptr, a.ptr + 3 * n, n, .u8, stream.handle);
        try fast.allReduce(a.ptr, a.ptr + 5 * n, n / 2, .bf16, .sum, stream.handle);
        try slow.allReduce(a.ptr, a.ptr + 6 * n, n / 2, .bf16, .sum, stream.handle);
        try stream.synchronize();
        try fast.check();
        var got: [6 * n]u8 = undefined;
        try a.download(n, &got);
        const gathers = std.mem.eql(u8, got[0 .. 2 * n], got[2 * n .. 4 * n]);
        const sums = std.mem.eql(u8, got[4 * n .. 5 * n], got[5 * n .. 6 * n]);
        if (!gathers or !sums) log.warn("[tensorfold] tp roce probe: gather {s}, sum {s} against NCCL", .{ if (gathers) "equal" else "DIFFERS", if (sums) "equal" else "DIFFERS" });
        return gathers and sums;
    }

    fn mailboxPossible(b: *Bootstrap, image: []const u8) !bool {
        if (b.world != 2) return false;
        var id_buf: [64]u8 = undefined;
        const id = sock.hostId(&id_buf);
        var mine: [65]u8 = @splat(0);
        mine[0] = @intFromBool(image.len > 0);
        @memcpy(mine[1..][0..id.len], id);
        var blobs: [bootstrap.max_ranks][]const u8 = undefined;
        var buf: [2 * 65]u8 = undefined;
        const all = try b.allGather(&mine, &buf, &blobs);
        return all[0][0] == 1 and std.mem.eql(u8, all[0], all[1]);
    }

    fn abortHook(ctx: *anyopaque, _: []const u8) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        s.comm().abort();
    }

    /// The communicator the model uses: NCCL, or the hybrid route over the mailbox / RoCE.
    pub fn comm(s: *Session) Collective {
        if (s.mail != null or s.roce != null) return s.route.iface();
        return s.nccl.iface();
    }

    /// This rank failed inside a round: every rank goes down (exit 70). Without fail-fast it only aborts.
    pub fn fatal(s: *Session, where: []const u8, err: anyerror) void {
        if (s.fate) |*f| return f.fatal(where, err);
        s.comm().abort();
    }

    /// A clean stop announced by rank 0 before it closes: followers end with no fail-fast exit.
    pub fn announceStop(s: *Session) void {
        if (s.fate) |*f| f.announceStop();
    }

    /// This rank failed after `open`, before serving: the other ranks learn why and go down (fail-fast).
    pub fn bootFailed(s: *Session, err: anyerror) void {
        if (s.fate) |*f| f.failBoot("boot", err);
        s.close();
    }

    /// A normal stop: every rank calls it (rank 0 first marks the stop as expected with `fate.expectStop`).
    pub fn close(s: *Session) void {
        if (s.fate) |*f| f.close();
        s.plan.close();
        if (s.roce) |r| r.deinit();
        if (s.mail) |*m| m.deinit();
        s.nccl.deinit();
        s.lib.close();
    }
};

test {
    _ = collective;
    _ = reference;
    _ = config;
    _ = sock;
    _ = bootstrap;
    _ = mailbox;
    _ = roce_verbs;
    _ = planlink;
    _ = @import("roce_test.zig");
    _ = @import("host_test.zig");
    _ = @import("fate_test.zig");
}
