//! Multi-rank P2P startup diagnostics: NVML and CUDA peer capability, NCCL's transport, versions; never a failure.
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const nvml = @import("p2p_nvml.zig");
const cudap = @import("p2p_cuda.zig");
const ncclog = @import("p2p_nccl_log.zig");
const probe = @import("p2p_probe.zig");

/// TENSORFOLD_P2P_DIAGNOSTICS=0 turns the report (and its NCCL log capture) off.
pub const off_variable = "TENSORFOLD_P2P_DIAGNOSTICS";

/// The settings that change how ranks on one host reach each other, reported when set.
pub const settings = [_][:0]const u8{ "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_P2P_READ_ENABLE", "NCCL_SHM_DISABLE", "NCCL_ALGO", "NCCL_PROTO", "CUDA_VISIBLE_DEVICES" };

pub const Level = enum { info, warn };
pub const Sink = *const fn (Level, []const u8) void;

fn logSink(level: Level, line: []const u8) void {
    switch (level) {
        .info => std.log.info("{s}", .{line}),
        .warn => std.log.warn("{s}", .{line}),
    }
}

/// What NCCL's log said about this rank's connections, or why it says nothing.
pub const Transport = union(enum) { later, not_reported, user, unavailable, links: ncclog.Summary };

pub const ProbeOutcome = union(enum) { done: probe.Result, failed: anyerror };

pub const Facts = struct {
    rank: u32,
    world: u32,
    env: ncclog.Env,
    nccl_version: ?c_int = null,
    host: ?*const nvml.Survey = null,
    cuda: ?*const cudap.Survey = null,
    transport: Transport = .later,
    probe: ?ProbeOutcome = null,
};

/// One rank's diagnostics from just before its first NCCL call to the report after init.
pub const Session = struct {
    rank: u32 = 0,
    world: u32 = 1,
    active: bool = false,
    env: ncclog.Env = ncclog.process,
    sink: Sink = logSink,
    capture: ncclog.Capture = .{},

    /// Defer it at the call site: a failed init still hands NCCL_DEBUG* back and removes the log; else a no-op.
    pub fn end(s: *Session) void {
        s.capture.restore(s.env);
        s.capture.remove();
        s.active = false;
    }
};

/// Call before this rank's first NCCL call: one rank, or TENSORFOLD_P2P_DIAGNOSTICS=0, does nothing at all.
pub fn begin(rank: usize, world: usize) Session {
    return beginWith(ncclog.process, logSink, rank, world);
}

pub fn beginWith(env: ncclog.Env, sink: Sink, rank: usize, world: usize) Session {
    if (world < 2) return .{};
    if (env.get(off_variable)) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return .{};
    return .{ .rank = @intCast(rank), .world = @intCast(world), .active = true, .env = env, .sink = sink, .capture = .begin(env) };
}

pub const After = struct {
    lib: *const nccl.Library,
    comm: nccl.Comm,
    /// The probe's device memory and events; null skips TENSORFOLD_P2P_PROBE.
    d: ?*const driver.Driver = null,
    /// A later `ready` (tp_watch.ready or the group's close) reports the transport if init logged none.
    later: bool = false,
};

/// After this rank's NCCL init: restores NCCL_DEBUG*, runs the opt-in probe, then logs the report.
pub fn afterInit(s: *Session, a: After) void {
    if (!s.active) return;
    // both stay open until exit, as optional_libraries keeps every loader reference it finds
    const nl = nvml.Library.open();
    const cl = cudap.Library.open();
    afterInitWith(s, a, if (nl) |*l| &l.api else null, if (cl) |*l| &l.api else null);
}

pub fn afterInitWith(s: *Session, a: After, nvml_api: ?*const nvml.Api, cuda_api: ?*const cudap.Api) void {
    if (!s.active) return;
    s.active = false;
    s.capture.restore(s.env);
    var f: Facts = .{ .rank = s.rank, .world = s.world, .env = s.env };
    if (a.d) |d| if (probe.wanted(if (s.env.get(probe.variable)) |v| std.mem.span(v) else null)) {
        f.probe = if (probe.run(d, a.lib, a.comm, s.rank, s.world)) |r| .{ .done = r } else |e| .{ .failed = e };
    };
    f.transport = take(&s.capture, s.rank, s.world, s.sink, a.later);
    var v: c_int = 0;
    if (a.lib.api.ncclGetVersion(&v) == 0) f.nccl_version = v;
    var cs: cudap.Survey = .{};
    if (cuda_api) |api| {
        cudap.survey(api, &cs);
        f.cuda = &cs;
    }
    var ns: nvml.Survey = .{};
    var none: [0]nvml.Pair = .{};
    if (nvml_api) |api| {
        nvml.survey(api, if (cs.current()) |d| d.key else null, &ns, &none);
        f.host = &ns;
    }
    emit(s.sink, &f);
}

var log_buf: [1 << 20]u8 = undefined;
const Pending = struct { capture: ncclog.Capture, rank: u32, world: u32, sink: Sink };
var pending: ?Pending = null;

fn take(c: *ncclog.Capture, rank: u32, world: u32, sink: Sink, later: bool) Transport {
    switch (c.state) {
        .user => return .user,
        .unavailable => return .unavailable,
        .off => return .not_reported,
        .on => {},
    }
    const found = ncclog.parse(c.read(&log_buf), rank);
    if (found.count == 0 and later and pending == null) {
        pending = .{ .capture = c.*, .rank = rank, .world = world, .sink = sink };
        c.state = .off;
        return .later;
    }
    c.remove();
    return if (found.count > 0) .{ .links = found } else .not_reported;
}

/// The deferred transport line once the ranks have exchanged (tp_watch.ready, or the group's close at the latest).
pub fn ready() void {
    var p = pending orelse return;
    pending = null;
    const t = take(&p.capture, p.rank, p.world, p.sink, false);
    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    transportLine(&w, t) catch {};
    lines(p.sink, .info, w.buffered());
}

fn emit(sink: Sink, f: *const Facts) void {
    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    info(&w, f) catch {};
    lines(sink, .info, w.buffered());
    w = std.Io.Writer.fixed(&buf);
    warnings(&w, f) catch {};
    lines(sink, .warn, w.buffered());
}

fn lines(sink: Sink, level: Level, text: []const u8) void {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > 0) sink(level, l);
}

fn yes(v: ?c_int) []const u8 {
    return if (v) |x| (if (x != 0) "yes" else "no") else "?";
}

fn name(v: ?c_int, comptime f: fn (c_int) []const u8) []const u8 {
    return if (v) |x| f(x) else "?";
}

/// The report's facts, one line each: this rank's GPU and versions, each other GPU, peer access, settings, NCCL.
pub fn info(w: *std.Io.Writer, f: *const Facts) !void {
    const host = f.host;
    const mine: ?nvml.Gpu = if (host) |h| (if (h.mine) |m| h.gpus[m] else null) else null;
    const cur: ?*const cudap.Dev = if (f.cuda) |c| c.current() else null;
    try w.print("p2p: rank {d} of {d}", .{ f.rank, f.world });
    if (mine) |g| try w.print(": GPU {d} {s} ({s})", .{ g.index, g.name.slice(), g.bus.slice() }) else if (cur) |d| try w.print(": CUDA device {d} {s} ({s})", .{ d.ordinal, d.name.slice(), d.bus.slice() });
    try w.writeAll("; driver ");
    if (host != null and host.?.driver.len > 0) try w.writeAll(host.?.driver.slice()) else try w.writeAll("?");
    if (f.cuda) |c| if (c.driver_api) |v| try w.print(", CUDA driver API {d}.{d}", .{ @divTrunc(v, 1000), @divTrunc(@mod(v, 1000), 10) });
    if (f.nccl_version) |v| try w.print(", NCCL {d}.{d}.{d}", .{ @divTrunc(v, 10000), @mod(@divTrunc(v, 100), 100), @mod(v, 100) });
    try w.writeAll("\n");
    if (host) |h| {
        if (h.failed) |what| try w.print("p2p: NVML {s} failed ({d}): P2P status and PCIe paths not checked\n", .{ what, h.code });
        if (h.failed == null and h.mine == null) try w.writeAll("p2p: NVML lists no GPU on this rank's bus: P2P status not checked\n");
        if (h.mine) |m| for (h.list(), 0..) |g, i| if (i != m) {
            try w.print("p2p: GPU {d} {s} ({s}): P2P read {s}, write {s}, path {s}", .{ g.index, g.name.slice(), g.bus.slice(), name(g.read, nvml.statusName), name(g.write, nvml.statusName), name(g.path, nvml.levelName) });
            if (f.cuda) |c| if (c.onBus(g.key)) |d| try cudaPeer(w, d);
            try w.writeAll("\n");
        };
    } else try w.writeAll("p2p: NVML (libnvidia-ml.so.1) not found: P2P status and PCIe paths not checked\n");
    if (host == null or host.?.mine == null) if (f.cuda) |c| if (c.mine) |m| for (c.list(), 0..) |d, i| if (i != m) {
        try w.print("p2p: CUDA device {d} {s} ({s})", .{ d.ordinal, d.name.slice(), d.bus.slice() });
        try cudaPeer(w, &d);
        try w.writeAll("\n");
    };
    try w.writeAll("p2p: peer access: TensorFold enables none itself (no cuCtxEnablePeerAccess); NCCL sets up its own peers\n");
    var first = true;
    for (settings) |s| if (f.env.get(s)) |v| {
        try w.print("{s}{s}={s}", .{ if (first) "p2p: set: " else ", ", s, std.mem.span(v) });
        first = false;
    };
    if (!first) try w.writeAll("\n");
    try transportLine(w, f.transport);
    if (f.probe) |p| switch (p) {
        .done => |r| try w.print("p2p: probe: all-gather of {d} MiB a rank in {d:.2} ms, bus bandwidth {d:.1} GB/s\n", .{ probe.bytes_per_rank >> 20, r.ms, r.bus_gbps }),
        .failed => {},
    };
}

fn cudaPeer(w: *std.Io.Writer, d: *const cudap.Dev) !void {
    try w.print("; CUDA peer access {s}, native atomics {s}", .{ yes(d.can_access), yes(d.atomics) });
    if (d.performance_rank) |r| try w.print(", performance rank {d}", .{r});
}

fn transportLine(w: *std.Io.Writer, t: Transport) !void {
    switch (t) {
        .later => {},
        .links => |s| {
            try w.writeAll("p2p: NCCL transport (this rank's connections): ");
            try ncclog.render(w, &s);
            try w.writeAll("\n");
        },
        .not_reported => try w.writeAll("p2p: NCCL transport not reported: NCCL logged no connection by then (it connects on first use unless NCCL_RUNTIME_CONNECT=0)\n"),
        .user => try w.writeAll("p2p: NCCL transport not captured: NCCL_DEBUG, NCCL_DEBUG_SUBSYS or NCCL_DEBUG_FILE is set, so NCCL logs as the user asked\n"),
        .unavailable => try w.writeAll("p2p: NCCL transport not captured: no private temporary directory for NCCL's log\n"),
    }
}

/// What an NVML P2P status means (its NVML name) and where the user could look; no setting is recommended.
pub fn meaning(s: c_int) []const u8 {
    return switch (s) {
        1 => "the chipset does not support P2P between these GPUs; check the platform's PCIe chipset and `nvidia-smi topo -p2p rw`",
        2 => "the GPU does not support P2P; check this GPU model's P2P support and `nvidia-smi topo -p2p rw`",
        3 => "the I/O hub (PCIe) topology between these GPUs does not support P2P; `nvidia-smi topo -m` shows their path",
        4 => "a driver registry setting disables P2P; check the NVIDIA driver's registry settings (RegistryDwords in /proc/driver/nvidia/params)",
        5 => "P2P is not supported between these GPUs; `nvidia-smi topo -p2p rw` shows the driver's view",
        6 => "NVML could not determine P2P support; `nvidia-smi topo -p2p rw` shows the driver's view",
        else => "NVML returned a status this build does not know; `nvidia-smi topo -p2p rw` shows the driver's view",
    };
}

/// One line per problem, each naming what NVML or CUDA reported and what the user could check.
pub fn warnings(w: *std.Io.Writer, f: *const Facts) !void {
    if (f.probe) |p| if (p == .failed) try w.print("p2p: probe failed ({t}); startup continues without its figure\n", .{p.failed});
    const h = f.host orelse return cudaWarnings(w, f, null);
    const m = h.mine orelse return cudaWarnings(w, f, null);
    const me = h.gpus[m].index;
    for (h.list(), 0..) |g, i| if (i != m) {
        if (g.read != null and g.read == g.write and g.read.? != 0) {
            try w.print("p2p: GPU {d} -> GPU {d}: NVML P2P read and write {s}: {s}\n", .{ me, g.index, nvml.statusName(g.read.?), meaning(g.read.?) });
            continue;
        }
        inline for (.{ "read", "write" }) |c| if (@field(g, c)) |s| if (s != 0)
            try w.print("p2p: GPU {d} -> GPU {d}: NVML P2P {s} {s}: {s}\n", .{ me, g.index, c, nvml.statusName(s), meaning(s) });
    };
    try cudaWarnings(w, f, h);
}

fn cudaWarnings(w: *std.Io.Writer, f: *const Facts, h: ?*const nvml.Survey) !void {
    const c = f.cuda orelse return;
    const m = c.mine orelse return;
    for (c.list(), 0..) |d, i| if (i != m) if (d.can_access) |x| if (x == 0) {
        const why = if (h != null and h.?.mine != null) "the NVML status above names NVML's reason" else "`nvidia-smi topo -p2p rw` shows the driver's view";
        try w.print("p2p: CUDA device {d} -> {d}: cuDeviceCanAccessPeer reports no peer access; {s}\n", .{ c.devs[m].ordinal, d.ordinal, why });
    };
}

/// `capabilities --json`'s "p2p": the driver, NVML's GPUs and every pair, only on a host with two or more GPUs.
pub fn compact(a: std.mem.Allocator) ?[]const u8 {
    const l = nvml.Library.open() orelse return null;
    return compactWith(a, &l.api);
}

pub fn compactWith(a: std.mem.Allocator, api: *const nvml.Api) ?[]const u8 {
    var s: nvml.Survey = .{};
    var all: [nvml.max_gpus * (nvml.max_gpus - 1) / 2]nvml.Pair = undefined;
    nvml.survey(api, null, &s, &all);
    if (s.failed != null or s.count < 2) return null;
    const G = struct { index: u32, name: []const u8, bus: []const u8 };
    const P = struct { gpus: [2]u32, read: ?[]const u8, write: ?[]const u8, path: ?[]const u8 };
    var gpus: [nvml.max_gpus]G = undefined;
    for (s.list(), 0..) |*g, i| gpus[i] = .{ .index = g.index, .name = g.name.slice(), .bus = g.bus.slice() };
    var ps: [all.len]P = undefined;
    for (all[0..s.pair_count], 0..) |p, i| ps[i] = .{ .gpus = .{ p.a, p.b }, .read = opt(p.read, nvml.statusName), .write = opt(p.write, nvml.statusName), .path = opt(p.path, nvml.levelName) };
    const doc = .{ .driver = s.driver.slice(), .gpus = gpus[0..s.count], .pairs = ps[0..s.pair_count] };
    return std.json.Stringify.valueAlloc(a, doc, .{}) catch null;
}

fn opt(v: ?c_int, comptime f: fn (c_int) []const u8) ?[]const u8 {
    return if (v) |x| f(x) else null;
}
