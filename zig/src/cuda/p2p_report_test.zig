//! P2P startup diagnostics over fake NVML, CUDA and NCCL tables: every reason, missing libraries, one rank.
const std = @import("std");
const builtin = @import("builtin");
const nccl = @import("nccl.zig");
const nvml = @import("p2p_nvml.zig");
const cudap = @import("p2p_cuda.zig");
const report = @import("p2p_report.zig");
const log_test = @import("p2p_nccl_log_test.zig");
const FakeEnv = log_test.fake_env;

const nvml_buses = [_][]const u8{ "00000000:01:00.0", "00000000:C1:00.0", "00000000:41:00.0", "00000000:81:00.0" };
const cuda_buses = [_][]const u8{ "0000:01:00.0", "0000:C1:00.0", "0000:41:00.0", "0000:81:00.0" };

fn fill(buf: [*]u8, len: c_uint, text: []const u8) void {
    const n = @min(text.len, len - 1);
    @memcpy(buf[0..n], text[0..n]);
    buf[n] = 0;
}

const FakeNvml = struct {
    var init_result: c_int = 0;
    var count: c_uint = 2;
    var read: c_int = 0;
    var write: c_int = 0;
    var path: c_int = 30;

    fn reset() void {
        init_result = 0;
        count = 2;
        read = 0;
        write = 0;
        path = 30;
    }
    fn idx(d: nvml.Device) usize {
        return @intFromPtr(d) - 1;
    }
    fn init() callconv(.c) c_int {
        return init_result;
    }
    fn shutdown() callconv(.c) c_int {
        return 0;
    }
    fn driver(buf: [*]u8, len: c_uint) callconv(.c) c_int {
        fill(buf, len, "580.65.06");
        return 0;
    }
    fn getCount(n: *c_uint) callconv(.c) c_int {
        n.* = count;
        return 0;
    }
    fn byIndex(i: c_uint, d: *nvml.Device) callconv(.c) c_int {
        d.* = @ptrFromInt(i + 1);
        return 0;
    }
    fn name(_: nvml.Device, buf: [*]u8, len: c_uint) callconv(.c) c_int {
        fill(buf, len, "NVIDIA RTX PRO 6000");
        return 0;
    }
    fn pci(d: nvml.Device, p: *nvml.PciInfo) callconv(.c) c_int {
        fill(&p.bus_id, p.bus_id.len, nvml_buses[idx(d)]);
        return 0;
    }
    fn p2p(_: nvml.Device, _: nvml.Device, cap: c_int, s: *c_int) callconv(.c) c_int {
        s.* = if (cap == nvml.caps_read) read else write;
        return 0;
    }
    fn topo(_: nvml.Device, _: nvml.Device, l: *c_int) callconv(.c) c_int {
        l.* = path;
        return 0;
    }
    const api: nvml.Api = .{
        .nvmlInit_v2 = init,
        .nvmlShutdown = shutdown,
        .nvmlSystemGetDriverVersion = driver,
        .nvmlDeviceGetCount_v2 = getCount,
        .nvmlDeviceGetHandleByIndex_v2 = byIndex,
        .nvmlDeviceGetName = name,
        .nvmlDeviceGetPciInfo_v3 = pci,
        .nvmlDeviceGetP2PStatus = p2p,
        .nvmlDeviceGetTopologyCommonAncestor = topo,
    };
};

const FakeCuda = struct {
    var count: c_int = 2;
    var current: c_int = 0;
    var access: c_int = 1;

    fn reset() void {
        count = 2;
        current = 0;
        access = 1;
    }
    fn version(v: *c_int) callconv(.c) c_int {
        v.* = 13000;
        return 0;
    }
    fn ctxDevice(d: *c_int) callconv(.c) c_int {
        d.* = current;
        return 0;
    }
    fn getCount(n: *c_int) callconv(.c) c_int {
        n.* = count;
        return 0;
    }
    fn get(d: *c_int, ordinal: c_int) callconv(.c) c_int {
        d.* = ordinal;
        return 0;
    }
    fn name(buf: [*]u8, len: c_int, _: c_int) callconv(.c) c_int {
        fill(buf, @intCast(len), "NVIDIA RTX PRO 6000");
        return 0;
    }
    fn bus(buf: [*]u8, len: c_int, d: c_int) callconv(.c) c_int {
        fill(buf, @intCast(len), cuda_buses[@intCast(d)]);
        return 0;
    }
    fn canAccess(x: *c_int, _: c_int, _: c_int) callconv(.c) c_int {
        x.* = access;
        return 0;
    }
    fn attribute(x: *c_int, a: c_int, _: c_int, _: c_int) callconv(.c) c_int {
        x.* = if (a == cudap.attr_performance_rank) 0 else access;
        return 0;
    }
    const api: cudap.Api = .{
        .cuDriverGetVersion = version,
        .cuCtxGetDevice = ctxDevice,
        .cuDeviceGetCount = getCount,
        .cuDeviceGet = get,
        .cuDeviceGetName = name,
        .cuDeviceGetPCIBusId = bus,
        .cuDeviceCanAccessPeer = canAccess,
        .cuDeviceGetP2PAttribute = attribute,
    };
};

const Lines = struct {
    var buf: [16384]u8 = undefined;
    var len: usize = 0;
    var warns: usize = 0;

    fn reset() void {
        len = 0;
        warns = 0;
    }
    fn sink(level: report.Level, line: []const u8) void {
        if (level == .warn) warns += 1;
        const out = std.fmt.bufPrint(buf[len..], "{s} {s}\n", .{ if (level == .warn) "W" else "I", line }) catch return;
        len += out.len;
    }
    fn text() []const u8 {
        return buf[0..len];
    }
    fn has(s: []const u8) bool {
        return std.mem.indexOf(u8, text(), s) != null;
    }
};

fn ncclVersion(v: *c_int) callconv(.c) nccl.Result {
    v.* = 23007;
    return 0;
}

fn noNcclVersion(_: *c_int) callconv(.c) nccl.Result {
    return 1;
}

/// One rank's whole report with the fakes: begin, then afterInit (no probe: no driver is given).
fn run(rank: usize, world: usize, with_nvml: bool, with_cuda: bool) !void {
    Lines.reset();
    var lib: nccl.Library = undefined;
    lib.api.ncclGetVersion = ncclVersion;
    var s = report.beginWith(FakeEnv.env, Lines.sink, rank, world);
    report.afterInitWith(&s, .{ .lib = &lib, .comm = null }, if (with_nvml) &FakeNvml.api else null, if (with_cuda) &FakeCuda.api else null);
}

fn reset() void {
    FakeEnv.reset();
    FakeNvml.reset();
    FakeCuda.reset();
    FakeEnv.put("NCCL_DEBUG", "WARN"); // most cases leave the log capture out; its own cases unset this
}

test "p2p report: a healthy pair names both GPUs, OK both ways, the path, CUDA peer access and the versions" {
    reset();
    try run(0, 2, true, true);
    try std.testing.expect(Lines.has("I p2p: rank 0 of 2: GPU 0 NVIDIA RTX PRO 6000 (00000000:01:00.0); driver 580.65.06, CUDA driver API 13.0, NCCL 2.30.7\n"));
    try std.testing.expect(Lines.has("I p2p: GPU 1 NVIDIA RTX PRO 6000 (00000000:C1:00.0): P2P read OK, write OK, path PHB; CUDA peer access yes, native atomics yes, performance rank 0\n"));
    try std.testing.expect(Lines.has("I p2p: peer access: TensorFold enables none itself"));
    try std.testing.expectEqual(@as(usize, 0), Lines.warns);
}

test "p2p report: each unavailable NVML status is one warning naming it and what to check" {
    const want = [_][]const u8{ "", "CHIPSET_NOT_SUPPORTED: the chipset", "GPU_NOT_SUPPORTED: the GPU", "IOH_TOPOLOGY_NOT_SUPPORTED: the I/O hub", "DISABLED_BY_REGKEY: a driver registry setting", "NOT_SUPPORTED: P2P is not supported", "UNKNOWN: NVML could not determine" };
    for (want[1..], 1..) |w, s| {
        reset();
        FakeNvml.read = @intCast(s);
        FakeNvml.write = @intCast(s);
        FakeCuda.access = 0;
        try run(0, 2, true, true);
        try std.testing.expectEqual(@as(usize, 2), Lines.warns); // NVML's reason, then CUDA's no access pointing at it
        var line: [128]u8 = undefined;
        try std.testing.expect(Lines.has(try std.fmt.bufPrint(&line, "W p2p: GPU 0 -> GPU 1: NVML P2P read and write {s}", .{w})));
        try std.testing.expect(Lines.has("W p2p: CUDA device 0 -> 1: cuDeviceCanAccessPeer reports no peer access; the NVML status above"));
    }
}

test "p2p report: read and write that differ get a line each, OK staying silent" {
    reset();
    FakeNvml.read = 4;
    try run(0, 2, true, true);
    try std.testing.expectEqual(@as(usize, 1), Lines.warns);
    try std.testing.expect(Lines.has("W p2p: GPU 0 -> GPU 1: NVML P2P read DISABLED_BY_REGKEY: "));
}

test "p2p report: NVML missing says so and still reports CUDA's peers" {
    reset();
    FakeCuda.access = 0;
    try run(1, 2, false, true);
    try std.testing.expect(Lines.has("I p2p: rank 1 of 2: CUDA device 0 NVIDIA RTX PRO 6000 (0000:01:00.0); driver ?, CUDA driver API 13.0"));
    try std.testing.expect(Lines.has("I p2p: NVML (libnvidia-ml.so.1) not found: P2P status and PCIe paths not checked\n"));
    try std.testing.expect(Lines.has("I p2p: CUDA device 1 NVIDIA RTX PRO 6000 (0000:C1:00.0); CUDA peer access no"));
    try std.testing.expect(Lines.has("W p2p: CUDA device 0 -> 1: cuDeviceCanAccessPeer reports no peer access; `nvidia-smi topo -p2p rw`"));
}

test "p2p report: NVML that fails to initialize is reported by call and code, never fatal" {
    reset();
    FakeNvml.init_result = 9;
    try run(0, 2, true, true);
    try std.testing.expect(Lines.has("I p2p: NVML nvmlInit_v2 failed (9): P2P status and PCIe paths not checked\n"));
    try std.testing.expect(Lines.has("I p2p: CUDA device 1 "));
}

test "p2p report: more devices than one peer: NVML sees four GPUs where CUDA_VISIBLE_DEVICES shows one" {
    reset();
    FakeNvml.count = 4;
    FakeCuda.count = 1;
    FakeEnv.put("CUDA_VISIBLE_DEVICES", "0");
    try run(0, 4, true, true);
    for ([_][]const u8{ "GPU 1 ", "GPU 2 ", "GPU 3 " }) |g| {
        var line: [64]u8 = undefined;
        try std.testing.expect(Lines.has(try std.fmt.bufPrint(&line, "I p2p: {s}NVIDIA", .{g})));
    }
    try std.testing.expect(!Lines.has("CUDA peer access"));
    try std.testing.expect(Lines.has("I p2p: set: CUDA_VISIBLE_DEVICES=0\n"));
}

test "p2p report: one rank changes nothing and prints nothing" {
    FakeEnv.reset();
    Lines.reset();
    var s = report.beginWith(FakeEnv.env, Lines.sink, 0, 1);
    try std.testing.expect(!s.active);
    report.afterInitWith(&s, .{ .lib = undefined, .comm = null }, &FakeNvml.api, &FakeCuda.api);
    try std.testing.expectEqual(@as(usize, 0), Lines.len);
    try std.testing.expectEqual(@as(usize, 0), FakeEnv.sets + FakeEnv.unsets);
}

test "p2p report: TENSORFOLD_P2P_DIAGNOSTICS=0 turns it off on every rank count" {
    FakeEnv.reset();
    FakeEnv.put(report.off_variable, "0");
    Lines.reset();
    var s = report.beginWith(FakeEnv.env, Lines.sink, 0, 8);
    report.afterInitWith(&s, .{ .lib = undefined, .comm = null }, &FakeNvml.api, &FakeCuda.api);
    try std.testing.expectEqual(@as(usize, 0), Lines.len + FakeEnv.sets);
}

test "p2p report: every library missing and NCCL's version unreadable still reports, never failing startup" {
    reset();
    Lines.reset();
    var lib: nccl.Library = undefined;
    lib.api.ncclGetVersion = noNcclVersion;
    var s = report.beginWith(FakeEnv.env, Lines.sink, 1, 2);
    report.afterInitWith(&s, .{ .lib = &lib, .comm = null }, null, null);
    try std.testing.expect(Lines.has("I p2p: rank 1 of 2; driver ?\n"));
    try std.testing.expect(Lines.has("NCCL transport not captured: NCCL_DEBUG, NCCL_DEBUG_SUBSYS or NCCL_DEBUG_FILE is set"));
    try std.testing.expectEqual(@as(usize, 0), Lines.warns);
}

test "p2p report: NCCL's log names the transport at init, its environment handed back" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    FakeEnv.reset();
    Lines.reset();
    var lib: nccl.Library = undefined;
    lib.api.ncclGetVersion = ncclVersion;
    var s = report.beginWith(FakeEnv.env, Lines.sink, 0, 2);
    try log_test.writeLog("gpu-a:1:2 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct\n");
    report.afterInitWith(&s, .{ .lib = &lib, .comm = null, .later = true }, null, null);
    try std.testing.expect(Lines.has("I p2p: NCCL transport (this rank's connections): rank 1 SHM (1 connection)\n"));
    try std.testing.expect(FakeEnv.get("NCCL_DEBUG") == null);
    report.ready(); // nothing was deferred: no second line
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, Lines.text(), "NCCL transport"));
}

test "p2p report: connections NCCL makes on first use are reported at ready; none at all says not reported" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    for ([_]bool{ true, false }) |connects| {
        FakeEnv.reset();
        Lines.reset();
        var lib: nccl.Library = undefined;
        lib.api.ncclGetVersion = ncclVersion;
        var s = report.beginWith(FakeEnv.env, Lines.sink, 1, 2);
        try log_test.writeLog("gpu-a:1:2 [1] NCCL INFO NCCL version 2.30.7+cuda13.0\n");
        const file = try std.testing.allocator.dupe(u8, std.mem.span(FakeEnv.get("NCCL_DEBUG_FILE").?));
        defer std.testing.allocator.free(file);
        report.afterInitWith(&s, .{ .lib = &lib, .comm = null, .later = true }, null, null);
        try std.testing.expect(!Lines.has("NCCL transport"));
        FakeEnv.put("NCCL_DEBUG_FILE", file); // where NCCL keeps writing after the environment is handed back
        if (connects) try log_test.writeLog("gpu-a:1:2 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/CUMEM\n");
        report.ready();
        try std.testing.expect(Lines.has(if (connects) "I p2p: NCCL transport (this rank's connections): rank 0 P2P write (1 connection)\n" else "I p2p: NCCL transport not reported: NCCL logged no connection"));
    }
}

test "p2p report: a failed probe is one warning naming its error, and no bandwidth line" {
    const f: report.Facts = .{ .rank = 0, .world = 2, .env = FakeEnv.env, .transport = .not_reported, .probe = .{ .failed = error.ProbeTimedOut } };
    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try report.warnings(&w, &f);
    try std.testing.expectEqualStrings("p2p: probe failed (ProbeTimedOut); startup continues without its figure\n", w.buffered());
    w = std.Io.Writer.fixed(&buf);
    try report.info(&w, &f);
    try std.testing.expect(std.mem.indexOf(u8, w.buffered(), "probe") == null);
}

test "p2p compact: two GPUs give the driver, both GPUs and their pair; one GPU gives nothing" {
    FakeNvml.reset();
    FakeNvml.write = 4;
    const doc = report.compactWith(std.testing.allocator, &FakeNvml.api).?;
    defer std.testing.allocator.free(doc);
    try std.testing.expectEqualStrings(
        \\{"driver":"580.65.06","gpus":[{"index":0,"name":"NVIDIA RTX PRO 6000","bus":"00000000:01:00.0"},{"index":1,"name":"NVIDIA RTX PRO 6000","bus":"00000000:C1:00.0"}],"pairs":[{"gpus":[0,1],"read":"OK","write":"DISABLED_BY_REGKEY","path":"PHB"}]}
    , doc);
    FakeNvml.count = 1;
    try std.testing.expect(report.compactWith(std.testing.allocator, &FakeNvml.api) == null);
}
