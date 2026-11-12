//! kernel-contract's opt-in parallel run: `--jobs N` starts N shards of this binary and prints one merged SUMMARY.
const std = @import("std");
const cuda = @import("cuda");
const shard = @import("contract_shard.zig");
const dump = @import("byte_dump.zig");
const children = @import("contract_children.zig");

/// The options of a `kernel-contract` command line, or null when `args` is another command's or does not parse.
fn options(a: std.mem.Allocator, args: []const [:0]const u8) ?shard.Options {
    if (args.len < 2 or !std.mem.eql(u8, args[1], "kernel-contract")) return null;
    return shard.parse(a, args[2..]) catch null;
}

/// The visible GPU a `kernel-contract` shard runs on; 0 for every other command.
pub fn device(a: std.mem.Allocator, args: []const [:0]const u8) c_int {
    const opt = options(a, args) orelse return 0;
    defer opt.deinit(a);
    return opt.device;
}

/// Before any GPU context: `--help`, a refused command line or `--jobs` ends the process with its exit code; else null.
pub fn parent(init: std.process.Init, driver: *const cuda.Driver, args: []const [:0]const u8) !?u8 {
    if (args.len < 2 or !std.mem.eql(u8, args[1], "kernel-contract")) return null;
    for (args[2..]) |arg| if (std.mem.eql(u8, arg, "--help")) {
        std.debug.print("{s}", .{shard.help});
        return 0;
    };
    var culprit: shard.Culprit = .{};
    const opt = shard.parseNamed(init.gpa, args[2..], &culprit) catch |err| {
        std.debug.print("FAIL kernel-contract: {f}: {t} ({s})\n{s}", .{ culprit, err, shard.hint(err), shard.help });
        return 2;
    };
    defer opt.deinit(init.gpa);
    if (opt.parent) |pid| children.followParent(pid) catch |err| {
        std.debug.print("FAIL kernel-contract: --parent {d}: {t}, so this shard stops rather than run on\n", .{ pid, err });
        return 2;
    };
    const gpus = std.math.cast(u32, try driver.deviceCount()) orelse 0;
    if (opt.shard != null and opt.device >= gpus) {
        std.debug.print("FAIL kernel-contract: --device {d}: only {d} GPUs (NoSuchDevice)\n", .{ opt.device, gpus });
        return 2;
    }
    const jobs = opt.jobs orelse return null;
    if (dump.active()) {
        std.debug.print("FAIL kernel-contract: --jobs {d}: TF_CUDA_DUMP=1 would write every record {d} times (DumpWithJobs)\n", .{ jobs, jobs });
        return 2;
    }
    fit(init.io, driver, jobs, opt.words.len > 0, gpus) catch return 2;
    return try run(init, opt, jobs, gpus);
}

/// Whether `jobs` processes fit the GPUs they would use and the host; a refusal is printed with what it compared.
fn fit(io: std.Io, driver: *const cuda.Driver, jobs: u32, filtered: bool, gpus: u32) !void {
    var seen: [shard.max_jobs]shard.Gpu = undefined;
    var m: shard.Machine = .{ .gpus = gpus, .gpu_mib = std.math.maxInt(u64), .host_mib = 0 };
    const used: u8 = @intCast(@min(gpus, jobs));
    var at: u8 = 0;
    while (at < used) : (at += 1) {
        const ordinal: c_int = at;
        var dev: cuda.abi.Device = 0;
        try driver.check(driver.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        seen[at] = .{ .sm = try driver.capability(ordinal), .multiprocessors = 0 };
        var count: c_int = 0;
        try driver.check(driver.api.cuDeviceGetAttribute(&count, .multiprocessor_count, dev), "cuDeviceGetAttribute");
        seen[at].multiprocessors = std.math.cast(u32, count) orelse 0;
        try driver.check(driver.api.cuDeviceGetName(&seen[at].name, seen[at].name.len - 1, dev), "cuDeviceGetName");
        var bytes: usize = 0;
        try driver.check(driver.api.cuDeviceTotalMem_v2(&bytes, dev), "cuDeviceTotalMem");
        m.gpu_mib = @min(m.gpu_mib, bytes >> 20);
        var shared: c_int = 0;
        try driver.check(driver.api.cuDeviceGetAttribute(&shared, .integrated, dev), "cuDeviceGetAttribute");
        m.integrated = m.integrated or shared != 0;
    }
    if (shard.firstMixed(seen[0..used])) |mix| {
        var line: [256]u8 = undefined;
        var w: std.Io.Writer = .fixed(&line);
        shard.writeMix(&w, seen[0..used], mix) catch {};
        std.debug.print("FAIL kernel-contract: --jobs {d}: {s}, so their shards would not list the same kernels (MixedGpus: {t})\n", .{ jobs, w.buffered(), mix.field });
        return error.MixedGpus;
    }
    var buf: [8192]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch |err| {
        std.debug.print("FAIL kernel-contract: --jobs {d}: cannot read /proc/meminfo, which --jobs needs (Linux only): {t}\n", .{ jobs, err });
        return err;
    };
    m.host_mib = shard.memAvailableMib(text) orelse {
        std.debug.print("FAIL kernel-contract: --jobs {d}: /proc/meminfo has no MemAvailable line (UnreadableMeminfo)\n", .{jobs});
        return error.UnreadableMeminfo;
    };
    shard.afford(jobs, filtered, m) catch |err| {
        const host = if (filtered) shard.host_mib_filtered else shard.host_mib_full;
        std.debug.print("FAIL kernel-contract: --jobs {d}: {t} (each process is planned for {d} GiB of host and {d} GiB of GPU memory; {d} GiB of host memory is available, {d} GPUs have {d} GiB in total{s})\n", .{ jobs, err, host >> 10, shard.device_mib_per_process >> 10, m.host_mib >> 10, m.gpus, m.gpu_mib >> 10, if (m.integrated) ", sharing the host's" else "" });
        return err;
    };
}

/// One child: its command, what it printed that the parent reads, and how it ended.
const Job = struct {
    io: std.Io,
    slot: usize,
    jobs: u32,
    argv: []const []const u8,
    report: ?shard.Report = null,
    exit_code: ?u8 = null,
    term: ?std.process.Child.Term = null,
    failure: ?anyerror = null,

    fn finished(job: *const Job) bool {
        return job.failure == null and shard.accepted(job.exit_code, job.report);
    }

    fn watch(job: *Job) void {
        job.drive() catch |err| {
            job.failure = err;
        };
        if (!job.finished() and children.stopAll(job.slot, .TERM))
            std.debug.print("FAIL kernel-contract: shard {d}/{d} did not finish, so the other shards are stopped\n", .{ job.slot, job.jobs });
    }

    /// Prints each line the child writes, whatever its length, and keeps its TALLY line.
    const Lines = struct {
        job: *Job,

        pub fn line(l: Lines, text: []const u8, cut: bool) !void {
            if (cut) std.debug.print("{s} [line cut at {d} bytes]\n", .{ text, text.len }) else std.debug.print("{s}\n", .{text});
            if (!std.mem.startsWith(u8, text, "TALLY ")) return;
            if (cut) return error.BadTally;
            l.job.report = try shard.Report.parseLine(text);
        }
    };

    fn drive(job: *Job) !void {
        var child = try std.process.spawn(job.io, .{ .argv = job.argv, .stdin = .ignore, .stderr = .pipe });
        defer child.kill(job.io);
        children.register(job.slot, child.id.?);
        defer children.forget(job.slot);
        if (children.isStopping()) std.posix.kill(child.id.?, .TERM) catch {};
        var buf: [1 << 16]u8 = undefined;
        var reader = child.stderr.?.readerStreaming(job.io, &buf);
        try shard.eachLine(&reader.interface, Lines{ .job = job });
        const term = try child.wait(job.io);
        children.forget(job.slot);
        job.term = term;
        if (term == .exited) job.exit_code = term.exited;
    }
};

fn run(init: std.process.Init, opt: shard.Options, jobs: u32, gpus: u32) !u8 {
    const a = init.arena.allocator();
    const exe = try std.process.executablePathAlloc(init.io, a);
    const me = try std.fmt.allocPrint(a, "{d}", .{children.selfPid()});
    var all: [shard.max_jobs]Job = undefined;
    for (all[0..jobs], 0..) |*job, k| {
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(a, &.{ exe, "kernel-contract", "--shard", try std.fmt.allocPrint(a, "{d}/{d}", .{ k, jobs }), "--device", try std.fmt.allocPrint(a, "{d}", .{k % gpus}), "--parent", me });
        try argv.appendSlice(a, opt.words);
        job.* = .{ .io = init.io, .slot = k, .jobs = jobs, .argv = argv.items };
    }
    children.forwardSignals();
    var threads: [shard.max_jobs]?std.Thread = @splat(null);
    for (all[0..jobs], threads[0..jobs]) |*job, *thread| thread.* = std.Thread.spawn(.{}, Job.watch, .{job}) catch |err| {
        job.failure = err;
        _ = children.stopAll(job.slot, .TERM);
        continue;
    };
    std.debug.print("kernel-contract: {d} shards over {d} GPUs\n", .{ jobs, gpus });
    for (threads[0..jobs]) |t| if (t) |thread| thread.join();
    var reports: [shard.max_jobs]shard.Report = undefined;
    var unfinished: usize = 0;
    for (all[0..jobs], 0..) |job, k| {
        if (job.finished()) {
            reports[k] = job.report.?;
            continue;
        }
        unfinished += 1;
        if (job.failure) |err| {
            std.debug.print("FAIL kernel-contract: shard {d}/{d}: {t}\n", .{ k, jobs, err });
        } else if (children.wasStopped(k) and job.term != null and job.term.? == .signal) {
            std.debug.print("kernel-contract: shard {d}/{d} was stopped: {f}\n", .{ k, jobs, job.term.? });
        } else if (job.term) |term| {
            std.debug.print("FAIL kernel-contract: shard {d}/{d} did not finish: {f}, {s} its TALLY line\n", .{ k, jobs, term, if (job.report == null) "without" else "with" });
        }
    }
    if (unfinished > 0) {
        std.debug.print("FAIL kernel-contract: {d} of {d} shards did not finish, so there is no merged SUMMARY\n", .{ unfinished, jobs });
        return 1;
    }
    const sum = shard.merge(reports[0..jobs], opt.words.len > 0) catch |err| {
        std.debug.print("FAIL kernel-contract: the shards do not add up: {t}\n", .{err});
        return 1;
    };
    if (shard.printsGradientPass(.parent, opt.words.len > 0)) std.debug.print("{s}", .{@import("contract_back_suites.zig").passed});
    std.debug.print("{f}\n", .{sum});
    if (sum.problems() == 0) return 0;
    std.debug.print("FAIL kernel-contract: {d} gate failures, {d} controls missed, {d} limit problems\n", .{ sum.tally.failed, sum.tally.missed, sum.limits });
    return 1;
}
