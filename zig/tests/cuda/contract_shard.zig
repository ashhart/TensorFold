//! kernel-contract's opt-in sharded run: its options, a shard's share, the tally a shard reports and their merge.
const std = @import("std");
const gates = @import("core").contract.gates;

/// Most child processes `--jobs` starts.
pub const max_jobs: u32 = 16;
/// Memory one process is planned for (the host's by whether words filter the run); `--jobs` refuses what won't fit.
pub const host_mib_full: u64 = 28 * 1024;
pub const host_mib_filtered: u64 = 8 * 1024;
pub const device_mib_per_process: u64 = 7 * 1024;

/// One slice of a run: the items whose index modulo `count` is `index`.
pub const Shard = struct {
    index: u32 = 0,
    count: u32 = 1,

    pub fn owns(s: Shard, i: usize) bool {
        return i % s.count == s.index;
    }

    /// `K/N` with K < N, N > 0, both plain decimal digits.
    pub fn parse(text: []const u8) error{InvalidShard}!Shard {
        const slash = std.mem.indexOfScalar(u8, text, '/') orelse return error.InvalidShard;
        const index = number(u32, text[0..slash]) orelse return error.InvalidShard;
        const count = number(u32, text[slash + 1 ..]) orelse return error.InvalidShard;
        if (count == 0 or index >= count) return error.InvalidShard;
        return .{ .index = index, .count = count };
    }
};

fn number(comptime T: type, text: []const u8) ?T {
    if (text.len == 0) return null;
    for (text) |c| if (c < '0' or c > '9') return null;
    return std.fmt.parseUnsigned(T, text, 10) catch null;
}

pub const Options = struct {
    /// Name filter: the kernels, controls and launches whose name holds one of these.
    words: []const []const u8,
    shard: ?Shard = null,
    /// Child processes to run (`--jobs`); null runs in this process.
    jobs: ?u32 = null,
    /// The visible GPU a shard runs on (`--device`).
    device: u8 = 0,
    /// The `--jobs` process that started this shard (`--parent`).
    parent: ?u31 = null,

    pub fn effectiveShard(o: Options) Shard {
        return o.shard orelse .{};
    }

    pub fn deinit(o: Options, a: std.mem.Allocator) void {
        a.free(o.words);
    }
};

pub const ParseError = error{ InvalidShard, InvalidJobs, InvalidDevice, DuplicateOption, ShardWithJobs, DeviceWithoutShard, ParentWithoutShard, InvalidParent, UnknownOption, MissingValue } || std.mem.Allocator.Error;

/// The option and value a refused command line names.
pub const Culprit = struct {
    option: []const u8 = "",
    value: []const u8 = "",

    pub fn format(c: Culprit, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll(c.option);
        if (c.value.len > 0) try w.print(" {s}", .{c.value});
    }
};

/// What the refused option needs, for the line that names it.
pub fn hint(err: ParseError) []const u8 {
    return switch (err) {
        error.InvalidShard => "K/N with K below N",
        error.InvalidJobs => std.fmt.comptimePrint("1 to {d}", .{max_jobs}),
        error.InvalidDevice => "a GPU number, 0 to 255",
        error.DuplicateOption => "at most once",
        error.ShardWithJobs => "--jobs makes the shards",
        error.DeviceWithoutShard => "--device goes with --shard",
        error.InvalidParent => "a process id",
        error.ParentWithoutShard => "--parent goes with --shard",
        error.UnknownOption => "options are --jobs, --shard, --device, --parent and --help",
        error.MissingValue => "the option needs a value",
        error.OutOfMemory => "out of memory",
    };
}

/// `kernel-contract`'s arguments: name words and, each once at most, `--jobs`, `--shard`, `--device` and `--parent`.
pub fn parse(a: std.mem.Allocator, args: []const [:0]const u8) ParseError!Options {
    var culprit: Culprit = .{};
    return parseNamed(a, args, &culprit);
}

/// `parse`, naming in `culprit` the option and value it refuses.
pub fn parseNamed(a: std.mem.Allocator, args: []const [:0]const u8, culprit: *Culprit) ParseError!Options {
    var words: std.ArrayList([]const u8) = .empty;
    errdefer words.deinit(a);
    var o: Options = .{ .words = &.{} };
    var device_text: ?[]const u8 = null;
    var parent_text: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg: []const u8 = args[i];
        if (!std.mem.startsWith(u8, arg, "--")) {
            try words.append(a, arg);
            continue;
        }
        culprit.* = .{ .option = arg };
        const is_jobs = std.mem.eql(u8, arg, "--jobs");
        const is_shard = std.mem.eql(u8, arg, "--shard");
        const is_device = std.mem.eql(u8, arg, "--device");
        const is_parent = std.mem.eql(u8, arg, "--parent");
        if (!is_jobs and !is_shard and !is_device and !is_parent) return error.UnknownOption;
        i += 1;
        if (i == args.len) return error.MissingValue;
        culprit.value = args[i];
        if (is_jobs) {
            if (o.jobs != null) return error.DuplicateOption;
            const n = number(u32, args[i]) orelse return error.InvalidJobs;
            if (n == 0 or n > max_jobs) return error.InvalidJobs;
            o.jobs = n;
        } else if (is_shard) {
            if (o.shard != null) return error.DuplicateOption;
            o.shard = try Shard.parse(args[i]);
        } else if (is_parent) {
            if (parent_text != null) return error.DuplicateOption;
            o.parent = number(u31, args[i]) orelse return error.InvalidParent;
            parent_text = args[i];
        } else {
            if (device_text != null) return error.DuplicateOption;
            o.device = number(u8, args[i]) orelse return error.InvalidDevice;
            device_text = args[i];
        }
    }
    if (o.jobs != null and o.shard != null) {
        culprit.* = .{ .option = "--shard", .value = "with --jobs" };
        return error.ShardWithJobs;
    }
    if (device_text) |text| if (o.shard == null) {
        culprit.* = .{ .option = "--device", .value = text };
        return error.DeviceWithoutShard;
    };
    if (parent_text) |text| if (o.shard == null) {
        culprit.* = .{ .option = "--parent", .value = text };
        return error.ParentWithoutShard;
    };
    o.words = try words.toOwnedSlice(a);
    return o;
}

/// Keeps the items named by `only`'s words (empty: all), then `sh`'s share of those; returns how many the words kept.
pub fn keep(comptime T: type, items: *std.ArrayList(T), only: []const []const u8, sh: Shard) usize {
    var kept: usize = 0;
    for (items.items) |item| {
        const named = only.len == 0 or for (only) |w| {
            if (std.mem.indexOf(u8, item.spec.name, w) != null) break true;
        } else false;
        if (!named) continue;
        items.items[kept] = item;
        kept += 1;
    }
    var mine: usize = 0;
    for (items.items[0..kept], 0..) |item, i| {
        if (!sh.owns(i)) continue;
        items.items[mine] = item;
        mine += 1;
    }
    items.shrinkRetainingCapacity(mine);
    return kept;
}

/// How many of a list's items a shard ran and how many the list holds.
pub const Coverage = struct { ran: u64 = 0, of: u64 = 0 };

/// What one shard did: the tally of its kernels and controls, its limit problems and what it ran of each list.
pub const Report = struct {
    shard: Shard,
    /// Kernels and controls named by the words, before sharding.
    listed: u64,
    tally: gates.Tally,
    limits: u64,
    suites: Coverage,
    records: Coverage,
    gates_ms: u64,
    limits_ms: u64,

    pub fn problems(r: Report) u64 {
        return r.tally.problems() + r.limits;
    }

    const names = [_][]const u8{ "listed", "kernels", "passed", "failed", "known", "controls", "caught", "missed", "limits", "suites", "of_suites", "records", "of_records", "gates_ms", "limits_ms" };

    fn values(r: Report) [names.len]u64 {
        const t = r.tally;
        return .{ r.listed, t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, r.limits, r.suites.ran, r.suites.of, r.records.ran, r.records.of, r.gates_ms, r.limits_ms };
    }

    /// The machine-readable line a shard prints and the parent parses; the human SUMMARY is never parsed.
    pub fn format(r: Report, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("TALLY shard={d}/{d}", .{ r.shard.index, r.shard.count });
        inline for (names, r.values()) |name, v| try w.print(" {s}={d}", .{ name, v });
    }

    pub fn parseLine(line: []const u8) error{BadTally}!Report {
        var it = std.mem.splitScalar(u8, line, ' ');
        if (!std.mem.eql(u8, it.next().?, "TALLY")) return error.BadTally;
        const head = it.next() orelse return error.BadTally;
        if (!std.mem.startsWith(u8, head, "shard=")) return error.BadTally;
        const shard = Shard.parse(head["shard=".len..]) catch return error.BadTally;
        var v: [names.len]u64 = undefined;
        inline for (names, &v) |name, *x| {
            const field = it.next() orelse return error.BadTally;
            if (!std.mem.startsWith(u8, field, name ++ "=")) return error.BadTally;
            x.* = number(u64, field[name.len + 1 ..]) orelse return error.BadTally;
        }
        if (it.next() != null) return error.BadTally;
        return .{
            .shard = shard,
            .listed = v[0],
            .tally = .{ .kernels = v[1], .passed = v[2], .failed = v[3], .known = v[4], .controls = v[5], .caught = v[6], .missed = v[7] },
            .limits = v[8],
            .suites = .{ .ran = v[9], .of = v[10] },
            .records = .{ .ran = v[11], .of = v[12] },
            .gates_ms = v[13],
            .limits_ms = v[14],
        };
    }
};

/// The run's totals as its one SUMMARY line states them.
pub const Summary = struct {
    filtered: bool,
    tally: gates.Tally,
    limits: u64,
    gates_ms: u64,
    limits_ms: u64,

    pub fn problems(s: Summary) u64 {
        return s.tally.problems() + s.limits;
    }

    pub fn format(s: Summary, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const t = s.tally;
        try w.print("{s}SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems; gates {d} ms, limits {d} ms", .{
            if (s.filtered) "(filtered) " else "", t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed, s.limits, s.gates_ms, s.limits_ms,
        });
    }
};

pub const MergeError = error{ NoShards, ShardCount, ShardRepeated, ShardMissing, ListDisagrees, KernelsNotOnce, SuitesNotOnce, RecordsNotOnce };

/// One summary from every shard's report: counts add, times are the slowest's, and each list must have run once.
pub fn merge(reports: []const Report, filtered: bool) MergeError!Summary {
    if (reports.len == 0) return error.NoShards;
    const first = reports[0];
    if (first.shard.count != reports.len) return error.ShardMissing;
    var sum: Summary = .{ .filtered = filtered, .tally = .{}, .limits = 0, .gates_ms = 0, .limits_ms = 0 };
    var ran: u64 = 0;
    var suites: u64 = 0;
    var records: u64 = 0;
    for (reports, 0..) |r, i| {
        if (r.shard.count != first.shard.count) return error.ShardCount;
        for (reports[0..i]) |before| if (before.shard.index == r.shard.index) return error.ShardRepeated;
        if (r.listed != first.listed or r.suites.of != first.suites.of or r.records.of != first.records.of) return error.ListDisagrees;
        sum.tally.kernels += r.tally.kernels;
        sum.tally.passed += r.tally.passed;
        sum.tally.failed += r.tally.failed;
        sum.tally.known += r.tally.known;
        sum.tally.controls += r.tally.controls;
        sum.tally.caught += r.tally.caught;
        sum.tally.missed += r.tally.missed;
        sum.limits += r.limits;
        sum.gates_ms = @max(sum.gates_ms, r.gates_ms);
        sum.limits_ms = @max(sum.limits_ms, r.limits_ms);
        ran += r.tally.kernels + r.tally.controls;
        suites += r.suites.ran;
        records += r.records.ran;
    }
    if (ran != first.listed) return error.KernelsNotOnce;
    if (suites != first.suites.of) return error.SuitesNotOnce;
    if (records != first.records.of) return error.RecordsNotOnce;
    return sum;
}

/// A shard's outcome is accepted only with its TALLY line, and an exit of 0 or an exit of 1 that its problems explain.
pub fn accepted(exit_code: ?u8, report: ?Report) bool {
    const r = report orelse return false;
    const code = exit_code orelse return false;
    return code == 0 or (code == 1 and r.problems() > 0);
}

pub const Refusal = error{ NoGpu, DeviceMemory, HostMemory };

/// The machine `--jobs` would run on: its GPUs, the smallest's total memory, the host's available memory.
pub const Machine = struct {
    gpus: u32,
    gpu_mib: u64,
    host_mib: u64,
    /// A GPU shares the host's memory, so a process's host and GPU plans draw on the same pool.
    integrated: bool = false,
};

/// Whether `jobs` processes of a full or `filtered` run fit `m`: `ceil(jobs / gpus)` on a GPU, all on the host.
pub fn afford(jobs: u32, filtered: bool, m: Machine) Refusal!void {
    if (m.gpus == 0) return error.NoGpu;
    const host = if (filtered) host_mib_filtered else host_mib_full;
    if (m.integrated) {
        if (jobs * (host + device_mib_per_process) > m.host_mib) return error.HostMemory;
        return;
    }
    const per_gpu = (jobs + m.gpus - 1) / m.gpus;
    if (per_gpu * device_mib_per_process > m.gpu_mib) return error.DeviceMemory;
    if (jobs * host > m.host_mib) return error.HostMemory;
}

/// What a registration reads of a GPU to choose its kernels' configurations.
pub const Gpu = struct {
    sm: u32,
    multiprocessors: u32,
    name: [96]u8 = @splat(0),

    pub fn named(sm: u32, multiprocessors: u32, name: []const u8) Gpu {
        var g: Gpu = .{ .sm = sm, .multiprocessors = multiprocessors };
        @memcpy(g.name[0..@min(name.len, g.name.len - 1)], name[0..@min(name.len, g.name.len - 1)]);
        return g;
    }

    pub fn nameText(g: *const Gpu) []const u8 {
        return std.mem.sliceTo(&g.name, 0);
    }
};

pub const Field = enum { compute_capability, multiprocessors, name };

pub const Mix = struct { gpu: usize, field: Field };

/// The first GPU that differs from GPU 0 in compute capability, multiprocessor count or name, and in which.
pub fn firstMixed(gpus: []const Gpu) ?Mix {
    for (gpus[@min(1, gpus.len)..], 1..) |g, i| {
        if (g.sm != gpus[0].sm) return .{ .gpu = i, .field = .compute_capability };
        if (g.multiprocessors != gpus[0].multiprocessors) return .{ .gpu = i, .field = .multiprocessors };
        if (!std.mem.eql(u8, g.nameText(), gpus[0].nameText())) return .{ .gpu = i, .field = .name };
    }
    return null;
}

/// The pair `firstMixed` found, each GPU with the value that differs.
pub fn writeMix(w: *std.Io.Writer, gpus: []const Gpu, m: Mix) std.Io.Writer.Error!void {
    const first = &gpus[0];
    const other = &gpus[m.gpu];
    switch (m.field) {
        .compute_capability => try w.print("GPU 0 is sm_{d} and GPU {d} is sm_{d}", .{ first.sm, m.gpu, other.sm }),
        .multiprocessors => try w.print("GPU 0 has {d} multiprocessors and GPU {d} has {d}", .{ first.multiprocessors, m.gpu, other.multiprocessors }),
        .name => try w.print("GPU 0 is a {s} and GPU {d} is a {s}", .{ first.nameText(), m.gpu, other.nameText() }),
    }
}

/// Who prints the gradient suites' PASS line: the serial run or the `--jobs` parent, never a shard; not when filtered.
pub const Role = enum { serial, parent, shard };

pub fn printsGradientPass(role: Role, filtered: bool) bool {
    return !filtered and role != .shard;
}

/// Hands `sink.line(text, cut)` each line of `r`; one longer than the buffer comes cut, the lines after it whole.
pub fn eachLine(r: *std.Io.Reader, sink: anytype) !void {
    while (true) {
        const line = r.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                try sink.line(r.buffered(), true);
                r.tossBuffered();
                _ = r.discardDelimiterInclusive('\n') catch |e| switch (e) {
                    error.EndOfStream => return,
                    else => return e,
                };
                continue;
            },
            else => |e| return e,
        } orelse return;
        try sink.line(line, false);
    }
}

/// `/proc/meminfo`'s MemAvailable in MiB.
pub fn memAvailableMib(meminfo: []const u8) ?u64 {
    var lines = std.mem.tokenizeScalar(u8, meminfo, '\n');
    while (lines.next()) |line| {
        const key = "MemAvailable:";
        if (!std.mem.startsWith(u8, line, key)) continue;
        var words = std.mem.tokenizeScalar(u8, line[key.len..], ' ');
        const kib = number(u64, words.next() orelse return null) orelse return null;
        if (!std.mem.eql(u8, words.next() orelse return null, "kB")) return null;
        return kib / 1024;
    }
    return null;
}

pub const help = std.fmt.comptimePrint(
    \\usage: tf-cuda-test kernel-contract [WORD...] [--jobs N | --shard K/N [--device D]]
    \\  (no option)   the full serial run in this process
    \\  WORD...       only the kernels, controls and launches whose name holds a WORD (no suites, no offline audit)
    \\  --jobs N      opt-in: N child processes (1 to {d}), one shard each, GPUs round-robin, one merged SUMMARY;
    \\                a process is planned for {d} GiB of host memory ({d} GiB with WORDs) and {d} GiB of GPU memory;
    \\                refused where N processes do not fit the host's available memory (Linux: /proc/meminfo) or the GPU's
    \\                total memory, or where the GPUs differ in compute capability
    \\  --shard K/N   opt-in: share K of N of the kernels, suites and launches, then a TALLY line in place of a SUMMARY
    \\  --device D    the visible GPU a --shard process runs on
    \\  --parent PID  the --jobs process that started a --shard process; the shard stops with it
    \\Words and --jobs or --shard compose: the words filter first, then the shard slices what is left.
    \\
, .{ max_jobs, host_mib_full / 1024, host_mib_filtered / 1024, device_mib_per_process / 1024 });
