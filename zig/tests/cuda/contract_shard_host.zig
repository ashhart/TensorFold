//! kernel-contract's sharded run without a GPU: the partition, the options, the tally line and the merge.
const std = @import("std");
const t = std.testing;
const shard = @import("contract_shard.zig");
const gates = @import("core").contract.gates;

fn args(comptime words: []const [:0]const u8) []const [:0]const u8 {
    return words;
}

test "shard partition: every index belongs to exactly one shard, in shares within one of each other" {
    for (1..70) |count| for ([_]usize{ 0, 1, 2, 7, 63, 64, 65, 1000, 4096 }) |n| {
        var owned: [70]usize = @splat(0);
        for (0..n) |i| {
            var owners: usize = 0;
            for (0..count) |k| {
                const s: shard.Shard = .{ .index = @intCast(k), .count = @intCast(count) };
                if (s.owns(i)) {
                    owners += 1;
                    owned[k] += 1;
                }
            }
            try t.expectEqual(@as(usize, 1), owners);
        }
        for (owned[0..count]) |share| try t.expect(share == n / count or share == n / count + 1);
    };
}

const Named = struct { spec: struct { name: []const u8 } };

fn tally(list: []const Named) gates.Tally {
    var sum: gates.Tally = .{};
    for (list) |k| {
        if (std.mem.startsWith(u8, k.spec.name, "control")) {
            sum.controls += 1;
            sum.caught += 1;
        } else {
            sum.kernels += 1;
            if (k.spec.name.len % 7 == 0) sum.failed += 1 else sum.passed += 1;
        }
    }
    return sum;
}

test "shard keep: the words filter first, the shards run each named kernel and control once, and their tallies add up" {
    var names: [200][]const u8 = undefined;
    var buf: [200][16]u8 = undefined;
    for (&names, &buf, 0..) |*name, *b, i| name.* = std.fmt.bufPrint(b, "{s} {d}", .{ if (i % 5 == 0) "control" else if (i % 3 == 0) "lane" else "norm", i }) catch unreachable;
    for ([_][]const []const u8{ &.{}, &.{"lane"}, &.{ "lane", "control" }, &.{"absent"} }) |words| {
        var serial: std.ArrayList(Named) = .empty;
        defer serial.deinit(t.allocator);
        for (names) |name| try serial.append(t.allocator, .{ .spec = .{ .name = name } });
        const listed = shard.keep(Named, &serial, words, .{});
        try t.expectEqual(serial.items.len, listed);
        for ([_]u32{ 1, 2, 3, 4 }) |count| {
            var reports: [4]shard.Report = undefined;
            var seen: usize = 0;
            for (0..count) |k| {
                var mine: std.ArrayList(Named) = .empty;
                defer mine.deinit(t.allocator);
                for (names) |name| try mine.append(t.allocator, .{ .spec = .{ .name = name } });
                const kept = shard.keep(Named, &mine, words, .{ .index = @intCast(k), .count = count });
                try t.expectEqual(listed, kept);
                if (words.len > 0) for (mine.items) |item| {
                    var named = false;
                    for (words) |w| named = named or std.mem.indexOf(u8, item.spec.name, w) != null;
                    try t.expect(named);
                };
                seen += mine.items.len;
                reports[k] = .{ .shard = .{ .index = @intCast(k), .count = count }, .listed = kept, .tally = tally(mine.items), .limits = 0, .suites = .{}, .records = .{}, .gates_ms = 1, .limits_ms = 1 };
            }
            try t.expectEqual(listed, seen);
            const merged = try shard.merge(reports[0..count], words.len > 0);
            const want = tally(serial.items);
            try t.expectEqual(want, merged.tally);
        }
    }
}

test "shard options: words, flags and their refusals by name" {
    const a = t.allocator;
    const ok = try shard.parse(a, args(&.{ "lane", "--shard", "1/4", "--device", "1", "norm" }));
    defer ok.deinit(a);
    try t.expectEqual(@as(usize, 2), ok.words.len);
    try t.expectEqualStrings("norm", ok.words[1]);
    try t.expectEqual(shard.Shard{ .index = 1, .count = 4 }, ok.shard.?);
    try t.expectEqual(@as(u8, 1), ok.device);
    const child = try shard.parse(a, args(&.{ "--shard", "0/2", "--parent", "4242" }));
    defer child.deinit(a);
    try t.expectEqual(@as(?u31, 4242), child.parent);
    const jobs = try shard.parse(a, args(&.{ "--jobs", "16" }));
    defer jobs.deinit(a);
    try t.expectEqual(@as(?u32, 16), jobs.jobs);
    try t.expect(jobs.shard == null);
    const bad = [_]struct { []const [:0]const u8, anyerror }{
        .{ args(&.{ "--shard", "4/4" }), error.InvalidShard },
        .{ args(&.{ "--shard", "0/0" }), error.InvalidShard },
        .{ args(&.{ "--shard", "1" }), error.InvalidShard },
        .{ args(&.{ "--shard", "-1/2" }), error.InvalidShard },
        .{ args(&.{ "--shard", "1/+2" }), error.InvalidShard },
        .{ args(&.{ "--shard", "a/b" }), error.InvalidShard },
        .{ args(&.{ "--jobs", "0" }), error.InvalidJobs },
        .{ args(&.{ "--jobs", "17" }), error.InvalidJobs },
        .{ args(&.{ "--jobs", "two" }), error.InvalidJobs },
        .{ args(&.{ "--jobs", "1_0" }), error.InvalidJobs },
        .{ args(&.{ "--shard", "0/2", "--device", "x" }), error.InvalidDevice },
        .{ args(&.{ "--shard", "0/2", "--device", "256" }), error.InvalidDevice },
        .{ args(&.{ "--parent", "4242" }), error.ParentWithoutShard },
        .{ args(&.{ "--jobs", "2", "--parent", "4242" }), error.ParentWithoutShard },
        .{ args(&.{ "--shard", "0/2", "--parent", "-1" }), error.InvalidParent },
        .{ args(&.{ "--shard", "0/2", "--parent", "4242", "--parent", "4243" }), error.DuplicateOption },
        .{ args(&.{ "--device", "1" }), error.DeviceWithoutShard },
        .{ args(&.{ "--jobs", "2", "--device", "0" }), error.DeviceWithoutShard },
        .{ args(&.{ "--shard", "0/2", "--device", "0", "--device", "1" }), error.DuplicateOption },
        .{ args(&.{ "--jobs", "2", "--shard", "0/2" }), error.ShardWithJobs },
        .{ args(&.{ "--jobs", "2", "--jobs", "2" }), error.DuplicateOption },
        .{ args(&.{ "--shard", "0/2", "--shard", "1/2" }), error.DuplicateOption },
        .{ args(&.{"--jobs"}), error.MissingValue },
        .{ args(&.{"--parallel"}), error.UnknownOption },
    };
    for (bad) |case| try t.expectError(case[1], shard.parse(a, case[0]));
}

test "shard options: a refusal names the option and value, and what it needs" {
    const a = t.allocator;
    const cases = [_]struct { []const [:0]const u8, anyerror, []const u8, []const u8 }{
        .{ args(&.{ "norm", "--jobs", "17" }), error.InvalidJobs, "--jobs 17", "1 to 16" },
        .{ args(&.{ "--shard", "3/3" }), error.InvalidShard, "--shard 3/3", "K/N with K below N" },
        .{ args(&.{ "--parallel", "2" }), error.UnknownOption, "--parallel", "options are --jobs, --shard, --device, --parent and --help" },
        .{ args(&.{ "--device", "1" }), error.DeviceWithoutShard, "--device 1", "--device goes with --shard" },
        .{ args(&.{ "--jobs", "2", "--shard", "0/2" }), error.ShardWithJobs, "--shard with --jobs", "--jobs makes the shards" },
        .{ args(&.{"--shard"}), error.MissingValue, "--shard", "the option needs a value" },
        .{ args(&.{ "--parent", "7" }), error.ParentWithoutShard, "--parent 7", "--parent goes with --shard" },
    };
    for (cases) |case| {
        var culprit: shard.Culprit = .{};
        try t.expectError(case[1], shard.parseNamed(a, case[0], &culprit));
        var buf: [128]u8 = undefined;
        try t.expectEqualStrings(case[2], try std.fmt.bufPrint(&buf, "{f}", .{culprit}));
        try t.expectEqualStrings(case[3], shard.hint(@as(shard.ParseError, @errorCast(case[1]))));
    }
}

const sample: shard.Report = .{
    .shard = .{ .index = 2, .count = 4 },
    .listed = 1000,
    .tally = .{ .kernels = 300, .passed = 280, .failed = 10, .known = 10, .controls = 100, .caught = 90, .missed = 10 },
    .limits = 4,
    .suites = .{ .ran = 3, .of = 12 },
    .records = .{ .ran = 200, .of = 1000 },
    .gates_ms = 1_000_000,
    .limits_ms = 200_000,
};

test "shard tally line: formats and parses back to the same report, and refuses other lines" {
    var buf: [512]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{f}", .{sample});
    try t.expectEqualStrings("TALLY shard=2/4 listed=1000 kernels=300 passed=280 failed=10 known=10 controls=100 caught=90 missed=10 limits=4 suites=3 of_suites=12 records=200 of_records=1000 gates_ms=1000000 limits_ms=200000", line);
    try t.expectEqual(sample, try shard.Report.parseLine(line));
    var damaged: [512]u8 = undefined;
    var extended: [512]u8 = undefined;
    const bad = [_][]const u8{
        "",
        "SUMMARY 1 kernels",
        "TALLY",
        "TALLY shard=2/4",
        "TALLY shard=4/4 listed=1",
        try std.fmt.bufPrint(&damaged, "{s}x", .{line[0 .. line.len - 1]}),
        try std.fmt.bufPrint(&extended, "{s} extra=1", .{line}),
        "TALLY shard=2/4 kernels=300 listed=1000",
    };
    for (bad) |text| try t.expectError(error.BadTally, shard.Report.parseLine(text));
}

fn part(index: u32, count: u32, listed: u64, kernels: usize, controls: usize) shard.Report {
    var r = sample;
    r.shard = .{ .index = index, .count = count };
    r.listed = listed;
    r.tally = .{ .kernels = kernels, .passed = kernels, .controls = controls, .caught = controls };
    r.limits = 0;
    r.suites = .{ .ran = 1, .of = 2 };
    r.records = .{ .ran = 5, .of = 10 };
    return r;
}

test "shard merge: counts add, times are the slowest shard's, and a list not run exactly once is refused" {
    const merged = try shard.merge(&.{ part(0, 2, 10, 3, 2), part(1, 2, 10, 4, 1) }, false);
    try t.expectEqual(@as(usize, 7), merged.tally.kernels);
    try t.expectEqual(@as(usize, 7), merged.tally.passed);
    try t.expectEqual(@as(usize, 3), merged.tally.controls);
    try t.expectEqual(@as(u64, 1_000_000), merged.gates_ms);
    try t.expectEqual(@as(u64, 0), merged.problems());
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings(
        "SUMMARY 7 kernels: 7 pass gates 1-4, 0 fail, 0 known findings; 3 controls: 3 red as planted, 0 missed; 0 limit problems; gates 1000000 ms, limits 200000 ms",
        try std.fmt.bufPrint(&buf, "{f}", .{merged}),
    );
    try t.expectError(error.NoShards, shard.merge(&.{}, false));
    try t.expectError(error.ShardMissing, shard.merge(&.{part(0, 2, 10, 5, 5)}, false));
    try t.expectError(error.ShardRepeated, shard.merge(&.{ part(0, 2, 10, 5, 0), part(0, 2, 10, 5, 0) }, false));
    try t.expectError(error.ShardCount, shard.merge(&.{ part(0, 2, 10, 5, 0), part(1, 3, 10, 5, 0) }, false));
    try t.expectError(error.ListDisagrees, shard.merge(&.{ part(0, 2, 10, 5, 0), part(1, 2, 11, 5, 0) }, false));
    try t.expectError(error.KernelsNotOnce, shard.merge(&.{ part(0, 2, 10, 5, 0), part(1, 2, 10, 4, 0) }, false));
    var twice = part(1, 2, 10, 5, 0);
    twice.suites.ran = 2;
    try t.expectError(error.SuitesNotOnce, shard.merge(&.{ part(0, 2, 10, 5, 0), twice }, false));
    var records = part(1, 2, 10, 5, 0);
    records.records.ran = 4;
    try t.expectError(error.RecordsNotOnce, shard.merge(&.{ part(0, 2, 10, 5, 0), records }, false));
}

test "shard merge: a failed gate or a missed control in any shard fails the run" {
    var failing = part(1, 2, 10, 5, 0);
    failing.tally.failed = 1;
    failing.tally.passed = 4;
    try t.expectEqual(@as(u64, 1), (try shard.merge(&.{ part(0, 2, 10, 5, 0), failing }, false)).problems());
    var missed = part(1, 2, 10, 5, 0);
    missed.tally.missed = 1;
    try t.expectEqual(@as(u64, 1), (try shard.merge(&.{ part(0, 2, 10, 5, 0), missed }, false)).problems());
    var limit = part(1, 2, 10, 5, 0);
    limit.limits = 2;
    try t.expectEqual(@as(u64, 2), (try shard.merge(&.{ part(0, 2, 10, 5, 0), limit }, false)).problems());
}

test "shard accepted: a crash, a missing tally or an unexplained exit fails the shard" {
    const clean = part(0, 2, 10, 5, 0);
    var failing = clean;
    failing.tally.failed = 1;
    try t.expect(shard.accepted(0, clean));
    try t.expect(shard.accepted(1, failing));
    try t.expect(!shard.accepted(1, clean));
    try t.expect(!shard.accepted(0, null));
    try t.expect(!shard.accepted(1, null));
    try t.expect(!shard.accepted(null, clean));
    try t.expect(!shard.accepted(137, failing));
}

const gib: u64 = 1024;

fn machine(gpus: u32, gpu_gib: u64, host_gib: u64, integrated: bool) shard.Machine {
    return .{ .gpus = gpus, .gpu_mib = gpu_gib * gib, .host_mib = host_gib * gib, .integrated = integrated };
}

test "shard afford: a GPU or the host that cannot hold the processes refuses them" {
    try shard.afford(4, false, machine(2, 16, 200, false));
    try shard.afford(2, false, machine(1, 16, 200, false));
    try shard.afford(1, false, machine(1, 8, 40, false));
    try t.expectError(error.DeviceMemory, shard.afford(3, false, machine(1, 16, 200, false)));
    try t.expectError(error.DeviceMemory, shard.afford(1, false, machine(1, 4, 200, false)));
    try t.expectError(error.HostMemory, shard.afford(4, false, machine(2, 16, 100, false)));
    try t.expectError(error.NoGpu, shard.afford(1, false, machine(0, 16, 200, false)));
}

test "shard afford: a filtered run plans less host memory, and processes per GPU round up" {
    try shard.afford(16, true, machine(2, 56, 128, false));
    try t.expectError(error.HostMemory, shard.afford(16, true, machine(2, 56, 127, false)));
    try t.expectError(error.HostMemory, shard.afford(5, false, machine(5, 16, 100, false)));
    try shard.afford(4, true, machine(4, 16, 100, false));
    try t.expectError(error.DeviceMemory, shard.afford(16, true, machine(2, 48, 200, false)));
    try t.expectError(error.DeviceMemory, shard.afford(5, true, machine(2, 20, 200, false)));
    try shard.afford(5, true, machine(2, 21, 200, false));
}

test "shard afford: an integrated GPU draws on the host's memory, so both plans add and no GPU total is checked" {
    // 4 full processes plan 4 x (28 + 7) GiB of the one pool; the GPU's own total is not a limit
    try shard.afford(4, false, machine(1, 1, 140, true));
    try t.expectError(error.HostMemory, shard.afford(4, false, machine(1, 1, 139, true)));
    try shard.afford(8, true, machine(1, 1, 120, true));
    try t.expectError(error.HostMemory, shard.afford(8, true, machine(1, 1, 119, true)));
    try t.expectError(error.NoGpu, shard.afford(1, true, machine(0, 1, 120, true)));
}

fn gpuList(comptime specs: []const struct { u32, u32, []const u8 }) [specs.len]shard.Gpu {
    var out: [specs.len]shard.Gpu = undefined;
    for (specs, &out) |spec, *g| g.* = .named(spec[0], spec[1], spec[2]);
    return out;
}

fn mixText(list: []const shard.Gpu) ![]const u8 {
    const mix = shard.firstMixed(list) orelse return "same";
    var buf: [200]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try shard.writeMix(&w, list, mix);
    return t.allocator.dupe(u8, w.buffered());
}

test "shard mixed GPUs: the first that differs from GPU 0 in capability, multiprocessors or name, with both values" {
    try t.expectEqual(@as(?shard.Mix, null), shard.firstMixed(&.{}));
    const same = gpuList(&.{ .{ 120, 36, "RTX 5060 Ti" }, .{ 120, 36, "RTX 5060 Ti" } });
    try t.expectEqual(@as(?shard.Mix, null), shard.firstMixed(&same));
    const sm = gpuList(&.{ .{ 120, 36, "A" }, .{ 120, 36, "A" }, .{ 121, 36, "A" } });
    const sm_text = try mixText(&sm);
    defer t.allocator.free(sm_text);
    try t.expectEqualStrings("GPU 0 is sm_120 and GPU 2 is sm_121", sm_text);
    const count = gpuList(&.{ .{ 120, 36, "A" }, .{ 120, 70, "A" } });
    try t.expectEqual(shard.Mix{ .gpu = 1, .field = .multiprocessors }, shard.firstMixed(&count).?);
    const count_text = try mixText(&count);
    defer t.allocator.free(count_text);
    try t.expectEqualStrings("GPU 0 has 36 multiprocessors and GPU 1 has 70", count_text);
    const name = gpuList(&.{ .{ 120, 36, "RTX 5060 Ti" }, .{ 120, 36, "RTX 5070" } });
    try t.expectEqual(shard.Mix{ .gpu = 1, .field = .name }, shard.firstMixed(&name).?);
    const name_text = try mixText(&name);
    defer t.allocator.free(name_text);
    try t.expectEqualStrings("GPU 0 is a RTX 5060 Ti and GPU 1 is a RTX 5070", name_text);
}

test "shard meminfo: MemAvailable in MiB, and nothing from other text" {
    const text = "MemTotal:       263640884 kB\nMemFree:         1048576 kB\nMemAvailable:   209715200 kB\nBuffers:            1 kB\n";
    try t.expectEqual(@as(?u64, 204800), shard.memAvailableMib(text));
    try t.expectEqual(@as(?u64, null), shard.memAvailableMib("MemTotal: 1 kB\n"));
    try t.expectEqual(@as(?u64, null), shard.memAvailableMib("MemAvailable: lots kB\n"));
    try t.expectEqual(@as(?u64, null), shard.memAvailableMib("MemAvailable: 5 MB\n"));
}

test "shard help: states the cap, both host plans, the GPU plan and every option" {
    for ([_][]const u8{ "1 to 16", "28 GiB", "8 GiB with WORDs", "7 GiB", "--jobs N", "--shard K/N", "--device D", "/proc/meminfo" }) |needle|
        try t.expect(std.mem.indexOf(u8, shard.help, needle) != null);
}

test "shard pass line: the process that prints the gradient PASS line is one, and none when filtered" {
    const runs = [_][]const shard.Role{ &.{.serial}, &.{ .parent, .shard }, &.{ .parent, .shard, .shard, .shard } };
    for (runs) |roles| for ([_]bool{ false, true }) |filtered| {
        var printers: usize = 0;
        for (roles) |role| printers += @intFromBool(shard.printsGradientPass(role, filtered));
        try t.expectEqual(@as(usize, if (filtered) 0 else 1), printers);
    };
}

const Collect = struct {
    lines: *std.ArrayList([]u8),
    cuts: *std.ArrayList(bool),

    pub fn line(c: Collect, text: []const u8, cut: bool) !void {
        try c.lines.append(t.allocator, try t.allocator.dupe(u8, text));
        try c.cuts.append(t.allocator, cut);
    }
};

test "shard lines: a line longer than the buffer is cut and the lines after it still arrive whole" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const line = "TALLY shard=0/1 listed=1 kernels=1 passed=1";
    const long: [100]u8 = @splat('x');
    const mid: [60]u8 = @splat('y');
    const text = try std.fmt.allocPrint(t.allocator, "first\n{s}\n{s}\n{s}\nlast", .{ long, line, mid });
    defer t.allocator.free(text);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "log", .data = text });
    const file = try tmp.dir.openFile(t.io, "log", .{});
    defer file.close(t.io);
    var buf: [48]u8 = undefined;
    var reader = file.readerStreaming(t.io, &buf);
    var lines: std.ArrayList([]u8) = .empty;
    var cuts: std.ArrayList(bool) = .empty;
    defer {
        for (lines.items) |l| t.allocator.free(l);
        lines.deinit(t.allocator);
        cuts.deinit(t.allocator);
    }
    try shard.eachLine(&reader.interface, Collect{ .lines = &lines, .cuts = &cuts });
    try t.expectEqual(@as(usize, 5), lines.items.len);
    try t.expectEqualStrings("first", lines.items[0]);
    try t.expectEqual(@as(usize, 48), lines.items[1].len);
    try t.expect(cuts.items[1]);
    try t.expectEqualStrings(line, lines.items[2]);
    try t.expect(!cuts.items[2]);
    try t.expect(cuts.items[3]);
    try t.expectEqual(@as(usize, 48), lines.items[3].len);
    try t.expectEqualStrings("last", lines.items[4]);
    try t.expect(!cuts.items[4]);
}
