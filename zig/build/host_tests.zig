//! Native Linux CI collects named host tests and compile checks after every build module has registered its steps.

// `test-host-a` and `test-host-b` are disjoint; `test-host` depends on both, so a local `zig build test-host` runs all.

// Every other named check lands in exactly one shard; a build-time check rejects a name that is missing or doubled.

// Slow names are pinned by group: A gets the CUDA host, the nightly host suite and DeepSeek's long AOT tests.

// B gets the server, goldens and the compile, fast and HIP suites with their check-* gates.

// The rest go by the low bit of FNV-1a of the name, stable across runs and independent of registration order.

const std = @import("std");
const Step = std.Build.Step;
const Set = std.AutoHashMapUnmanaged(*Step, void);

const Exclusion = struct { name: []const u8, reason: []const u8 };
const exclusions = [_]Exclusion{
    .{ .name = "test-grammar-budget", .reason = "Requires the optional xgrammar checkout and -Dxgrammar=DIR" },
    .{ .name = "test-grammar-tools", .reason = "Requires the optional xgrammar checkout and -Dxgrammar=DIR" },
    .{ .name = "check-release-paths", .reason = "The release-paths CI job checks the separately built release server" },
    .{ .name = "test-startup-states-timing", .reason = "Manual disk startup timing seeds up to 100,000 states" },
    .{ .name = "test-imprint-ops-timing", .reason = "Manual Imprint timing seeds up to 100,000 learned states" },
};

// A holds the long execution suites; B holds the compile-heavy and the remaining standalone suites.

const heavy_a = [_][]const u8{
    "test-cuda-host",
    "test-host-ci-linux-host",
    "test-host-ci-linux-deepseek",
    "test-dsv41-device-host",
    "test-dsv41-forward",
    "test-sessions-writes-host",
};

const heavy_b = [_][]const u8{
    "test-server-cpu",
    "test-golden",
    "test-host-ci-linux-fast",
    "test-host-ci-linux-compile",
    "test-host-ci-linux-hip",
    "check-drama-cuda-host",
    "check-glm-gather-gpu-host",
};

pub fn wire(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const host = b.graph.host.result;
    if (host.os.tag != .linux or target.result.os.tag != host.os.tag or
        target.result.cpu.arch != host.cpu.arch or target.result.abi != host.abi) return;

    const all = b.step("test-host", "Native Linux host tests and compile checks with reviewed exclusions");
    const shard_a = b.step("test-host-a", "First shard of the native Linux host tests and compile checks");
    const shard_b = b.step("test-host-b", "Second shard of the native Linux host tests and compile checks");
    all.dependOn(shard_a);
    all.dependOn(shard_b);

    var rejected: std.ArrayList(*Step) = .empty;
    for (b.top_level_steps.values()) |top| {
        const name = top.step.name;
        const step = &top.step;
        if (step == all or step == shard_a or step == shard_b or !isCheck(name) or exclusion(name) != null) continue;
        if (step.dependencies.items.len == 0)
            reject(b, &rejected, b.fmt("test-host: named check {s} has no build steps", .{name}));
        if (shardFor(name) == 0) shard_a.dependOn(step) else shard_b.dependOn(step);
    }

    // Read the shards back instead of trusting the loop: a doubled name must fail before CI runs it twice.
    var in_a: Set = .empty;
    var in_b: Set = .empty;
    for (shard_a.dependencies.items) |dep| _ = in_a.getOrPut(b.allocator, dep) catch @panic("OOM");
    for (shard_b.dependencies.items) |dep| _ = in_b.getOrPut(b.allocator, dep) catch @panic("OOM");
    for (shard_b.dependencies.items) |dep| {
        if (in_a.contains(dep))
            reject(b, &rejected, b.fmt("test-host: check {s} is in both shards", .{dep.name}));
    }
    for (b.top_level_steps.values()) |top| {
        const name = top.step.name;
        const step = &top.step;
        if (step == all or step == shard_a or step == shard_b or !isCheck(name) or exclusion(name) != null) continue;
        if (!in_a.contains(step) and !in_b.contains(step))
            reject(b, &rejected, b.fmt("test-host: check {s} is in neither shard", .{name}));
    }

    // An excluded step may still be reached through an alias declared in either shard.
    var included: Set = .empty;
    collect(b.allocator, shard_a, &included);
    collect(b.allocator, shard_b, &included);
    for (exclusions) |entry| {
        const top = b.top_level_steps.get(entry.name) orelse {
            reject(b, &rejected, b.fmt("test-host: stale exclusion {s}: {s}", .{ entry.name, entry.reason }));
            continue;
        };
        var visited: Set = .empty;
        if (includedBoundary(b.allocator, &top.step, &included, &visited))
            reject(b, &rejected, b.fmt("test-host reaches excluded {s}: {s}", .{ entry.name, entry.reason }));
    }
    if (rejected.items.len != 0) all.dependencies = rejected;
}

fn reject(b: *std.Build, rejected: *std.ArrayList(*Step), message: []const u8) void {
    rejected.append(b.allocator, &b.addFail(message).step) catch @panic("OOM");
}

fn isCheck(name: []const u8) bool {
    return std.mem.eql(u8, name, "test") or std.mem.startsWith(u8, name, "test-") or
        std.mem.startsWith(u8, name, "check-");
}

fn exclusion(name: []const u8) ?Exclusion {
    for (exclusions) |entry| if (std.mem.eql(u8, name, entry.name)) return entry;
    return null;
}

fn shardFor(name: []const u8) u1 {
    for (heavy_a) |heavy| if (std.mem.eql(u8, name, heavy)) return 0;
    for (heavy_b) |heavy| if (std.mem.eql(u8, name, heavy)) return 1;
    var h: u64 = 14695981039346656037;
    for (name) |c| h = (h ^ c) *% 1099511628211;
    return @intCast(h & 1);
}

fn collect(a: std.mem.Allocator, root: *Step, seen: *Set) void {
    if ((seen.getOrPut(a, root) catch @panic("OOM")).found_existing) return;
    for (root.dependencies.items) |dep| collect(a, dep, seen);
}

fn includedBoundary(a: std.mem.Allocator, root: *Step, included: *const Set, seen: *Set) bool {
    if ((seen.getOrPut(a, root) catch @panic("OOM")).found_existing) return false;
    if (included.contains(root)) return true;
    if (root.tag != .top_level) return false;
    for (root.dependencies.items) |dep| if (includedBoundary(a, dep, included, seen)) return true;
    return false;
}
