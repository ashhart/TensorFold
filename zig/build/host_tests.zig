//! Native Linux CI collects named host tests and compile checks after every build module has registered its steps.
const std = @import("std");
const Step = std.Build.Step;
const Set = std.AutoHashMapUnmanaged(*Step, void);

const Exclusion = struct { name: []const u8, reason: []const u8 };
const exclusions = [_]Exclusion{
    .{ .name = "test-grammar-budget", .reason = "Requires the optional xgrammar checkout and -Dxgrammar=DIR" },
    .{ .name = "test-grammar-tools", .reason = "Requires the optional xgrammar checkout and -Dxgrammar=DIR" },
    .{ .name = "check-release-paths", .reason = "The release-paths CI job checks the separately built release server" },
    .{ .name = "test-qwen35-draft-frame-host", .reason = "red on main, #2031 (recording fixture leaves seg_fn uninitialized)" },
    .{ .name = "test-qwen35-frame-invalidation-host", .reason = "red on main, #2031 (same recording fixture as test-qwen35-draft-frame-host)" },
    .{ .name = "test-gradient-serving", .reason = "red on main, #2031 (Flash Next hyper-connection source pin drifted)" },
};

pub fn wire(b: *std.Build, target: std.Build.ResolvedTarget) void {
    const host = b.graph.host.result;
    if (host.os.tag != .linux or target.result.os.tag != host.os.tag or
        target.result.cpu.arch != host.cpu.arch or target.result.abi != host.abi) return;

    const all = b.step("test-host", "Native Linux host tests and compile checks with reviewed exclusions");
    var rejected: std.ArrayList(*Step) = .empty;
    for (b.top_level_steps.values()) |top| {
        const name = top.step.name;
        if (&top.step == all or !isCheck(name) or exclusion(name) != null) continue;
        if (top.step.dependencies.items.len == 0)
            reject(b, &rejected, b.fmt("test-host: named check {s} has no build steps", .{name}));
        all.dependOn(&top.step);
    }

    var included: Set = .empty;
    collect(b.allocator, all, &included);
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
