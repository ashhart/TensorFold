//! Reads the kernels list of a zebin's .ze_info YAML: each kernel's work-group, SLM, sub-group and GRF mode.
const std = @import("std");
const limits = @import("xpu_limits.zig");

pub const Error = error{ BadZeInfo, OutOfMemory };

pub const Kernel = struct { name: []const u8, usage: limits.XpuUsage };

/// Every kernel of the `kernels:` list; `kernels_misc_info` and other top-level keys are skipped.
pub fn kernels(a: std.mem.Allocator, text: []const u8) Error![]Kernel {
    var list: std.ArrayList(Kernel) = .empty;
    errdefer list.deinit(a);
    var in_kernels = false;
    var cur: ?usize = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (line.len == 0) continue;
        if (line[0] != ' ') {
            in_kernels = std.mem.startsWith(u8, line, "kernels:");
            cur = null;
            continue;
        }
        if (!in_kernels) continue;
        if (std.mem.startsWith(u8, line, "  - ")) {
            const rest = std.mem.trimStart(u8, line[3..], " ");
            const name = field(rest, "name") orelse return error.BadZeInfo;
            try list.append(a, .{ .name = std.mem.trim(u8, name, "'\""), .usage = .{} });
            cur = list.items.len - 1;
            continue;
        }
        const at = cur orelse continue;
        const u = &list.items[at].usage;
        if (field(line, "intel_reqd_sub_group_size")) |v| u.sub_group = try number(v);
        if (field(line, "grf_count")) |v| u.grf = try number(v);
        if (field(line, "slm_size")) |v| u.slm = try number(v);
        if (field(line, "spill_mem_size")) |v| u.spill = try number(v);
        if (field(line, "private_mem_size")) |v| u.private = try number(v);
        if (field(line, "required_work_group_size")) |v| u.threads = try triple(v);
        if (field(line, "workgroup_size")) |v| u.threads = try triple(v);
    }
    return list.toOwnedSlice(a);
}

/// The usage of the kernel named `name`, or null when the image does not hold it.
pub fn find(list: []const Kernel, name: []const u8) ?limits.XpuUsage {
    for (list) |k| if (std.mem.eql(u8, k.name, name)) return k.usage;
    return null;
}

/// The text after `key:` on a line that holds it, else null.
fn field(line: []const u8, key: []const u8) ?[]const u8 {
    const t = std.mem.trim(u8, line, " ");
    if (!std.mem.startsWith(u8, t, key)) return null;
    if (t.len <= key.len or t[key.len] != ':') return null;
    return std.mem.trim(u8, t[key.len + 1 ..], " ");
}

fn number(text: []const u8) Error!u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, text, " "), 10) catch error.BadZeInfo;
}

fn triple(text: []const u8) Error![3]u32 {
    const inner = std.mem.trim(u8, std.mem.trim(u8, text, " "), "[]");
    var it = std.mem.splitScalar(u8, inner, ',');
    var out: [3]u32 = undefined;
    for (&out) |*v| {
        const part = it.next() orelse return error.BadZeInfo;
        v.* = try number(part);
    }
    return out;
}

const sample =
    \\---
    \\version:         '1.73'
    \\kernels:
    \\  - name:            add_rmsnorm_dec
    \\    user_attributes:
    \\      intel_reqd_sub_group_size: 16
    \\      reqd_work_group_size: [ 512, 1, 1 ]
    \\    execution_env:
    \\      grf_count:       256
    \\      required_work_group_size: [ 512, 1, 1 ]
    \\      simd_size:       16
    \\      slm_size:        10368
    \\      eu_thread_count: 4
    \\  - name:            embed
    \\    user_attributes:
    \\      intel_reqd_sub_group_size: 16
    \\      reqd_work_group_size: [ 256, 1, 1 ]
    \\    execution_env:
    \\      grf_count:       256
    \\      required_work_group_size: [ 256, 1, 1 ]
    \\      spill_mem_size:  64
    \\kernels_misc_info:
    \\  - name:            add_rmsnorm_dec
    \\    payload_arguments: []
    \\
;

test "the kernels list is read and the misc list is not" {
    const a = std.testing.allocator;
    const list = try kernels(a, sample);
    defer a.free(list);
    try std.testing.expectEqual(@as(usize, 2), list.len);
    const dec = find(list, "add_rmsnorm_dec").?;
    try std.testing.expectEqual([3]u32{ 512, 1, 1 }, dec.threads);
    try std.testing.expectEqual(@as(u32, 10368), dec.slm);
    try std.testing.expectEqual(@as(u32, 16), dec.sub_group);
    try std.testing.expectEqual(@as(u32, 256), dec.grf);
    const embed = find(list, "embed").?;
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, embed.threads);
    try std.testing.expectEqual(@as(u32, 0), embed.slm);
    try std.testing.expectEqual(@as(u32, 64), embed.spill);
    try std.testing.expect(find(list, "absent") == null);
}

test "a kernel line without a name is refused" {
    try std.testing.expectError(error.BadZeInfo, kernels(std.testing.allocator, "kernels:\n  - symbol: x\n"));
}
