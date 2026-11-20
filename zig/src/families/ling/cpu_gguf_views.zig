//! Source view selection; expert stacks are sliced without copying quantized payloads.
const std = @import("std");
const g = @import("core").gguf;
const names = @import("gguf_names.zig");
pub const View = struct { tensor: g.Tensor, name: names.Name };

pub fn expert(t: g.Tensor, index: usize) !g.Tensor {
    if (t.rank != 3 or t.shape[2] != 128 or index >= 128 or t.bytes.len != t.len or t.len % 128 != 0) return error.BadLingGgufExpert;
    const bytes = t.len / 128;
    var out = t;
    out.rank = 2;
    out.shape[2] = 1;
    out.offset = std.math.add(usize, out.offset, index * bytes) catch return error.BadLingGgufExpert;
    out.len = bytes;
    out.bytes = t.bytes[index * bytes ..][0..bytes];
    return out;
}

pub fn row(t: g.Tensor, index: usize) !g.Tensor {
    if (t.rank != 2 or index >= t.shape[1] or t.shape[1] == 0 or t.bytes.len != t.len or t.len % t.shape[1] != 0) return error.BadLingGgufRow;
    const bytes = t.len / @as(usize, @intCast(t.shape[1]));
    var out = t;
    out.shape[1] = 1;
    out.offset = std.math.add(usize, out.offset, index * bytes) catch return error.BadLingGgufRow;
    out.len = bytes;
    out.bytes = t.bytes[index * bytes ..][0..bytes];
    return out;
}

pub fn resolve(d: *const g.Document, plan: []const names.Name, key: []const u8) !View {
    var buffer: [256]u8 = undefined;
    var template = key;
    var index: ?usize = null;
    const marker = ".mlp.experts.";
    if (std.mem.indexOf(u8, key, marker)) |at| {
        const first = at + marker.len;
        const end = first + (std.mem.indexOfScalar(u8, key[first..], '.') orelse return error.BadLingGgufExpert);
        index = std.fmt.parseInt(usize, key[first..end], 10) catch return error.BadLingGgufExpert;
        template = try std.fmt.bufPrint(&buffer, "{s}{{id}}{s}", .{ key[0..first], key[end..] });
    }
    for (plan) |n| if (std.mem.eql(u8, n.native, template)) {
        var t = d.getTensor(n.source) orelse return error.MissingLingGgufTensor;
        if (index) |id| {
            if (n.role != .experts) return error.BadLingGgufExpert;
            t = try expert(t, id);
        } else if (n.role == .experts) return error.BadLingGgufExpert;
        return .{ .tensor = t, .name = n };
    };
    return error.MissingLingGgufTensor;
}

test "expert slices preserve source rows and reject invalid stacks and indices" {
    var bytes: [128 * 144]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i);
    const t = g.Tensor{ .name = "stack", .kind = .q4_k, .rank = 3, .shape = .{ 256, 1, 128, 1 }, .offset = 32, .len = bytes.len, .bytes = &bytes };
    const one = try expert(t, 127);
    try std.testing.expectEqual(@as(u8, 2), one.rank);
    try std.testing.expectEqualSlices(u8, bytes[127 * 144 ..], one.bytes);
    try std.testing.expectEqual(@as(usize, 32 + 127 * 144), one.offset);
    try std.testing.expectError(error.BadLingGgufExpert, expert(t, 128));
    var malformed = t;
    malformed.shape[2] = 127;
    try std.testing.expectError(error.BadLingGgufExpert, expert(malformed, 0));
}

test "native expert lookup substitutes only a checked numeric expert index" {
    const a = std.testing.allocator;
    var bytes: [128 * 144]u8 = @splat(0);
    var tensors = [_]g.Tensor{.{ .name = "blk.1.ffn_gate_exps.weight", .kind = .q4_k, .rank = 3, .shape = .{ 256, 1, 128, 1 }, .offset = 0, .len = bytes.len, .bytes = &bytes }};
    var doc = g.Document{ .gpa = a, .bytes = &bytes, .metadata = &.{}, .tensors = &tensors, .data_offset = 0, .alignment = 32 };
    defer doc.names.deinit(a);
    try doc.names.put(a, tensors[0].name, 0);
    const plan = [_]names.Name{.{ .source = tensors[0].name, .native = "model.layers.1.mlp.experts.{id}.gate_proj.weight", .role = .experts, .rank = 3, .dims = .{ 256, 1, 128 } }};
    const found = try resolve(&doc, &plan, "model.layers.1.mlp.experts.127.gate_proj.weight");
    try std.testing.expectEqual(@as(usize, 144), found.tensor.bytes.len);
    try std.testing.expectEqual(@intFromPtr(&bytes) + 127 * 144, @intFromPtr(found.tensor.bytes.ptr));
    try std.testing.expectError(error.BadLingGgufExpert, resolve(&doc, &plan, "model.layers.1.mlp.experts.128.gate_proj.weight"));
    try std.testing.expectError(error.BadLingGgufExpert, resolve(&doc, &plan, "model.layers.1.mlp.experts.-1.gate_proj.weight"));
}
