//! Pure host launch shapes, ABI structs, and FP64 references for Inkling 2 Mini tree conv and compaction.
const std = @import("std");

pub const hidden = 2304;
pub const visit_slots = 48;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const conv_states = 4;
pub const window_rows = 16;
pub const conv_widths = [conv_states]usize{ 1024, 1024, 2304, 2304 };
pub const conv_row_values = 1024 + 1024 + 2304 + 2304;

pub const ConvTreeArgs = extern struct { rows: u32, channels: u32 };
pub const KeepKvArgs = extern struct {
    key_at: [visit_slots]u64,
    value_at: [visit_slots]u64,
    rings: [visit_slots]u32,
    start: u32,
    path_len: u32,
    path: [window_rows]u32,
};
pub const KeepConvArgs = extern struct { src: [3]u32, tap_rows: u32 };

comptime {
    if (@sizeOf(ConvTreeArgs) != 8 or @sizeOf(KeepKvArgs) != 1032 or @sizeOf(KeepConvArgs) != 16)
        @compileError("Inkling tree ABI drift");
}

pub const Kind = enum {
    conv_static_tree,
    conv_dynamic_tree,
    keep_kv,
    keep_conv,

    pub fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .conv_static_tree => "tf_ink2_conv_static_tree",
            .conv_dynamic_tree => "tf_ink2_conv_dynamic_tree",
            .keep_kv => "tf_ink2_keep_kv",
            .keep_conv => "tf_ink2_keep_conv",
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            .conv_static_tree => "metal inkling2 tree conv_static tf_ink2_conv_static_tree",
            .conv_dynamic_tree => "metal inkling2 tree conv_dynamic tf_ink2_conv_dynamic_tree",
            .keep_kv => "metal inkling2 tree keep_kv tf_ink2_keep_kv",
            .keep_conv => "metal inkling2 tree keep_conv tf_ink2_keep_conv",
        };
    }
};

pub fn roundBf16(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    const bits: u32 = @bitCast(x);
    const rounded: u16 = if (std.math.isNan(x))
        @truncate((bits >> 16) | 0x40)
    else
        @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    const value: f32 = @bitCast(@as(u32, rounded) << 16);
    return value;
}

fn convTap(history: []const f32, raw: []const f32, channels: usize, idx: usize, c: usize) f64 {
    return if (idx < 3) history[idx * channels + c] else raw[(idx - 3) * channels + c];
}

pub fn convStaticTree(history: []const f32, raw: []const f32, weight: []const f32, src: []const u32, channels: usize, r: usize, c: usize) f64 {
    var sum: f64 = convTap(history, raw, channels, src[r * 4 + 3], c);
    for (0..4) |tap| sum += @as(f64, weight[c * 4 + tap]) * convTap(history, raw, channels, src[r * 4 + tap], c);
    return sum / @sqrt(5.0);
}

pub fn convDynamicTree(history: []const f32, raw: []const f32, input: []const f32, static_w: []const f32, projection: []const f32, src: []const u32, channels: usize, r: usize, c: usize) f64 {
    var sum: f64 = convTap(history, raw, channels, src[r * 4 + 3], c);
    for (0..4) |lag| {
        var dot: f64 = 0;
        for (0..hidden) |d| dot += @as(f64, projection[(4 * (c / 128) + lag) * hidden + d]) * input[r * hidden + d];
        const w = @as(f64, static_w[lag * channels + c]) + roundBf16(dot);
        sum += w * convTap(history, raw, channels, src[r * 4 + 3 - lag], c);
    }
    return sum / 3;
}

pub fn validateParents(parents: []const i32, depths_out: []u32) !void {
    if (parents.len == 0 or parents.len > window_rows or depths_out.len != parents.len) return error.BadParents;
    if (parents[0] != -1) return error.BadParents;
    depths_out[0] = 0;
    for (parents[1..], 1..) |p, r| {
        if (p < 0 or p >= @as(i32, @intCast(r))) return error.BadParents;
        depths_out[r] = depths_out[@intCast(p)] + 1;
    }
}

pub fn isChain(parents: []const i32) bool {
    for (parents, 0..) |p, r| if (p != @as(i32, @intCast(r)) - 1) return false;
    return true;
}

pub fn validateKeepPath(rows: usize, parents: ?[]const i32, path: []const u32) !void {
    if (path.len == 0 or path.len > rows or path[0] != 0) return error.UnsupportedInklingWindow;
    for (path[1..], 1..) |r, j| {
        if (r >= rows) return error.UnsupportedInklingWindow;
        if (parents) |ps| {
            if (ps[r] != @as(i32, @intCast(path[j - 1]))) return error.UnsupportedInklingWindow;
        } else if (r != j) return error.UnsupportedInklingWindow;
    }
}

pub fn buildConvSrc(parents: []const i32, out: []u32) void {
    std.debug.assert(out.len == parents.len * 4);
    for (0..parents.len) |r| {
        var at: i32 = @intCast(r);
        var hist: u32 = 2;
        for (0..4) |back| {
            const tap = 3 - back;
            if (at >= 0) {
                out[r * 4 + tap] = 3 + @as(u32, @intCast(at));
                at = parents[@intCast(at)];
            } else {
                out[r * 4 + tap] = hist;
                hist -|= 1;
            }
        }
    }
}

pub fn buildTreePaths(parents: []const i32, depths: []const u32, out: []u32) void {
    std.debug.assert(depths.len == parents.len and out.len == parents.len * window_rows);
    @memset(out, 0);
    for (0..parents.len) |r| {
        var at: usize = r;
        var k: usize = depths[r] + 1;
        while (k > 0) {
            k -= 1;
            out[r * window_rows + k] = @intCast(at);
            if (k > 0) at = @intCast(parents[at]);
        }
    }
}

pub fn convHistorySrc(path: []const u32) [3]u32 {
    std.debug.assert(path.len >= 1);
    return switch (path.len) {
        1 => .{ 1, 2, 3 + path[0] },
        2 => .{ 2, 3 + path[0], 3 + path[1] },
        else => .{ 3 + path[path.len - 3], 3 + path[path.len - 2], 3 + path[path.len - 1] },
    };
}
