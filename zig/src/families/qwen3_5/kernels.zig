//! Qwen pipelines: the kernel templates compiled with a checkpoint's geometry, the row decoder's fixed arithmetic kept.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const sources = @import("kernel_sources").qwen35;
const layout = @import("kernel_sources").qwen35_layout;

const layout_names = [_][:0]const u8{ "qwen35_embed", "qwen35_head_norm", "qwen35_queries", "qwen35_keys", "qwen35_attention_gate" };
pub const fixed_total = sources.all.len + layout_names.len;
const max_projections = 16;

pub const Projection = struct { n: usize, k: usize, pipeline: mtl.Pipeline, columns: usize, threads: usize };

pub const Kernels = struct {
    fixed: [fixed_total]mtl.Pipeline,
    projections: [max_projections]Projection,
    count: usize,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        return self.fixed[comptime index(key)];
    }

    pub fn byKey(self: *const Kernels, key: []const u8) ?mtl.Pipeline {
        inline for (sources.all, 0..) |s, i| if (std.mem.eql(u8, key, s.key)) return self.fixed[i];
        for (layout_names, 0..) |s, i| if (std.mem.eql(u8, key, s)) return self.fixed[sources.all.len + i];
        return null;
    }

    /// The projection pipeline for `n` outputs of `k` inputs, compiled at load for this checkpoint's shapes.
    pub fn projection(self: *const Kernels, n: usize, k: usize) ?Projection {
        for (self.projections[0..self.count]) |p| if (p.n == n and p.k == k) return p;
        return null;
    }

    pub fn total(self: *const Kernels) usize {
        return fixed_total + self.count;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.fixed) |p| p.deinit();
        for (self.projections[0..self.count]) |p| p.pipeline.deinit();
    }
};

fn index(comptime key: []const u8) usize {
    inline for (sources.all, 0..) |s, i| if (comptime std.mem.eql(u8, key, s.key)) return i;
    inline for (layout_names, 0..) |s, i| if (comptime std.mem.eql(u8, key, s)) return sources.all.len + i;
    @compileError("no Qwen3.5 kernel " ++ key);
}

/// Split-K chunks by output count, as the row decoder chose them: more chunks feed small outputs independent chains.
pub fn splits(n: usize) usize {
    return if (n <= 64) 32 else if (n <= 6144) 16 else 8;
}

/// Column tiles a threadgroup covers (8 columns each), inside the split reduction's 16 KB of threadgroup memory.
pub fn tiles(n: usize) usize {
    var nt: usize = if (n % 32 == 0) 4 else if (n % 16 == 0) 2 else 1;
    while (nt > 1 and splits(n) * nt * 64 * 4 > 16384) nt /= 2;
    return nt;
}

/// The distinct (outputs, inputs) projection shapes a geometry's layers and head use.
pub fn shapes(g: cfg.Geometry, out: *[max_projections][2]usize) usize {
    const all = [_][2]usize{
        .{ g.intermediate, g.hidden },   .{ g.hidden, g.intermediate }, .{ g.convDim(), g.hidden },    .{ g.vInner(), g.hidden },
        .{ g.linear_v_heads, g.hidden }, .{ g.hidden, g.vInner() },     .{ 2 * g.qInner(), g.hidden }, .{ g.kvInner(), g.hidden },
        .{ g.hidden, g.qInner() },       .{ cfg.vocab, g.hidden },
    };
    var n: usize = 0;
    for (all) |s| {
        for (out[0..n]) |seen| {
            if (seen[0] == s[0] and seen[1] == s[1]) break;
        } else {
            out[n] = s;
            n += 1;
        }
    }
    return n;
}

/// The constants a template reads, for `key` under geometry `g`, as macros (Metal has no program-scope constexpr).
fn constants(buf: []u8, key: []const u8, g: cfg.Geometry) ![]const u8 {
    const eql = std.mem.eql;
    if (eql(u8, key, "norm") or eql(u8, key, "norm_nores")) return std.fmt.bufPrint(buf, "#define K {d}\n", .{g.hidden});
    if (eql(u8, key, "mlp_act")) return std.fmt.bufPrint(buf, "#define N {d}\n", .{g.intermediate});
    if (eql(u8, key, "gdn_pre")) return std.fmt.bufPrint(buf, "#define NK {d}\n#define NV {d}\n#define DK {d}\n#define DV {d}\n#define TAPS {d}\n", .{ g.linear_k_heads, g.linear_v_heads, cfg.linear_dim, cfg.linear_dim, cfg.conv_taps });
    if (eql(u8, key, "gdn_chain")) return std.fmt.bufPrint(buf, "#define Dk {d}\n#define Dv {d}\n#define Hk {d}\n#define Hv {d}\n", .{ cfg.linear_dim, cfg.linear_dim, g.linear_k_heads, g.linear_v_heads });
    if (eql(u8, key, "gdn_post")) return std.fmt.bufPrint(buf, "#define NV {d}\n#define DV {d}\n#define ZS {d}\n#define ZO 0\n", .{ g.linear_v_heads, cfg.linear_dim, g.vInner() });
    if (eql(u8, key, "attn_partial")) return std.fmt.bufPrint(buf, "#define D {d}\n#define G {d}\n#define CK 128\n#define SPLIT 4\n#define BLK 4\n", .{ cfg.head_dim, g.query_heads / g.kv_heads });
    if (eql(u8, key, "attn_merge")) return std.fmt.bufPrint(buf, "#define D {d}\n#define QH {d}\n", .{ cfg.head_dim, g.query_heads });
    if (eql(u8, key, "state_copy")) return buf[0..0];
    if (eql(u8, key, "layout")) return std.fmt.bufPrint(buf, "#define HID {d}\n#define QH {d}\n#define KVH {d}\n#define HD {d}\n#define QIN {d}\n", .{ g.hidden, g.query_heads, g.kv_heads, cfg.head_dim, g.qInner() });
    return error.UnknownQwenKernel;
}

fn library(gpa: std.mem.Allocator, device: mtl.Device, header: []const u8, source: []const u8) !mtl.Library {
    const text = try std.mem.concat(gpa, u8, &.{ header, source });
    defer gpa.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

pub fn load(gpa: std.mem.Allocator, device: mtl.Device, g: cfg.Geometry) !Kernels {
    var out: Kernels = undefined;
    out.count = 0;
    var loaded: usize = 0;
    errdefer {
        for (out.fixed[0..loaded]) |p| p.deinit();
        for (out.projections[0..out.count]) |p| p.pipeline.deinit();
    }
    var buf: [512]u8 = undefined;
    inline for (sources.all, 0..) |s, i| {
        const lib = try library(gpa, device, try constants(&buf, s.key, g), s.source);
        defer lib.deinit();
        out.fixed[i] = try mtl.Pipeline.init(device, lib, s.function, false);
        loaded += 1;
    }
    {
        const lib = try library(gpa, device, try constants(&buf, "layout", g), layout);
        defer lib.deinit();
        for (layout_names, 0..) |name, i| {
            out.fixed[sources.all.len + i] = try mtl.Pipeline.init(device, lib, name, false);
            loaded += 1;
        }
    }
    var list: [max_projections][2]usize = undefined;
    for (list[0..shapes(g, &list)]) |s| {
        const n, const k = .{ s[0], s[1] };
        const header = try std.fmt.bufPrint(&buf, "#define K {d}\n#define N {d}\n#define S {d}\n#define NT {d}\n", .{ k, n, splits(n), tiles(n) });
        const lib = try library(gpa, device, header, sources.qmm.source);
        defer lib.deinit();
        out.projections[out.count] = .{ .n = n, .k = k, .pipeline = try mtl.Pipeline.init(device, lib, sources.qmm.function, false), .columns = 8 * tiles(n), .threads = 256 };
        out.count += 1;
    }
    return out;
}

test "the 2B's projection shapes, chunks and tiles are the generated kernels'" {
    var list: [max_projections][2]usize = undefined;
    const n = shapes(cfg.geometries[0], &list);
    try std.testing.expectEqual(@as(usize, 7), n);
    for (list[0..n]) |s| {
        const nt: usize = if (s[0] == 16) 2 else 4;
        const sk: usize = if (s[0] == 16) 32 else if (s[0] == 248320) 8 else 16;
        try std.testing.expectEqual(nt, tiles(s[0]));
        try std.testing.expectEqual(sk, splits(s[0]));
    }
    try std.testing.expectEqual(@as(usize, 9), shapes(cfg.geometries[1], &list));
    try std.testing.expectEqual(@as(usize, 2), tiles(48));
}
