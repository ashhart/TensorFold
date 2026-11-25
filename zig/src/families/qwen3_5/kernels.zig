//! Qwen pipelines use the existing row decoder's fixed arithmetic, including on GPUs without tensor units.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources").qwen35;

const layout_names = [_][:0]const u8{ "qwen35_embed", "qwen35_head_norm", "qwen35_queries", "qwen35_keys", "qwen35_attention_gate" };
const layout = @import("kernel_sources").qwen35_layout;
/// The KV-page twins of qwen35_keys and attn_partial (TF_QWEN35_METAL_KV_PAGES=1), in their own library.
const paged_names = [_][:0]const u8{ "qwen35_keys_paged", "qwen35_attn_partial_paged" };
const paged = @import("kernel_sources").qwen35_paged;
pub const total = sources.all.len + layout_names.len + paged_names.len;

pub const Kernels = struct {
    pipelines: [total]mtl.Pipeline,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        return self.pipelines[comptime index(key)];
    }

    pub fn byKey(self: *const Kernels, key: []const u8) ?mtl.Pipeline {
        inline for (sources.all, 0..) |s, i| if (std.mem.eql(u8, key, s.key)) return self.pipelines[i];
        for (layout_names, 0..) |s, i| if (std.mem.eql(u8, key, s)) return self.pipelines[sources.all.len + i];
        for (paged_names, 0..) |s, i| if (std.mem.eql(u8, key, s)) return self.pipelines[sources.all.len + layout_names.len + i];
        return null;
    }

    pub fn projection(self: *const Kernels, n: usize, k: usize) ?Projection {
        @setEvalBranchQuota(10000);
        inline for (sources.all, 0..) |s, i| {
            if (s.columns > 0) {
                const nk = comptime dimensions(s.key);
                if (n == nk[0] and k == nk[1]) return .{ .pipeline = self.pipelines[i], .columns = s.columns, .threads = s.threads };
            }
        }
        return null;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
    }
};

pub const Projection = struct { pipeline: mtl.Pipeline, columns: usize, threads: usize };

fn dimensions(comptime key: []const u8) [2]usize {
    const tail = key[4..];
    const at = std.mem.indexOfScalar(u8, tail, '_').?;
    return .{ std.fmt.parseInt(usize, tail[0..at], 10) catch unreachable, std.fmt.parseInt(usize, tail[at + 1 ..], 10) catch unreachable };
}

fn index(comptime key: []const u8) usize {
    inline for (sources.all, 0..) |s, i| if (comptime std.mem.eql(u8, key, s.key)) return i;
    inline for (layout_names, 0..) |s, i| if (comptime std.mem.eql(u8, key, s)) return sources.all.len + i;
    inline for (paged_names, 0..) |s, i| if (comptime std.mem.eql(u8, key, s)) return sources.all.len + layout_names.len + i;
    @compileError("no Qwen3.5 kernel " ++ key);
}

/// One library `load` compiles: its source, the functions it takes in pipeline order and a projection's tile.
pub const Library = struct { source: []const u8, functions: []const [:0]const u8, columns: usize = 0, threads: usize = 0 };

/// Every library `load` compiles, in pipeline order: each projection's own, then the layout and the KV-page twins.
pub const libraries = blk: {
    var l: [sources.all.len + 2]Library = undefined;
    for (sources.all, 0..) |s, i| l[i] = .{ .source = s.source, .functions = &.{s.function}, .columns = s.columns, .threads = s.threads };
    l[sources.all.len] = .{ .source = layout, .functions = &layout_names };
    l[sources.all.len + 1] = .{ .source = paged, .functions = &paged_names };
    break :blk l;
};

/// The options every library compiles with: MLX's, with this OS's language version.
pub fn options() mtl.CompileOptions {
    return mtl.CompileOptions.mlx();
}

pub fn load(device: mtl.Device) !Kernels {
    var out: Kernels = undefined;
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |p| p.deinit();
    for (libraries) |l| {
        const lib = try mtl.Library.fromSource(device, l.source, options());
        defer lib.deinit();
        for (l.functions) |f| {
            out.pipelines[loaded] = try mtl.Pipeline.init(device, lib, f, false);
            loaded += 1;
        }
    }
    return out;
}
