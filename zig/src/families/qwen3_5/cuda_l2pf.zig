//! Optional read-only prefetch of resident Qwen3.5/3.6 dense and routed decode weights.
const std = @import("std");
const cuda = @import("cuda");
const src = cuda.l2_sources;
const Layer = @import("cuda_layers.zig").Layer;
const Proj = @import("cuda_layers.zig").Proj;
pub const Options = src.Settings("TF_QWEN_L2PF");
fn get(key: [:0]const u8) ?[]const u8 {
    const v = @import("sys").c.getenv(key) orelse return null;
    return std.mem.span(v);
}
pub fn options() !Options {
    var problem: []const u8 = "";
    return Options.read(get, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
}
fn proj(s: *src.Sources, p: Proj) void {
    switch (p) {
        inline .a4, .an => |a| s.affine(a),
        .ql => |a| s.linear(a),
        .x3 => |a| s.exl3(a),
        .bf => |a| s.add(a.w),
    }
}
pub fn sources(layers: []const Layer, i: usize, site: cuda.l2_prefetch.Site) src.Sources {
    var s: src.Sources = .{};
    if (site == .n and i + 1 == layers.len) return s;
    const l = layers[if (site == .n) i + 1 else i];
    switch (site) {
        .n => switch (l.mixer) {
            .delta => |m| {
                proj(&s, m.qkv);
                proj(&s, m.z);
                proj(&s, m.a);
                proj(&s, m.b);
            },
            .attention => |m| {
                proj(&s, m.q);
                proj(&s, m.k);
                proj(&s, m.v);
            },
        },
        .f => switch (l.mlp) {
            .dense => |m| {
                proj(&s, m.gate);
                proj(&s, m.up);
            },
            .moe => |m| s.add(m.router),
        },
        .o => switch (l.mixer) {
            .delta => |m| proj(&s, m.out),
            .attention => |m| proj(&s, m.o),
        },
    }
    return s;
}
pub fn init(e: anytype, o: Options, ranges: []const src.Range) !?cuda.l2_prefetch.Runtime {
    if (!o.on) return null;
    const views = try e.gpa.alloc(src.Sources, e.layers.len * 3);
    defer e.gpa.free(views);
    for (e.layers, 0..) |_, i| for ([_]cuda.l2_prefetch.Site{ .n, .f, .o }) |site| {
        views[i * 3 + @backingInt(site)] = sources(e.layers, i, site);
    };
    const p = try src.start(e.gpa, e.d, true, o.knobs, views, ranges);
    std.log.info("TF_QWEN_L2PF=1: {d} MiB/site, {d} CTAs, paced sites {d} GB/s", .{ o.knobs.mb, o.knobs.ctas, o.knobs.pace_gbps });
    return p;
}

test {
    _ = @import("cuda_l2pf_test.zig");
}
