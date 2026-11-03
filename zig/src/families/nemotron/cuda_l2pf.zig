//! Default-off resident-weight prefetch for Nemotron decode windows, never prompt chunks.
const std = @import("std");
const cuda = @import("cuda");
const src = cuda.l2_sources;
const weights = @import("cuda_weights.zig");
pub const Options = src.Settings("TF_NEMOTRON_L2PF");
fn get(key: [:0]const u8) ?[]const u8 {
    const v = @import("sys").c.getenv(key) orelse return null;
    return std.mem.span(v);
}
pub fn sources(blocks: []const weights.Block, i: usize, site: cuda.l2_prefetch.Site) src.Sources {
    var s: src.Sources = .{};
    if (site == .f or (site == .n and i + 1 == blocks.len)) return s;
    const b = blocks[if (site == .n) i + 1 else i];
    switch (b.kind) {
        .mamba => s.linear(if (site == .n) b.mamba.in_proj else b.mamba.out_proj),
        .attention => s.linear(if (site == .n) b.attn.qkv else b.attn.o),
        .moe => if (site == .n) {
            s.add(b.moe.router);
        },
    }
    return s;
}
pub fn init(e: anytype) !?cuda.l2_prefetch.Runtime {
    var problem: []const u8 = "";
    const o = Options.read(get, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    if (!o.on) return null;
    var ranges: std.ArrayList(src.Range) = .empty;
    defer ranges.deinit(e.gpa);
    for (e.w.buffers.items) |b| try ranges.append(e.gpa, .{ .ptr = b.ptr, .len = b.len });
    const views = try e.gpa.alloc(src.Sources, e.w.blocks.len * 3);
    defer e.gpa.free(views);
    for (e.w.blocks, 0..) |_, i| for ([_]cuda.l2_prefetch.Site{ .n, .f, .o }) |site| {
        views[i * 3 + @backingInt(site)] = sources(e.w.blocks, i, site);
    };
    const p = try src.start(e.gpa, e.ctx.d, true, o.knobs, views, ranges.items);
    std.log.info("TF_NEMOTRON_L2PF=1: {d} MiB/site, {d} CTAs, paced sites {d} GB/s", .{ o.knobs.mb, o.knobs.ctas, o.knobs.pace_gbps });
    return p;
}

test {
    _ = @import("cuda_l2pf_test.zig");
}
