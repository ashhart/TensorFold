//! TF_NEMOTRON_METAL_KV_PAGES=1: the paged twins' pipelines (kernels/metal/nemotron/paged.zig) and their bindings.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const st = @import("state.zig");
const Enc = @import("encoder.zig").Enc;
const simd_attention = @import("simd_attention.zig");
const frags = @import("../../core/frags.zig");
const kv_pool = @import("../../core/kv_pool.zig");
const kvp = @import("../../core/metal_kv_pages.zig");
const paged = sources.nemotron_paged;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Whether the streams' keys and values live in pages of the core pool (off by default).
pub fn on() bool {
    const v = std.c.getenv("TF_NEMOTRON_METAL_KV_PAGES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Backend `b` gets `pages` pages over its `o.streams` lanes for streams of at most `most` rows, and the twins.
pub fn attach(b: anytype, l: kv_pool.Layout, pages: usize, most: u64, served: u32) !void {
    const sized = kvp.Pages.bytes(l, b.o.streams, pages, most);
    if (sized.arena > b.m.device.maxBufferLength()) return error.ArenaTooLarge;
    b.kv_pipes = try Pipelines.load(b.gpa, b.m.device);
    b.m.kernels.paged = b.kv_pipes;
    b.kv_arena = try b.m.device.buffer(@intCast(sized.arena), opts);
    b.kv_tables = try b.m.device.buffer(@intCast(sized.tables), opts);
    if (b.residency) |f| for ([_]mtl.Buffer{ b.kv_arena.?, b.kv_tables.? }) |x| f.join(f.ctx, x);
    b.kv_pages = try kvp.Pages.initIn(b.gpa, l, b.o.streams, pages, most, b.kv_arena.?, b.kv_tables.?);
    const gib = @as(f64, @floatFromInt(sized.arena)) / (1 << 30);
    const mib = @as(f64, @floatFromInt(l.pageBytes())) / (1 << 20);
    std.log.info("nemotron: KV in pages of the core pool: {d} streams over {d} pages of 256 rows ({d:.2} GiB, {d:.2} MiB a page)", .{ served, pages, gib, mib });
}

/// What attach made goes back, the model's kernels pointing at main's again.
pub fn detach(b: anytype) void {
    if (b.kv_pages) |p| p.deinit();
    for ([_]?mtl.Buffer{ b.kv_arena, b.kv_tables }) |x| if (x) |buf| buf.deinit();
    if (b.kv_pipes) |p| {
        b.m.kernels.paged = null;
        p.deinit(b.gpa);
    }
}

/// With no stream open every page must be free: the census says so, or warns.
pub fn census(b: anytype) void {
    const p = b.kv_pages orelse return;
    const c = p.pool.census();
    if (p.idle() and c.free == c.total) {
        std.log.info("nemotron: KV pages idle, {d} of {d} free", .{ c.free, c.total });
    } else std.log.warn("nemotron: no stream open, yet {d} of {d} KV pages are not free", .{ c.total - c.free, c.total });
}

/// The twins as the loaders build their originals here: the attention rewrite and tree layout before the M5, nax.h.
pub const Pipelines = struct {
    glue: [paged.glue_names.len]mtl.Pipeline, // kv write, kv compact, prompt kv put
    partial: [paged.partial_names.len]mtl.Pipeline,
    tail: mtl.Pipeline,
    prompt: [paged.prefill_names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !*Pipelines {
        const p = try gpa.create(Pipelines);
        errdefer gpa.destroy(p);
        var made: std.ArrayList(mtl.Pipeline) = .empty;
        defer made.deinit(gpa);
        errdefer for (made.items) |x| x.deinit();
        const tensor = device.tensorUnits();
        try build(gpa, device, paged.glue, &paged.glue_names, &p.glue, &made);
        const partial = if (tensor) try gpa.dupe(u8, paged.partial) else try simd_attention.rewrite(gpa, paged.partial);
        defer gpa.free(partial);
        try build(gpa, device, partial, &paged.partial_names, &p.partial, &made);
        const tree = if (tensor) try gpa.dupe(u8, paged.tree) else try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", paged.tree });
        defer gpa.free(tree);
        var tail: [1]mtl.Pipeline = undefined;
        try build(gpa, device, tree, &.{paged.tree_name}, &tail, &made);
        p.tail = tail[0];
        const prompt = try frags.source(device, gpa, paged.prefill);
        defer gpa.free(prompt);
        try build(gpa, device, prompt, &paged.prefill_names, &p.prompt, &made);
        return p;
    }

    fn build(gpa: std.mem.Allocator, device: mtl.Device, text: []const u8, names: []const [:0]const u8, out: []mtl.Pipeline, made: *std.ArrayList(mtl.Pipeline)) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (names, out) |n, *o| {
            o.* = try mtl.Pipeline.init(device, lib, n, false);
            try made.append(gpa, o.*);
        }
    }

    pub fn deinit(p: *Pipelines, gpa: std.mem.Allocator) void {
        for (p.glue) |x| x.deinit();
        for (p.partial) |x| x.deinit();
        p.tail.deinit();
        for (p.prompt) |x| x.deinit();
        gpa.destroy(p);
    }

    /// The prompt attention twin for (whole query blocks, whole key blocks), as prefill_launch picks main's.
    pub fn promptFor(p: *const Pipelines, whole_q: bool, whole_k: bool) mtl.Pipeline {
        return p.prompt[@as(usize, @intFromBool(whole_q)) * 2 + @intFromBool(whole_k)];
    }
};

/// A cache's keys and values at slots `k` and `v`: its own buffers, or its part of every page.
pub fn bindKv(e: *Enc, kv: st.Kv, k: usize, v: usize) void {
    if (kv.paged) |p| {
        e.buf(p.arena(), p.k_off, k);
        e.buf(p.arena(), p.v_off, v);
    } else {
        e.buf(kv.k, 0, k);
        e.buf(kv.v, 0, v);
    }
}

/// A paged cache's table and page bytes at `slot` and `slot + 1`.
pub fn bindTable(e: *Enc, p: st.Paged, slot: usize) void {
    e.buf(p.table(), p.tableOff(), slot);
    e.bytes(@as(u64, p.pageBytes()), slot + 1);
}

test "the prompt twins are named in promptFor's order" {
    try std.testing.expectEqualStrings("nemotron_attention_paged_n_n", paged.prefill_names[0]);
    try std.testing.expectEqualStrings("nemotron_attention_paged_n_t", paged.prefill_names[1]);
    try std.testing.expectEqualStrings("nemotron_attention_paged_t_n", paged.prefill_names[2]);
    try std.testing.expectEqualStrings("nemotron_attention_paged_t_t", paged.prefill_names[3]);
}
