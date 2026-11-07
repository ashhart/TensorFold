//! Long synthetic indexer state checks; this internal experiment does not lift the native 2047-token admission cap.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const lanes = @import("lanes");
const Gpu = @import("gpu.zig").Gpu;
const Synth = @import("synth.zig").Synth;
const cases = @import("serve.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    {
        var plain = try Gpu.init(false);
        defer plain.deinit();
        if (plain.k.index != null) return error.KimiUnexpectedIndexPipelines;
        std.debug.print("KIMI_INDEX_LAZY PASS: plain K3 opens without experimental index pipelines\n", .{});
    }
    var gpu = try Gpu.init(true);
    defer gpu.deinit();
    const records = @import("contract").registry.all_metal;
    const ix = gpu.k.indexView();
    const pipes = .{ ix.key, ix.scores, ix.select, ix.attend, ix.merge };
    const names = .{ "k3_index_key", "k3_index_scores", "k3_index_topk", "k3_index_attend", "k3_index_merge" };
    inline for (pipes, names) |pipe, name| {
        const record = for (records) |r| {
            if (std.mem.endsWith(u8, r.name, name)) break r;
        } else return error.IndexLimitRecordMissing;
        const memory = mtl.objc.msg(usize, pipe.id, "staticThreadgroupMemoryLength", .{});
        if (memory > record.tg_memory or (!record.adapts and pipe.maxThreads() < record.threads)) return error.IndexLaunchLimit;
        std.debug.print("KIMI_INDEX_LIMIT {s} max_threads={d} threadgroup_bytes={d} declared_threads={d} adapts={}\n", .{ name, pipe.maxThreads(), memory, record.threads, record.adapts });
    }
    var source: Synth = .{ .gpa = a, .device = gpu.device, .queue = gpu.queue, .k = &gpu.k };
    defer source.deinit();
    var c: k3.config.Config = .{};
    c.layers = 1;
    c.experts = 16;
    c.vocab = 512;
    c.full_attn = .empty;
    c.full_attn.set(0);
    c.dsa = .{};
    c.experimental_index = true;
    var graph = try k3.graph.Graph.load(a, gpu.device, gpu.queue, &gpu.k, c, source.source());
    defer graph.deinit();
    source.sync();
    const model = try k3.backend.Metal.init(a, &graph, gpu.device, gpu.queue, &gpu.k, 3, 4096, source.bufs.items);
    defer model.deinit();
    var p0: [2061]u32 = undefined;
    var p1: [2051]u32 = undefined;
    for (&p0, 0..) |*v, i| v.* = @intCast(1 + (i * 7 + i / 23) % 500);
    for (&p1, 0..) |*v, i| v.* = @intCast(1 + (i * 13 + i / 31) % 500);
    const plain = try cases.tokens(a, init.io, model, &p0, null, null, null);
    defer a.free(plain);
    for (model.logits(0)) |bits| try std.testing.expect(std.math.isFinite(@as(f32, @bitCast(@as(u32, bits) << 16))));
    const solo = try cases.tokens(a, init.io, model, &p1, null, null, null);
    defer a.free(solo);
    var proposal: cases.Proposal = .{};
    const drafted = try cases.tokens(a, init.io, model, &p0, &proposal, null, null);
    defer a.free(drafted);
    try std.testing.expect(proposal.widest > 1);
    try std.testing.expectEqualSlices(u32, plain, drafted);
    try cases.together(a, init.io, model, &.{ &p0, &p1 }, &.{ plain, solo }, null);
    const short = try cases.tokens(a, init.io, model, p0[0..17], null, null, null);
    defer a.free(short);
    const crossing = try cases.tokens(a, init.io, model, p0[0..2043], null, null, null);
    defer a.free(crossing);
    try cases.together(a, init.io, model, &.{ &p0, p0[0..17], p0[0..2043] }, &.{ plain, short, crossing }, null);
    try cases.together(a, init.io, model, &.{ p0[0..2043], p0[0..17], &p0 }, &.{ crossing, short, plain }, null);
    std.debug.print("KIMI_INDEX_MIXED PASS: short, crossing and long streams equal solo in both row orders\n", .{});
    const backend = model.backend();
    for ([_]usize{ 1, 2047, 2048, 2049, 2050 }) |cut| {
        var prefix = try lanes.Stream.init(a, .{ .id = "prefix", .prompt = p0[0..cut], .max_new = 0, .drafts = false });
        defer prefix.deinit(a);
        try backend.vtable.prefill(backend.ptr, &prefix);
        const slot = try model.find(&prefix);
        const saved = try k3.snapshot.save(model, slot, prefix.prompt());
        defer saved.deinit(a);
        try std.testing.expect(saved.charge() <= k3.snapshot.bound(&c, @intCast(cut), k3.backend.window));
        backend.vtable.release(backend.ptr, &prefix);
        const resumed = try cases.tokens(a, init.io, model, &p0, null, saved, null);
        defer a.free(resumed);
        try std.testing.expectEqualSlices(u32, plain, resumed);
    }
    var base = try lanes.Stream.init(a, .{ .id = "rollback", .prompt = p0[0..2050], .max_new = 8, .drafts = false });
    defer base.deinit(a);
    try backend.vtable.prefill(backend.ptr, &base);
    const positions = [_]u64{ 2051, 2052, 2053, 2054, 2055, 2056 };
    const windows = [_]lanes.backend.Window{.{ .stream = &base, .pending = p0[2050], .held = 0, .tokens = p0[2051..2056], .parents = null, .positions = &positions }};
    var drawn: [6]u32 = undefined;
    var held: [5]u32 = undefined;
    var verified = [_]lanes.backend.Verified{.{ .sampled = &drawn, .drafts = &held }};
    try backend.vtable.verify(backend.ptr, &windows, &verified);
    try backend.vtable.keep(backend.ptr, &windows, &.{&.{ 0, 1, 2 }});
    const slot = try model.find(&base);
    const saved = try k3.snapshot.save(model, slot, p0[0..2053]);
    defer saved.deinit(a);
    backend.vtable.release(backend.ptr, &base);
    const resumed = try cases.tokens(a, init.io, model, &p0, null, saved, null);
    defer a.free(resumed);
    try std.testing.expectEqualSlices(u32, plain, resumed);
    std.debug.print("KIMI_INDEX_SYNTHETIC PASS: 2061-row prompt; drafted=plain concurrent=solo; five cuts across top-k and partial-verify restore equal fresh; plain={any} other={any}\n", .{ plain, solo });
}
