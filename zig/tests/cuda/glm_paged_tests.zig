//! GLM-5.3-Flash's KV pages on CUDA: a paged stream against its twin on a whole window, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lanes_tests = @import("glm_lanes_tests.zig");
const Gpu = check.Gpu;

/// Prompts at a page's edges (512 rows), a prompt chunk's (2,048), past the dense pass, and at the served lengths.
const lengths = [_]usize{ 511, 512, 513, 2047, 2048, 2049, 5000, 11_220, 16_810, 30_001 };
/// Past this, a prompt goes in the lanes' steps (Leader.chunk): its chunks' indexer rows split from TF_GLM_INDEX_SPLIT.
const stepped = 5000;
/// A window of 32,768 and a verify window, as cuda_native.capacity makes it.
const capacity = 32_832;
const paged_slot = 0; // the engine's own sequence, on pages from its init
const shuffled = 1; // takes and gives back pages between prompts, so slot 0's come out of the free list's order
const twin_slot = 3; // past the paged sequences: a whole window, as without TF_KV_PAGES

/// `glm-paged-check`: pages out of order, NaN-filled; prompts, windows across page edges, keeps, drafts; owner classes.
pub fn pagedCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try lanes_tests.openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.pages = true; // every rank's engine makes its own sequence on pages, as TF_KV_PAGES does
    set.kv_split = glm.start.splitting(set.kv_split, @intCast(world), true, set.kv_host != 0) catch |err| {
        std.debug.print("REFUSE {s}\n", .{@errorName(err)});
        return err;
    };
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} made rank 0's pool and mapped its pages through every step", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const P = glm.pages;
    const l = try P.layout(gpu.gpa, e, 0);
    const classes: u32 = l.classes; // TF_GLM_KV_SPLIT: an owner class a rank, each two windows and a page over
    const count = if (classes > 1) classes * (2 * P.tableLen(capacity) + 2) else 2 * P.tableLen(capacity) + 3; // two windows and more
    try x.pool(2, count, l.pageBytes());
    e.pages = try P.Pages.init(gpu.gpa, e, l, 2, count, true);
    const pages = e.pages.?;
    pages.poison = true; // a row read before its write turns the reply to NaN
    const V = e.share.vocab.len();
    const mine = try a.alloc(u16, 4 * V);
    const twin = try a.alloc(u16, 4 * V);
    var failed = false;
    var unordered: usize = 0;
    var compared: usize = 0;
    for (lengths, 0..) |len, k| {
        if (k % 2 == 0) {
            try x.bind(shuffled);
            _ = try pages.openOn(shuffled, (k + 1) * 512 - 100, 0, @intCast(k % classes)); // the paged slot's class
        } else x.release(shuffled) catch {};
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, i| u.* = @intCast(100 + (i * 7919 + k * 31) % 5000);
        const held = try pages.openOn(paged_slot, len, 64, @intCast(k % classes));
        for (held[1..], held[0 .. held.len - 1]) |b, prev| unordered += @intFromBool(b != prev + classes);
        if (classes > 1) std.debug.print("  a {d}-token prompt on owner {d}\n", .{ len, k % classes });
        if (classes > 1) failed = !try ownerTable(gpu, e, held) or failed;
        var first: [2]u32 = undefined;
        var drafts: [2][glm.engine.max_drafts]u32 = undefined;
        for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
            try x.bind(slot);
            if (slot == paged_slot) try x.beginOn(null, held, 0) else try x.begin(null);
            first[j] = if (len > stepped) try chunked(x, ids) else try x.prefill(ids, null, null, null);
            try cuda.memory.download(gpu.d, e.bp.logits, std.mem.sliceAsBytes(if (j == 0) mine else twin)[0 .. V * 2]);
            try x.draft(true, &.{first[j]}, 2, &drafts[j], null);
        }
        if (first[0] != first[1] or !std.mem.eql(u16, mine[0..V], twin[0..V]) or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
            std.debug.print("FAIL paged: a {d}-token prompt's first token {d}, logits or drafts differ from its twin's ({d})\n", .{ len, first[0], first[1] });
            failed = true;
        }
        for (0..6) |round| { // windows of 1 to 4 rows from the prompt's end, across its page's edge
            const rows = 1 + round % 4;
            var w: [4]u32 = undefined;
            for (w[0..rows], 0..) |*u, r| u.* = @intCast(100 + (round * 31 + k * 7 + r * 13) % 5000);
            var outs: [2][4]u32 = undefined;
            for ([_]u32{ paged_slot, twin_slot }, 0..) |slot, j| {
                try x.bind(slot);
                try x.verify(w[0..rows], outs[j][0..rows], 0);
                try cuda.memory.download(gpu.d, e.bd.logits, std.mem.sliceAsBytes(if (j == 0) mine else twin)[0 .. rows * V * 2]);
                const keep = 1 + (round + k) % rows;
                try x.keep(keep);
                try x.draft(false, outs[j][0..keep], 2, &drafts[j], null);
            }
            if (!std.mem.eql(u16, mine[0 .. rows * V], twin[0 .. rows * V]) or !std.mem.eql(u32, outs[0][0..rows], outs[1][0..rows]) or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
                std.debug.print("FAIL paged: a {d}-token prompt's window {d} ({d} rows) differs from its twin's\n", .{ len, round, rows });
                failed = true;
            }
            compared += rows;
        }
        try x.release(paged_slot);
        try x.release(twin_slot);
    }
    x.release(shuffled) catch {};
    const c = pages.pool.census();
    if (!pages.pool.idle() or c.free != c.total) {
        std.debug.print("FAIL paged: the pool is not idle after every stream ended (free {d} of {d}, promised {d})\n", .{ c.free, c.total, c.promised });
        failed = true;
    }
    for (0..classes) |o| if (pages.pool.classFree(@intCast(o)) != pages.pool.classCap() or pages.pool.classPromised(@intCast(o)) != 0) {
        std.debug.print("FAIL paged: owner class {d} is not idle (free {d} of {d})\n", .{ o, pages.pool.classFree(@intCast(o)), pages.pool.classCap() });
        failed = true;
    };
    if (unordered == 0) {
        std.debug.print("FAIL paged: every stream's pages came in order; the check did not reach a shuffled table\n", .{});
        failed = true;
    }
    const owned = if (e.room) |r| r.stats else glm.owner_attn.Stats{};
    if (classes > 1 and (owned.layers == 0 or owned.rows == 0)) {
        std.debug.print("FAIL paged: no layer attended owner-computed; the check did not reach a stream's owner-only rows\n", .{});
        failed = true;
    }
    if (classes > 1 and owned.fetches == 0) {
        std.debug.print("FAIL paged: no prompt chunk fetched its history; the check did not reach a chunk past a stream's first\n", .{});
        failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("pages at TP={d}{s}: on the engine's own sequence, {d} prompts at the pages' edges and past them (511 to 30,001 tokens) and {d} window rows equal their twins' logits, tokens and drafts byte for byte, pages out of order ({d} jumps) and NaN before use; the pool idle after", .{ world, if (classes > 1) " on owner classes (TF_GLM_KV_SPLIT), each stream's latent rows on its owner alone" else "", lengths.len, compared, unordered });
    if (classes > 1) std.debug.print("  owner-computed: {d} layers, {d} rows, {d} bytes sent by rank 0\n", .{ owned.layers, owned.rows, owned.bytes });
    if (classes > 1) std.debug.print("  history fetched: {d} layers, {d} rows, {d} bytes on rank 0\n", .{ owned.fetches, owned.fetched, owned.fetch_bytes });
}

/// A prompt in the lanes' steps, a chunk at a time as served: the first token.
fn chunked(x: anytype, ids: []const u32) !u32 {
    var start: usize = 0;
    while (true) {
        const c = try x.chunk(ids, start, false, null, null, null);
        if (c.first) |t| return t;
        start += c.rows;
    }
}

/// TF_GLM_KV_SPLIT: rank 0's latent table for the paged stream holds its region's places, or the trash page alone.
fn ownerTable(gpu: Gpu, e: anytype, held: []const u32) !bool {
    const st = &e.seqs.items[paged_slot].?.st;
    const got = try gpu.gpa.alloc(u32, held.len);
    defer gpu.gpa.free(got);
    try cuda.memory.download(gpu.d, st.ltable, std.mem.sliceAsBytes(got));
    const region = e.pages.?.region();
    for (held, got) |id, local| if (local != region.entry(id) or (st.owner.? != 0) != (local == region.cap)) {
        std.debug.print("FAIL paged: rank 0's latent table maps page {d} of owner {d} to {d}\n", .{ id, st.owner.?, local });
        return false;
    };
    return true;
}
