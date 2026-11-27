//! One or two GLM MTP streams' tokens, consumed rows and committed states as a binary trace.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const widths = [_]usize{ 1, 2, 4, 8, 16, 32 };

fn putBytes(w: *std.Io.Writer, bytes: []const u8) !void {
    try w.writeAll(bytes);
}

fn put(w: *std.Io.Writer, value: usize) !void {
    try w.writeInt(u64, @intCast(value), .little);
}

/// One draft's observable rows and the whole committed sequence state, downloaded and written immediately.
fn record(gpu: Gpu, e: *@import("glm_cuda").engine.Engine, w: *std.Io.Writer, rows: usize, logits: u64, out: []const u32, draft: [1]u32, control: []const u8, plant: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try put(w, rows);
    try putBytes(w, std.mem.sliceAsBytes(out));
    try putBytes(w, std.mem.sliceAsBytes(&draft));
    const V = e.share.vocab.len();
    const main = try a.alloc(u16, rows * V);
    try cuda.memory.download(gpu.d, logits, std.mem.sliceAsBytes(main));
    try putBytes(w, std.mem.sliceAsBytes(main));
    const proposed = try a.alloc(u16, V);
    try cuda.memory.download(gpu.d, e.bm.logits, std.mem.sliceAsBytes(proposed));
    try putBytes(w, std.mem.sliceAsBytes(proposed));
    const partial = try a.alloc(u8, e.c.hidden * 4);
    try cuda.memory.download(gpu.d, e.bm.gathered, partial);
    if (plant and std.mem.eql(u8, control, "output")) partial[0] ^= 1;
    try putBytes(w, partial);
    const normed = try a.alloc(u8, e.c.hidden * 2);
    try cuda.memory.download(gpu.d, e.bm.fnormed, normed);
    try putBytes(w, normed);
    const st = &e.cur.st;
    try put(w, st.pos);
    try put(w, st.mtp_len);
    try put(w, st.base);
    try put(w, st.mtp_base);
    try putBytes(w, std.mem.sliceAsBytes(st.cur));
    const state = try a.alloc(u8, st.arena.buf.len);
    try st.arena.buf.download(0, state);
    if (plant and std.mem.eql(u8, control, "state")) state[0] ^= 1;
    try putBytes(w, state);
}

/// CKPT DC_JSON RANK WORLD ID_FILE STREAMS OUT [none|output|state]: set TF_GLM_MTP_LAST_ROW alike on every rank.
pub fn absorbCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8, streams: usize, out_path: []const u8, control: []const u8) !void {
    const glm = @import("glm_cuda");
    if (streams < 1 or streams > 2) return error.BadStreamCount;
    if (!std.mem.eql(u8, control, "none") and !std.mem.eql(u8, control, "output") and !std.mem.eql(u8, control, "state")) return error.BadControl;
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192 }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    glm.ranks.toolPieces(&set, &glm.ranks.environment);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = dc.capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, dc.capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed every MTP absorb window and commit", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    var f = try std.Io.Dir.cwd().createFile(gpu.io, out_path, .{});
    defer f.close(gpu.io);
    var buf: [1 << 16]u8 = undefined;
    var file = f.writer(gpu.io, &buf);
    const w = &file.interface;
    try put(w, streams);
    var compared: usize = 0;
    for (0..streams) |s| {
        const len = if (streams == 1) 3000 else 600 + 2400 * s; // dense, and past the dense/sparse boundary
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*id, j| id.* = @intCast(100 + (j * 7919 + s * 31) % 5000);
        try x.bind(@intCast(s));
        try x.begin(null);
        const first = try x.prefill(ids, null, null, null);
        try putBytes(w, std.mem.asBytes(&first));
    }
    for (widths, 0..) |rows, round| {
        var ids: [2][32]u32 = undefined;
        var outs: [2][32]u32 = undefined;
        var drafts: [2][1]u32 = undefined;
        for (0..streams) |s| for (0..rows) |r| {
            ids[s][r] = @intCast(100 + (round * 31 + s * 7 + r * 13) % 5000);
        };
        if (streams == 2) {
            var parts: [2]glm.engine.Part = undefined;
            for (0..streams) |s| parts[s] = .{ .slot = @intCast(s), .ids = ids[s][0..rows], .out = outs[s][0..rows] };
            try x.verifyShared(parts[0..streams]);
        } else {
            try x.verify(ids[0][0..rows], outs[0][0..rows], 0);
        }
        for (0..streams) |s| {
            try x.bind(@intCast(s));
            const logits = if (streams == 2) e.bs.?.logits + s * rows * e.share.vocab.len() * 2 else e.bd.logits;
            try x.keep(rows);
            try x.draft(false, outs[s][0..rows], 1, &drafts[s], null);
            try record(gpu, e, w, rows, logits, outs[s][0..rows], drafts[s], control, round == 0 and s == 0);
            compared += rows;
        }
    }
    try w.flush();
    check.pass("MTP absorption at TP={d}: {d} stream(s), {d} accepted rows at widths 1 through 32; tokens, bf16 logits, fp32 consumed rows, final norms and every committed state byte in {s} ({s} control)", .{ world, streams, compared, out_path, control });
}
