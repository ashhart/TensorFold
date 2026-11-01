//! GLM-5.3-Flash DFlash2 prompt taps at TP>1: split and ring retained taps against the gather path, byte for byte.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Variant = struct { name: []const u8, split: bool, ring: bool };

/// The first retained row that differs in any loaded tap slot, or null; a row is layers.len tap slots of `hidden2`.
fn firstDiffRow(got: []const u8, want: []const u8, rows: usize, tap_row: usize, hidden2: usize, slots: []const usize) ?usize {
    for (0..rows) |r| for (slots) |k| {
        const off = r * tap_row + k * hidden2;
        if (!std.mem.eql(u8, got[off..][0..hidden2], want[off..][0..hidden2])) return r;
    };
    return null;
}

/// `glm-tap-split-check`: every rank's retained DFlash2 prompt taps are equal under split, ring and gather.
pub fn tapSplitCheck(gpu: Gpu, checkpoint: []const u8, drafter: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    try check.expect(world > 1, "glm-tap-split-check needs two ranks or more", .{});
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const WIN: usize = @intCast(glm.dflash.WIN);
    const C = glm.engine.prompt_rows;
    const lens = [_]usize{ WIN - 1, WIN, WIN + 1, 8000 }; // 2,047 / 2,048 / 2,049, then about 8k in several chunks
    const capacity = lens[lens.len - 1] + 4 * WIN;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set, .drafter = true });
    defer e.deinit();
    e.drafter = try glm.dflash_load.load(gpu.gpa, gpu.io, e, drafter, @intCast(rank), @intCast(world), 4);
    const dr = e.drafter.?;
    const TAP_ROW = glm.dflash.TAP_ROW;
    const hidden2 = glm.taps.hidden * 2;
    // the tap slots whose target layer is loaded: slots 0..4 read layers 5, 14, 24, 33, 42 (DFlash2's target_layer_ids)
    var slots: [glm.taps.layers.len]usize = undefined;
    var ns: usize = 0;
    for (glm.taps.layers, 0..) |layer, k| if (layer < @as(usize, e.c.run)) {
        slots[ns] = k;
        ns += 1;
    };
    try check.expect(ns > 0, "no tapped layer is loaded: TF_GLM_LAYERS must reach the first tap at layer {d}", .{glm.taps.layers[0]});
    const variants = [_]Variant{ .{ .name = "gather", .split = false, .ring = false }, .{ .name = "split", .split = true, .ring = false }, .{ .name = "ring", .split = true, .ring = true } };
    var failed = false;
    for (lens) |len| {
        var pa = std.heap.ArenaAllocator.init(gpu.gpa);
        defer pa.deinit();
        const a = pa.allocator();
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
        const keep_from = len -| WIN;
        var gather = std.ArrayList([]u8).empty; // one host copy of the retained taps a tapped chunk, the gather baseline
        for (variants) |v| {
            e.settings.split = v.split;
            e.settings.ring = v.ring;
            try e.bind(0);
            try e.begin(null);
            var at: usize = 0;
            var chunk: usize = 0;
            while (true) : (at += C) {
                const rows = @min(len - at, C);
                const part = ids[at..@min(len, at + C + 1)];
                const done = e.prefillChunk(part, at, len, null, null, null) catch |err| {
                    std.debug.print("FAIL tap: {s} prompt {d} chunk at {d}: {s}\n", .{ v.name, len, at, @errorName(err) });
                    return err;
                };
                if (at + rows > keep_from) { // tapped: the engine leaves f.taps set only for chunks the window reaches
                    try e.stream.synchronize();
                    const lo = keep_from -| at; // the first retained row of this chunk (dflash.promptRun)
                    const n = rows - lo;
                    const host = try gpu.gpa.alloc(u8, n * TAP_ROW);
                    defer gpu.gpa.free(host);
                    try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(host.ptr, dr.taps_prefill + lo * TAP_ROW, n * TAP_ROW), "cuMemcpyDtoH");
                    if (v.split or v.ring) {
                        if (firstDiffRow(host, gather.items[chunk], n, TAP_ROW, hidden2, slots[0..ns])) |bad| {
                            std.debug.print("FAIL tap: {s} prompt {d} rank {d}: retained row {d} differs from gather (chunk at {d})\n", .{ v.name, len, rank, at + lo + bad, at });
                            failed = true;
                        }
                    } else try gather.append(a, try a.dupe(u8, host));
                    chunk += 1;
                }
                if (done != null) break;
            }
        }
    }
    if (failed) return error.TestFailed;
    check.pass("tap split at TP={d} rank {d}: split and ring retained taps equal gather for prompts of {any} rows, tap slots {any}", .{ world, rank, lens, slots[0..ns] });
}
