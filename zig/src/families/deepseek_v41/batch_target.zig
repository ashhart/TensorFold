//! draft/iface.zig's Target over row windows: a round's segments, their picks, keeps and prefill start.

const rowtab = @import("rowtab.zig");
const iface = @import("draft/iface.zig");
const samp = @import("sampling_gpu.zig");
const batch = @import("batch.zig");
const Batch = batch.Batch;

/// Target.window: the segments as one round (several forwards past the cap); each row's greedy or keyed choice.
pub fn targetWindow(b: *Batch, segments: []const iface.Segment, choices: [][]u32, sampler: ?*samp.GpuSampler) !void {
    if (segments.len == 0 or segments.len > 16) return error.Slots;
    var mix: [16]rowtab.Seg = undefined;
    for (segments, mix[0..segments.len]) |sg, *m| {
        if (sg.parents != null) return error.TreeWindow;
        if (samp.sampled(sg.sampling) and sampler == null) return error.Sampling;
        if (sg.slot >= b.ss.n) return error.Slots;
        // a pending window must end where the segment starts (a chain never kept keeps every row)
        if (b.pend[sg.slot]) |p| if (sg.start != p.start + p.n) return error.NotAChain;
        m.* = .{ .slot = sg.slot, .start = sg.start, .rows = @intCast(sg.tokens.len) };
    }
    var ranges: [16][2]usize = undefined;
    const nf = try rowtab.pack(mix[0..segments.len], b.cap, &ranges);
    var ids: [64]u32 = undefined;
    var picks: [64]u32 = undefined;
    for (ranges[0..nf], 0..) |rg, j| {
        var n: usize = 0;
        for (segments[rg[0]..rg[1]]) |sg| {
            @memcpy(ids[n..][0..sg.tokens.len], sg.tokens);
            n += sg.tokens.len;
        }
        try b.window(mix[rg[0]..rg[1]], ids[0..n], j > 0);
        var greedy_rows = false;
        for (segments[rg[0]..rg[1]]) |sg| greedy_rows = greedy_rows or !samp.sampled(sg.sampling);
        if (greedy_rows) try b.f.greedy(picks[0..n]);
        // TF_DSV41_SAMP_BATCH: the forward's keyed segments in one sampler call (two or more), others one each
        var many: [16]samp.Seg = undefined;
        var nm: usize = 0;
        var batched: [16]bool = @splat(false);
        if (sampler) |x| if (x.batch) {
            var a: usize = 0;
            for (segments[rg[0]..rg[1]], choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
                if (samp.sampled(sg.sampling) and x.batchable(sg.sampling.?)) {
                    many[nm] = .{ .row0 = @intCast(a), .n = @intCast(sg.tokens.len), .s = sg.sampling.?, .positions = sg.draws, .out = ch[0..sg.tokens.len] };
                    batched[i] = true;
                    nm += 1;
                }
                a += sg.tokens.len;
            }
            if (nm >= 2) try x.chooseSegments(many[0..nm]) else batched = @splat(false);
        };
        var a: usize = 0;
        for (segments[rg[0]..rg[1]], choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
            const rows = ch[0..sg.tokens.len];
            if (batched[i]) {} else if (samp.sampled(sg.sampling))
                try sampler.?.choose(@intCast(a), @intCast(rows.len), sg.sampling.?, sg.draws, rows)
            else
                @memcpy(rows, picks[a..][0..rows.len]);
            a += rows.len;
        }
    }
}

/// Target.keep: a chain's rows 0..k.
pub fn keepPath(b: *Batch, slot: u32, path: []const u32) !void {
    if (slot >= b.ss.n or path.len == 0) return error.NotAChain;
    for (path, 0..) |row, i| if (row != i) return error.NotAChain;
    try b.f.engramCheck(); // a gated window whose rows failed: its picks are not committed
    return b.keep(slot, @intCast(path.len - 1));
}

/// Target.prefill's start: pending windows kept, the slot active for the one-slot prefill.
pub fn beginPrefill(b: *Batch, slot: u32) !void {
    if (slot >= b.ss.n) return error.Slots;
    try b.settle();
    try b.ss.activate(slot);
}
