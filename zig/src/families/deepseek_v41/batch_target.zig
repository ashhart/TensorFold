//! draft/iface.zig's Target over row windows: a round's segments, their picks, keeps and prefill start.

const lanes = @import("lanes");
const rowtab = @import("rowtab.zig");
const iface = @import("draft/iface.zig");
const samp = @import("sampling_gpu.zig");
const branches = @import("draft/branches.zig");
const batch = @import("batch.zig");
const Batch = batch.Batch;

/// The tree resolver's chains context: the batch plus the round's keyed sampler (chainRun needs both).
const ChainCtx = struct { b: *Batch, sampler: ?*samp.GpuSampler };

fn chainSelf(p: *anyopaque) *ChainCtx {
    return @ptrCast(@alignCast(p));
}

/// The tree resolver's chains: a single slot's chain window through the batch's row path, dropped on every rank.
const chains: branches.Chains.VTable = .{ .run = chainRun, .drop = chainDrop };

fn chainRun(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, sampling: ?lanes.Sampling, out: []u32) anyerror!void {
    const c = chainSelf(p);
    const b = c.b;
    // a pending window on the slot: keep it whole (as the one-slot window does), then the chain runs from its end
    if (b.pend[slot]) |w| {
        if (start != w.start + w.n) return error.NotAChain;
        try b.keep(slot, w.n - 1);
    }
    // the slot's chain alone, `more` so the other slots' pending taps are stashed before this forward overwrites them
    try b.runSlot(slot, start, tokens, true);
    if (samp.sampled(sampling)) {
        const smp = c.sampler orelse return error.Sampling;
        try smp.choose(0, @intCast(tokens.len), sampling.?, draws, out);
    } else {
        var picks: [64]u32 = undefined;
        try b.f.greedy(picks[0..tokens.len]);
        @memcpy(out, picks[0..tokens.len]);
    }
}

fn chainDrop(p: *anyopaque, slot: u32) anyerror!void {
    const c = chainSelf(p);
    return c.b.drop(slot);
}

/// Target.window: the segments as one round (several forwards past the cap); each row's greedy or keyed choice.
pub fn targetWindow(b: *Batch, segments: []const iface.Segment, choices: [][]u32, sampler: ?*samp.GpuSampler) !void {
    if (segments.len == 0 or segments.len > 16) return error.Slots;
    // flat chains and draft trees (`parents`) split: trees resolve per-slot through branches.Resolver, chains batch.
    var mix: [16]rowtab.Seg = undefined;
    var flat: [16]iface.Segment = undefined;
    var flat_choices: [16][]u32 = undefined;
    var nflat: usize = 0;
    for (segments, choices) |sg, ch| {
        if (sg.parents != null) continue;
        if (samp.sampled(sg.sampling) and sampler == null) return error.Sampling;
        if (sg.slot >= b.ss.n) return error.Slots;
        // a pending window must end where the segment starts (a chain never kept keeps every row)
        if (b.pend[sg.slot]) |p| if (sg.start != p.start + p.n) return error.NotAChain;
        b.tree[sg.slot].clear();
        mix[nflat] = .{ .slot = sg.slot, .start = sg.start, .rows = @intCast(sg.tokens.len) };
        flat[nflat] = sg;
        flat_choices[nflat] = ch;
        nflat += 1;
    }
    // the chains first (their first forward settles the round), then the trees, so a tree's lazy sibling chains
    // stash the chains' taps as they overwrite the shared window.
    if (nflat > 0) {
        var ranges: [16][2]usize = undefined;
        const nf = try rowtab.pack(mix[0..nflat], b.cap, &ranges);
        var ids: [64]u32 = undefined;
        var picks: [64]u32 = undefined;
        for (ranges[0..nf], 0..) |rg, j| {
            var n: usize = 0;
            for (flat[rg[0]..rg[1]]) |sg| {
                @memcpy(ids[n..][0..sg.tokens.len], sg.tokens);
                n += sg.tokens.len;
            }
            try b.window(mix[rg[0]..rg[1]], ids[0..n], j > 0);
            var greedy_rows = false;
            for (flat[rg[0]..rg[1]]) |sg| greedy_rows = greedy_rows or !samp.sampled(sg.sampling);
            if (greedy_rows) try b.f.greedy(picks[0..n]);
            // TF_DSV41_SAMP_BATCH: the forward's keyed segments in one sampler call (two or more), others one each
            var many: [16]samp.Seg = undefined;
            var nm: usize = 0;
            var batched: [16]bool = @splat(false);
            if (sampler) |x| if (x.batch) {
                var a: usize = 0;
                for (flat[rg[0]..rg[1]], flat_choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
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
            for (flat[rg[0]..rg[1]], flat_choices[rg[0]..rg[1]], 0..) |sg, ch, i| {
                const rows = ch[0..sg.tokens.len];
                if (batched[i]) {} else if (samp.sampled(sg.sampling))
                    try sampler.?.choose(@intCast(a), @intCast(rows.len), sg.sampling.?, sg.draws, rows)
                else
                    @memcpy(rows, picks[a..][0..rows.len]);
                a += rows.len;
            }
        }
    }
    var ctx = ChainCtx{ .b = b, .sampler = sampler };
    for (segments, choices) |sg, ch| {
        if (sg.parents == null) continue;
        if (sg.slot >= b.ss.n) return error.Slots;
        try b.tree[sg.slot].window(.{ .ptr = &ctx, .vtable = &chains }, sg, ch);
    }
}

/// Target.keep: a tree's accepted path through its resolver's map, else a chain's rows 0..k.
pub fn keepPath(b: *Batch, slot: u32, path: []const u32) !void {
    if (slot >= b.ss.n or path.len == 0) return error.NotAChain;
    try b.f.engramCheck(); // a gated window whose rows failed: its picks are not committed
    if (b.tree[slot].live) return b.keep(slot, try b.tree[slot].keep(path));
    for (path, 0..) |row, i| if (row != i) return error.NotAChain;
    return b.keep(slot, @intCast(path.len - 1));
}

/// Target.prefill's start: pending windows kept, the slot active for the one-slot prefill.
pub fn beginPrefill(b: *Batch, slot: u32) !void {
    if (slot >= b.ss.n) return error.Slots;
    b.tree[slot].clear();
    try b.settle();
    try b.ss.activate(slot);
}
