//! Shared windows use the legacy engine's GPU sampler with each stream's absolute draw positions.
const lanes = @import("lanes");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const sample = @import("sample.zig");

pub fn active(params: ?lanes.Sampling) ?lanes.Sampling {
    const p = params orelse return null;
    return if (p.temperature > 0) p else null;
}

/// Buffer offsets isolate each window; its positions must be filled before the command buffer is committed.
pub fn encode(enc: mtl.ComputeEncoder, d: *const sample.Draws, logits: fz.Buf, positions: fz.Buf, picks: fz.Buf, row: usize, rows: usize, p: lanes.Sampling) void {
    sample.draw(enc, d, logits.at(row * fz.VOCAB * 2), positions.at(row * 4), picks.at(row * 4), rows, fz.VOCAB, p);
}

/// Every sampled window's rows in one dispatch a run of adjacent sampled windows (greedy rows keep their argmax picks).
pub fn encodeAll(enc: mtl.ComputeEncoder, d: *const sample.Draws, logits: fz.Buf, positions: fz.Buf, picks: fz.Buf, starts: []const usize, widths: []const usize, ps: []const ?lanes.Sampling) void {
    var each: [fz.MAXR]lanes.Sampling = undefined;
    var n: usize = 0; // rows in the open run
    var at: usize = 0; // the open run's first row
    for (starts, widths, ps) |row, rows, p| {
        const s = active(p) orelse {
            if (n > 0) sample.drawRows(enc, d, logits.at(at * fz.VOCAB * 2), positions.at(at * 4), picks.at(at * 4), each[0..n], fz.VOCAB);
            n = 0;
            continue;
        };
        if (n == 0) at = row;
        for (each[n..][0..rows]) |*x| x.* = s;
        n += rows;
    }
    if (n > 0) sample.drawRows(enc, d, logits.at(at * fz.VOCAB * 2), positions.at(at * 4), picks.at(at * 4), each[0..n], fz.VOCAB);
}

/// The last prompt row's pick, before another prompt or draft overwrites its logits.
pub fn first(e: *Engine, model: *fz.Model, p: lanes.Sampling, position: u32) !u32 {
    e.draw_at.b.slice(u32, fz.MAXR)[0] = position;
    const cb = e.r.queue.commandBuffer();
    e.r.begin(cb, .serial);
    errdefer if (e.r.fsr != null) e.r.abort();
    encode(e.r.enc, &e.r.draws, model.t.logits, e.draw_at, model.t.picks, 0, 1, p);
    try model.finish(cb);
    return model.t.picks.b.slice(u32, 1)[0];
}
