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
pub fn encode(enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, logits: fz.Buf, positions: fz.Buf, picks: fz.Buf, row: usize, rows: usize, p: lanes.Sampling) void {
    sample.draw(enc, pipe, logits.at(row * fz.VOCAB * 2), positions.at(row * 4), picks.at(row * 4), rows, fz.VOCAB, p);
}

/// The last prompt row's pick, before another prompt or draft overwrites its logits.
pub fn first(e: *Engine, model: *fz.Model, p: lanes.Sampling, position: u32) !u32 {
    e.draw_at.b.slice(u32, fz.MAXR)[0] = position;
    const cb = e.r.queue.commandBuffer();
    e.r.enc = cb.compute(.serial);
    encode(e.r.enc, e.r.draw_pipe, model.t.logits, e.draw_at, model.t.picks, 0, 1, p);
    try model.finish(cb);
    return model.t.picks.b.slice(u32, 1)[0];
}
