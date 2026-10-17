//! Persistence is separate from lesson rejection: stage writer inputs and preserve accepted live values on any error.
const std = @import("std");
const neuron = @import("neuron.zig");
pub fn save(p: anytype, t: anytype, host: anytype) !usize {
    _ = t;
    const P = @TypeOf(p.*);
    const n = p.up.items.len;
    p.save_pending = true;
    if (@hasDecl(@TypeOf(host), "saveBegin")) try host.saveBegin();
    const buffer = try p.gpa.alloc(f32, n * (if (P.swiglu) @as(usize, 3) else 2));
    defer p.gpa.free(buffer);
    const up = buffer[0..n];
    const down = buffer[n..][0..n];
    @memcpy(down, p.down.items);
    const bytes = if (P.swiglu) blk: {
        const gate = buffer[2 * n ..][0..n];
        @memcpy(gate, p.gate.items);
        neuron.upRow(P.of.neuron, p.up.items, up);
        const written = try host.pocketSaveSwiglu(p.moe.items, P.neurons, gate, up, down);
        neuron.fromUp(P.of.neuron, up, up);
        if (!same(gate, p.gate.items)) {
            p.saved_mismatch = true;
            return error.PocketSavedDifferently;
        }
        break :blk written;
    } else blk: {
        @memcpy(up, p.up.items);
        break :blk try host.pocketSave(p.moe.items, P.neurons, up, down);
    };
    if (!same(up, p.up.items) or !same(down, p.down.items)) {
        p.saved_mismatch = true;
        return error.PocketSavedDifferently;
    }
    if (@hasDecl(@TypeOf(host), "saveFinish")) try host.saveFinish();
    p.saved_mismatch = false;
    p.save_pending = false;
    p.snap.valid = false;
    return bytes;
}
fn same(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}
