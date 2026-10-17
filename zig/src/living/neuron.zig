//! A pocket neuron's activation for each family's kind, and the gate and up rows a SwiGLU neuron is written as.
const std = @import("std");
const Neuron = @import("family.zig").Neuron;

pub fn sigmoid(y: f32) f32 {
    return 1 / (1 + @exp(-y));
}

pub fn silu(y: f32) f32 {
    return y * sigmoid(y);
}

/// The activation of a neuron of kind `n` at input x: `z` is u x (u as fitted), `zg` its gate row's g x (SwiGLU).
pub fn act(n: Neuron, z: f32, zg: f32) f32 {
    return switch (n.act) {
        .relu2 => if (z > 0) z * z else 0,
        .swiglu => silu(zg) * (z / n.sharp), // its up row is u / sharp
    };
}

/// A SwiGLU neuron's gate row from its fitted input u: sharp u.
pub fn gateRow(n: Neuron, u: []const f32, out: []f32) void {
    for (out, u) |*o, x| o.* = n.sharp * x;
}

/// The up row a SwiGLU neuron is written as, from u: u / sharp (exact for a power of two).
pub fn upRow(n: Neuron, u: []const f32, out: []f32) void {
    for (out, u) |*o, x| o.* = x / n.sharp;
}

/// u back from an up row as written: sharp w.
pub fn fromUp(n: Neuron, w: []const f32, out: []f32) void {
    for (out, w) |*o, x| o.* = n.sharp * x;
}

test "a SwiGLU neuron written as sharp u and u / sharp fires as relu^2 on its rows and stays near zero past the margin" {
    const n: Neuron = .{ .act = .swiglu, .margin = 0.5, .sharp = 16 };
    const r: Neuron = .{ .act = .relu2 };
    // z = u x; the gate row reads sharp z, and the up row's z / sharp times silu is what the forward adds
    for ([_]f32{ 1, 1.5, 3 }) |z| {
        try std.testing.expectApproxEqRel(act(r, z, 0), act(n, z, n.sharp * z), 1e-6);
    }
    var most: f32 = 0;
    var z: f32 = -n.margin;
    while (z > -20) : (z -= 0.01) most = @max(most, @abs(act(n, z, n.sharp * z)));
    try std.testing.expect(most < n.margin * n.margin * sigmoid(-n.sharp * n.margin) * 1.01);
    try std.testing.expect(most < 1e-4);
    // the up row as written gives u back exactly
    const u = [_]f32{ 0.123, -7.5e-3, 3.25 };
    var w: [3]f32 = undefined;
    var back: [3]f32 = undefined;
    upRow(n, &u, &w);
    fromUp(n, &w, &back);
    try std.testing.expectEqualSlices(f32, &u, &back);
}
