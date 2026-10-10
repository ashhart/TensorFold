//! What a training step does, for both backends' trainers.

/// A step's loss, with its gradient (and Adam: `learn`), or only its site inputs read (`reads`: sketched, projected,
/// or the MoE layers' shared-expert activations kept as keys); `target`: the loss's gradient at a nudged residual row.
pub const Mode = enum { loss, grad, learn, avoid, seek, project, keys, target };

/// Whether a step only reads the site inputs (sketches, projections or keys), with no loss.
pub fn reads(mode: Mode) bool {
    return switch (mode) {
        .avoid, .seek, .project, .keys => true,
        else => false,
    };
}

/// Where a `target` step adds its nudges and reads the loss's gradient: for each of three rows, the residual after each
/// layer in its mask (a bit a layer), row i's in the buffers' i-th third.
pub const Nudge = struct { layers: [3]u64 = @splat(0), rows: [3]usize = @splat(0) };

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };
