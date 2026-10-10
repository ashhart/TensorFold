//! What a training step does, for both backends' trainers.

/// A step's loss, with its gradient (and Adam: `learn`), or only its site inputs read (`reads`).
pub const Mode = enum { loss, grad, learn, avoid, seek, project };

/// Whether a step only reads the site inputs (sketches or projections), with no loss.
pub fn reads(mode: Mode) bool {
    return switch (mode) {
        .avoid, .seek, .project => true,
        else => false,
    };
}

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };
