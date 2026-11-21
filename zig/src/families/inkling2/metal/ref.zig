//! Shared Metal buffer reference for Inkling 2 launchers and their state buffers.
const mtl = @import("metal");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
