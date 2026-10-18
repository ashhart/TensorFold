//! Compile the actual GLM restoration entry, including its default drained pool mover; never run this object.
const Backend = @import("families/glm/backend.zig").Backend;
const Stream = @import("lanes").Stream;

export fn glmRestoreCompileOnly(b: *Backend, s: *Stream) u8 {
    b.backend().prefill(s) catch return 1;
    return 0;
}
