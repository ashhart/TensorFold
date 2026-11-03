//! The family binds shared native launch validation to its owning shader source.
const core = @import("core");
pub fn Kernels(comptime Backend: type) type {
    return Adapter(Backend, core.living_rows);
}
pub fn Adapter(comptime Backend: type, comptime spec: type) type {
    return core.living_rows_cuda.Adapter(Backend, spec);
}
