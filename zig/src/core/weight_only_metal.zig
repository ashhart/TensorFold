//! Backend projection API over the shared launch adapter.
const A = @import("core").weight_only_metal_adapter(@import("core").weight_only, @import("metal"));
pub const Control = A.Control;
pub const View = A.View;
pub const Kernels = A.Kernels;
pub const names = A.names;
pub const source = A.source;
