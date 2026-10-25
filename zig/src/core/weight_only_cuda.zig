//! Backend projection API over the shared launch adapter.
const A = @import("core").weight_only_cuda_adapter(@import("core").weight_only, @import("cuda"));
pub const Control = A.Control;
pub const View = A.View;
pub const Kernels = A.Kernels;
pub const names = A.names;
