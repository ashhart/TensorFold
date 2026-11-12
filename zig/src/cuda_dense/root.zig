//! The shared CUDA dense-decoder engine: a recipe's layers over paged GQA attention, lanes, draws and the host decls.
pub const recipe = @import("recipe.zig");
pub const pages = @import("pages.zig");
pub const rope = @import("rope.zig");
pub const kernels = @import("kernels.zig");
pub const model = @import("model.zig");
pub const pass = @import("pass.zig");
pub const draws = @import("draws.zig");
pub const backend = @import("backend.zig");
pub const native = @import("native.zig");

pub const Native = native.Native;
pub const Pass = pass.Pass;
pub const Geometry = recipe.Geometry;
pub const check = recipe.check;

test {
    _ = recipe;
    _ = pages;
    _ = rope;
    _ = kernels;
    _ = model;
    _ = pass;
    _ = draws;
    _ = backend;
    _ = native;
    _ = @import("native_test.zig");
    _ = @import("put_test.zig");
    _ = @import("device_fail_test.zig");
}
