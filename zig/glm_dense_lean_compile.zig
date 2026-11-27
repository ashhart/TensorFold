//! Runtime-parameter entry points force analysis of device contract bodies without loading or running them.
const tests = @import("tests/cuda/contract_glm_dense_lean.zig");
const spec = @import("core").contract.spec;
export fn densePrepare(p: *tests.Product, shape: *const spec.Shape, fill: u8, seed: u64) void {
    _ = p.prepare(shape.*, @fromBackingInt(@intCast(fill % 3)), seed) catch false;
}
export fn denseRun(p: *tests.Product) void {
    p.run() catch {};
}
export fn denseCheck(p: *tests.Product) void {
    _ = p.check(.shapes) catch null;
}
export fn denseGates(products: *[4]tests.Product) void {
    tests.runPrepared(products) catch {};
}
export fn denseProfile(p: *tests.Product, enabled: bool) void {
    tests.profilePrepared(p, enabled, "dense-up") catch {};
}
