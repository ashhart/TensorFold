//! Analyze the real replacement fixture hooks with inert GPU images; never execute a device function.
comptime {
    const F = @import("gate").Fixture;
    _ = &F.prepare;
    _ = &F.run;
    _ = &F.replaced;
    _ = &F.check;
    _ = &F.reference;
    _ = &F.kernel;
    _ = &F.deinit;
}
