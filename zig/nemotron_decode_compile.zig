//! Compile the real verify callback and device contract bodies without linking or running Metal.
const tf = @import("tensorfold");
const probe = @import("probe");
export fn compileVerify(b: *tf.nemotron.backend.Metal) usize {
    return @intFromPtr(b.backend().vtable.verify);
}
export fn compileContract(p: *probe.Probe) usize {
    const vt = p.kernel().vt;
    return @intFromPtr(vt.prepare) ^ @intFromPtr(vt.run) ^ @intFromPtr(vt.reference) ^
        @intFromPtr(vt.replaced.?) ^ @intFromPtr(vt.simulate.?) ^ @intFromPtr(&probe.Probe.init);
}
