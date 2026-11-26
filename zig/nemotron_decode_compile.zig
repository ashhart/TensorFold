//! Compile the real verify callback and device contract bodies without linking or running Metal.
const tf = @import("tensorfold");
const probe = @import("probe").attention;
const experts = @import("probe").experts;
export fn compileVerify(b: *tf.nemotron.backend.Metal) usize {
    return @intFromPtr(b.backend().vtable.verify);
}
export fn compileContract(p: *probe.Probe) usize {
    const vt = p.kernel().vt;
    return @intFromPtr(vt.prepare) ^ @intFromPtr(vt.run) ^ @intFromPtr(vt.reference) ^
        @intFromPtr(vt.replaced.?) ^ @intFromPtr(vt.simulate.?) ^ @intFromPtr(&probe.Probe.init);
}

export fn compileExpert(p: *experts.Probe, s: *experts.Suite) usize {
    const vt = p.kernel().vt;
    return @intFromPtr(vt.prepare) ^ @intFromPtr(vt.run) ^ @intFromPtr(vt.replaced.?) ^ @intFromPtr(&experts.Suite.collect) ^ @intFromPtr(&experts.Suite.addPlain) ^ @intFromPtr(s);
}

export fn compileDecodeSuite(s: *probe.Suite) void {
    s.collect("nemotron verify attention") catch {};
}
