//! Saves and restores the engine views while a shared session owns its encoder.
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const Session = @import("session.zig").Session;
const Buf = fz.Buf;

pub const Binding = struct {
    model: *fz.Model,
    host: @FieldType(Engine, "host"),
    cins: [2]Buf,
    select: ?fz.Select,
    shape: @TypeOf(@as(fz.Select, undefined).q_shape),
    passed: @FieldType(Engine, "passed"),
    pub fn restore(old: Binding, e: *Engine) void {
        e.m = old.model;
        e.host = old.host;
        e.cins = old.cins;
        e.r.sel = old.select;
        e.r.shapes.put(e.r.arena, "Kc_shape", old.shape) catch unreachable;
        e.passed = old.passed;
    }
};

pub fn bind(e: *Engine, s: *Session) !Binding {
    const old: Binding = .{ .model = e.m, .host = e.host, .cins = e.cins, .select = e.r.sel, .shape = e.r.shapes.get("Kc_shape").?, .passed = e.passed };
    try e.r.shapes.put(e.r.arena, "Kc_shape", s.shape);
    e.m = &s.m;
    e.host = .{ .rows = s.m.t.rows, .mdims = s.m.t.mdims, .pos8 = s.m.t.pos8, .nk8 = s.m.t.nk8, .kvmeta = s.m.t.kvmeta, .slots = s.m.mtp.slots };
    e.cins[0] = s.m.ple.cin;
    e.r.sel = s.select[fz.CATCH - 1];
    e.passed = null;
    e.hostMode();
    return old;
}
