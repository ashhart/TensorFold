//! Unified prompt-cache snapshot adapter for Inkling 2 Mini drafters (DSpark and DFlash 2).
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const dspark = @import("dspark.zig");
const dspark_state = @import("dspark_state.zig");
const dflash2 = @import("dflash2.zig");
const dflash2_state = @import("dflash2_state.zig");

pub const Drafter = union(enum) {
    dspark: *dspark.Drafter,
    dflash2: *dflash2.Drafter,

    pub fn snapBytes(d: Drafter, at: usize) u64 {
        return switch (d) {
            .dspark => dspark_state.Snap.bytes(at),
            .dflash2 => dflash2_state.Snap.bytes(at),
        };
    }

    pub fn saveSnap(d: Drafter, stream: *lanes.Stream, at: u32, tap: st.Tap, tap_gen: u64) !?Snap {
        return switch (d) {
            .dspark => |ds| if (try ds.saveSnap(stream, at, tap, tap_gen)) |s| .{ .dspark = s } else null,
            .dflash2 => |df| if (try df.saveSnap(stream, at, tap, tap_gen)) |s| .{ .dflash2 = s } else null,
        };
    }

    pub fn restoreSnap(d: Drafter, stream: *lanes.Stream, snap: ?*const Snap, at: usize) !void {
        switch (d) {
            .dspark => |ds| {
                const s: ?*const dspark_state.Snap = if (snap) |dr| switch (dr.*) {
                    .dspark => |*x| x,
                    .dflash2 => return error.DSparkSnapshotMissing,
                } else null;
                try ds.restoreSnap(stream, s, at);
            },
            .dflash2 => |df| {
                const s: ?*const dflash2_state.Snap = if (snap) |dr| switch (dr.*) {
                    .dflash2 => |*x| x,
                    .dspark => return error.DFlash2SnapshotMissing,
                } else null;
                try df.restoreSnap(stream, s, at);
            },
        }
    }

    pub fn resetLane(d: Drafter, stream: *lanes.Stream) void {
        switch (d) {
            .dspark => |ds| ds.resetLane(stream),
            .dflash2 => |df| df.resetLane(stream),
        }
    }
};

pub const Snap = union(enum) {
    dspark: dspark_state.Snap,
    dflash2: dflash2_state.Snap,

    pub fn free(s: *Snap) void {
        switch (s.*) {
            .dspark => |*ds| ds.free(),
            .dflash2 => |*df| df.free(),
        }
    }

    pub fn setResident(s: *Snap, hook: ?mtl.ResidencySet.Follow) void {
        switch (s.*) {
            .dspark => |*ds| ds.setResident(hook),
            .dflash2 => |*df| df.setResident(hook),
        }
    }
};

pub const Saved = struct {
    target: snapshot.Snap,
    drafter: ?Snap = null,

    pub fn free(s: *Saved) void {
        if (s.drafter) |*dr| dr.free();
        s.target.free();
    }
};
