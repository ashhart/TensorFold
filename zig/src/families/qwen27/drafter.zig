//! The 27B's drafter: DFlash2 or DSpine as its checkpoint's config declares, behind Generation's interface.
const std = @import("std");
const core = @import("core");
const df = @import("dflash/runtime_model.zig");
const ds = @import("dspine/runtime.zig");
const Target = @import("model.zig").Model;
const Tree = @import("dflash/selector.zig").Tree;
const Ring = @import("tap_ring.zig").Ring;
pub const Ref = core.draft_ops.Ref;

pub const Kind = enum { dflash2, dspine };

/// Whether the drafter checkpoint at `dir` declares DSpine (else it is read as DFlash2).
pub fn kindOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Kind {
    const text = try readConfig(a, io, dir, 1 << 20);
    defer a.free(text);
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    return if (core.dspine.declared(doc.value)) .dspine else .dflash2;
}

pub const Any = union(Kind) {
    dflash2: *df.Model,
    dspine: *ds.Model,

    /// The drafter at `dir` for `target` (whose files `target_dir` holds: DSpine reads its LM head again).
    pub fn load(a: std.mem.Allocator, io: std.Io, target: *Target, dir: []const u8, bits: u8, target_dir: []const u8) !Any {
        return switch (try kindOf(a, io, dir)) {
            .dflash2 => .{ .dflash2 = try df.Model.load(a, io, target, dir, if (bits == 4) .prepared_q4_reference else .bf16_reference) },
            .dspine => .{ .dspine = try ds.Model.load(a, io, target, dir, if (bits == 4) .q4 else .bf16, target_dir) },
        };
    }
    pub fn deinit(d: Any) void {
        switch (d) {
            inline else => |m| m.deinit(),
        }
    }
    /// The name the host reports for this drafter.
    pub fn label(d: Any) []const u8 {
        return switch (d) {
            .dflash2 => "DFlash2",
            .dspine => "DSpine",
        };
    }
    /// DFlash2's model, for what only it does: parked contexts, batched trees and priced selection.
    pub fn dflash(d: Any) ?*df.Model {
        return switch (d) {
            .dflash2 => |m| m,
            .dspine => null,
        };
    }
    pub fn binds(d: *const Any, target: *const Target) bool {
        return switch (d.*) {
            inline else => |m| m.binds(target),
        };
    }
    pub fn tapIds(d: *const Any) []const u32 {
        return switch (d.*) {
            inline else => |m| m.tapIds(),
        };
    }
    pub fn tapWidth(d: *const Any) usize {
        return switch (d.*) {
            inline else => |m| m.tapWidth(),
        };
    }
    pub fn failed(d: *const Any) bool {
        return switch (d.*) {
            inline else => |m| m.failed(),
        };
    }
    pub fn idle(d: *const Any) bool {
        return switch (d.*) {
            inline else => |m| m.idle(),
        };
    }
    pub fn committedEnd(d: *const Any) u64 {
        return switch (d.*) {
            inline else => |m| m.committedEnd(),
        };
    }
    pub fn absorb(d: *Any, taps: Ref, rows: u32) !void {
        return switch (d.*) {
            inline else => |m| m.absorb(taps, rows),
        };
    }
    pub fn propose(d: *Any, pending: u32, nodes: u32) !Tree {
        return switch (d.*) {
            inline else => |m| m.propose(pending, nodes),
        };
    }
    /// The committed taps as the ring a lane's state copies.
    pub fn ring(d: Any) Ring {
        return switch (d) {
            inline else => |m| m.ring(),
        };
    }
    /// An empty context.
    pub fn reset(d: Any) !void {
        switch (d) {
            .dflash2 => |m| try m.reset(),
            .dspine => |m| m.reset(),
        }
    }
    /// After a GPU failure: an empty context that drafts again.
    pub fn recover(d: Any) !void {
        switch (d) {
            .dflash2 => |m| try m.recover(),
            .dspine => |m| m.reset(),
        }
    }
    /// A new session over the ring's kept taps: the context stands at `at`, its caches rebuilt from the ring.
    pub fn standAt(d: Any, at: u64) !void {
        try d.reset();
        switch (d) {
            inline else => |m| m.committed_end = at,
        }
    }
    /// Per-stream state sized for streams reaching `positions` rows (DSpine below its window; DFlash2 keeps its own).
    pub fn fit(d: Any, positions: u64) !void {
        switch (d) {
            .dflash2 => {},
            .dspine => |m| try m.fit(positions),
        }
    }
    /// The target's TF_Q27_PROFILE trace times DFlash2's passes too.
    pub fn profile(d: Any, trace: ?*core.gpu_profile.Trace) void {
        if (d.dflash()) |m| m.backend.profile(trace);
    }
};

/// The drafter at `draft_dir` binds the target at `target_dir`: refused before either loads.
pub fn admit(a: std.mem.Allocator, io: std.Io, target_dir: []const u8, draft_dir: []const u8) !void {
    switch (try kindOf(a, io, draft_dir)) {
        .dflash2 => _ = try @import("dflash/config.zig").admit(a, io, target_dir, draft_dir),
        .dspine => {
            const target = try readConfig(a, io, target_dir, 16 << 20);
            defer a.free(target);
            const draft = try readConfig(a, io, draft_dir, 1 << 20);
            defer a.free(draft);
            const t = try @import("config.zig").parse(a, target, null);
            try ds_config.bind(try ds_config.parse(a, draft), t.hidden, t.vocab, t.layers);
        },
    }
}

fn readConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(limit));
}
const ds_config = @import("dspine/config.zig");
