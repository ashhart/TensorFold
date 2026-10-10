//! The GPU a rank serves on and what the run may use there, chosen once by whoever opens an engine.

const std = @import("std");
const runtime = @import("runtime.zig");
const kernels = @import("kernels.zig");
const Caps = @import("caps.zig").Caps;
const Policy = @import("policy.zig").Policy;
const Group = @import("comm/group.zig").Group;

pub const Device = struct {
    /// The ordinal among the visible cards.
    index: c_int,
    caps: Caps,
    /// Rank 0's policy; the other ranks adopt it when they join the group.
    policy: Policy,
    /// The tensor-parallel group this rank belongs to (null: one rank).
    group: ?*Group = null,

    /// The card of `rank`: with every card visible rank r takes card r, with one card a process that card.
    pub fn ordinal(rank: u32) c_int {
        var r = runtime.Runtime.open() catch return 0;
        defer r.close();
        const count = r.deviceCount() catch return 0;
        return if (count > 0) @intCast(rank % @as(u32, @intCast(count))) else 0;
    }

    /// The caps of the card at `index` when this build's kernels were compiled for it; refused otherwise.
    pub fn kernelCaps(r: *runtime.Runtime, index: c_int) error{ UnsupportedGpu, KernelArch }!Caps {
        var buf: [64]u8 = undefined;
        const arch = archName(r, index, &buf) catch return error.UnsupportedGpu;
        const caps = Caps.of(arch) orelse return error.UnsupportedGpu;
        const end = std.mem.indexOfScalar(u8, arch, ':') orelse arch.len;
        if (!std.mem.eql(u8, arch[0..end], kernels.arch)) {
            std.log.err("this GPU is {s}; this build's kernels are for {s} (-Dhip-arch)", .{ arch[0..end], kernels.arch });
            return error.KernelArch;
        }
        return caps;
    }

    /// What every rank of a group must share: the GPU its kernels were built for.
    pub fn groupId(caps: Caps) u32 {
        return std.hash.Crc32.hash(kernels.arch) ^ @as(u32, @backingInt(caps.generation));
    }

    /// What the card at `index` can do, or null when it is no usable GPU or one outside the caps table.
    pub fn capsOf(index: c_int) ?Caps {
        var r = runtime.Runtime.open() catch return null;
        defer r.close();
        var buf: [64]u8 = undefined;
        const arch = archName(&r, index, &buf) catch return null;
        return Caps.of(arch);
    }

    /// The card's gfx name ("gfx1100", "gfx906:sramecc+:xnack-"), found in its properties record by its prefix.
    pub fn archName(r: *runtime.Runtime, index: c_int, out: []u8) runtime.Error![]const u8 {
        const Properties = *const fn (*anyopaque, c_int) callconv(.c) c_int;
        const get = r.lib.lookup(Properties, "hipGetDevicePropertiesR0600") orelse return error.MissingSymbol;
        var props: [16384]u8 align(8) = @splat(0);
        try runtime.check(get(&props, index));
        var at: usize = 256;
        while (at + 4 < props.len) : (at += 1) {
            if (!std.mem.startsWith(u8, props[at..], "gfx") or !std.ascii.isHex(props[at + 3])) continue;
            const text = std.mem.sliceTo(props[at..], 0);
            if (text.len > out.len) return error.Invalid;
            @memcpy(out[0..text.len], text);
            return out[0..text.len];
        }
        return error.Invalid;
    }
};
