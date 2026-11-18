// The datamove contract's rig: guarded device buffers, the little-endian byte helpers and the failure message.
const std = @import("std");
const vk = @import("vulkan");

/// Guard bytes on both sides of every rig payload (the largest minStorageBufferOffsetAlignment).
pub const guard = 256;
const pattern = 0xa5;

// Each call uploads inputs, poisons outputs with 0xff, and guards every payload.
pub const Rig = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    timeout_ns: u64,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,

    const Kind = enum { input, output };
    const Slot = struct { buf: vk.Buffer, len: usize, kind: Kind };

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        // A hook error after a recorded dispatch leaves the buffer open: drop those records before freeing slots.
        r.cmds.discard();
        for (r.slots.items) |*s| s.buf.deinit();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, data: ?[]const u8, len: usize, kind: Kind) !usize {
        const all = try r.a.alloc(u8, len + 2 * guard);
        defer r.a.free(all);
        @memset(all, pattern);
        if (data) |d| {
            @memcpy(all[guard..][0..len], d);
        } else {
            @memset(all[guard..][0..len], 0xff); // an output element no launch writes reads as NaN
        }
        var b = try vk.Buffer.init(r.dev, .device, all.len);
        errdefer b.deinit();
        // Timeout or InFlight leaves completion unknown: exit before any teardown can free GPU-owned resources.
        try vk.orExit(vk.upload(r.cmds, b, 0, r.staging, all, r.timeout_ns));
        try r.slots.append(r.a, .{ .buf = b, .len = len, .kind = kind });
        return r.slots.items.len - 1;
    }

    pub fn input(r: *Rig, data: []const u8) !usize {
        return r.slot(data, data.len, .input);
    }

    /// An output of `len` bytes, all 0xff until the launch writes it.
    pub fn output(r: *Rig, len: usize) !usize {
        return r.slot(null, len, .output);
    }

    /// The payload of slot `i` as a typed view for a binding; call between prepare and the next prepare.
    pub fn view(r: *Rig, i: usize, kind: vk.tensor.Kind, elements: usize) !vk.tensor.View {
        const s = &r.slots.items[i];
        return vk.tensor.View.init(&s.buf, kind, guard, elements);
    }

    fn gatherBytes(r: *Rig, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind or whole) {
            const at = out.items.len;
            const n = if (whole) s.len + 2 * guard else s.len;
            try out.resize(r.a, at + n);
            try vk.orExit(vk.download(r.cmds, s.buf, if (whole) 0 else guard, r.staging, out.items[at..][0..n], r.timeout_ns));
        };
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gatherBytes(.output, &r.outs, false);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gatherBytes(.input, &r.seen, false);
    }

    /// Every guard of every buffer still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        const all = try r.gatherBytes(.input, &r.seen, true);
        var at: usize = 0;
        for (r.slots.items) |s| {
            for (all[at..][0..guard]) |v| if (v != pattern) return false;
            for (all[at + guard + s.len ..][0..guard]) |v| if (v != pattern) return false;
            at += s.len + 2 * guard;
        }
        return true;
    }

    /// Every payload is its own buffer, so no output range overlaps an input range.
    pub fn overlap(r: *Rig) bool {
        _ = r;
        return false;
    }
};

pub fn read16(bytes: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
}

pub fn write16(bytes: []u8, i: usize, bits: u16) void {
    std.mem.writeInt(u16, bytes[2 * i ..][0..2], bits, .little);
}

pub fn read32(bytes: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
}

pub fn write32(bytes: []u8, i: usize, bits: u32) void {
    std.mem.writeInt(u32, bytes[4 * i ..][0..4], bits, .little);
}

pub fn buf(msg: *[160]u8, comptime fmt: []const u8, args: anytype) ?[]const u8 {
    return std.fmt.bufPrint(msg, fmt, args) catch "message too long";
}
