//! The prompt allocation sizes, shared by construction and preflight without a device.
const std = @import("std");
pub const trained = 262144;
pub const D = 2560;
pub const WIDE = 4 * D;
pub const PLE_TAIL = 9;
pub const PMAX = 8192;
pub const CS_ROW = 3 * WIDE * 2;
const R = PMAX;
pub const sizes = .{
    .ids = R * 4,
    .pids = R * 16 * 4,
    .h = .{ R * WIDE * 2, R * WIDE * 2 },
    .ssp = R * 10 * 4 * 4,
    .normed = R * WIDE * 2,
    .dn = R * 324 * 2,
    .hact = R * 320 * 2,
    .inj_a = R * 4 * 2,
    .inj_m = R * 4 * 2,
    .up = R * WIDE * 2,
    .mixed = R * D * 2,
    .p = R * 16480 * 2,
    .gout = R * 6144 * 2,
    .branch = R * D * 2,
    .cso = CS_ROW,
    .q = R * 24 * 256 * 2,
    .kout = R * 2 * 256 * 2,
    .iq = R * 4 * 128 * 2,
    .po = 64,
    .pm = 64,
    .aout = R * 6144 * 2,
    .pos = R * 4,
    .pos3 = R * 12,
    .nk = R * 4,
    .zeros = R * 4,
    .kvmeta = 16,
    .lg = R * 513 * 4,
    .pick = R * 10 * 4,
    .wts = R * 10 * 4,
    .cnt = 512 * 4,
    .off = 513 * 4,
    .cur = 512 * 4,
    .row_of = R * 10 * 4,
    .xs = R * 10 * D * 2,
    .g = R * 10 * 640 * 2,
    .u = R * 10 * 640 * 2,
    .a = R * 10 * 640 * 2,
    .ds = R * 10 * D * 2,
    .sg = R * 640 * 2,
    .su = R * 640 * 2,
    .sa = R * 640 * 2,
    .ydown = R * 11 * D * 2,
    .emb = R * D * 2,
    .kvp = R * 12800 * 2,
    .gated = R * WIDE * 2,
    .hout = R * WIDE * 2,
    .cin = (PLE_TAIL + R) * WIDE * 2,
    .rows = 16,
    .part = 8 * R * 324 * 4,
    .qn = R * 16 * 128 * 4,
    .kn = R * 16 * 128 * 4,
    .v = R * 48 * 128 * 4,
    .gg = R * 48 * 4,
    .beta = R * 48 * 4,
    .ys = R * 6144 * 4,
    .mids = R * 4,
    .n_add = 16,
};

pub fn bytes() u64 {
    var total: u64 = 0;
    inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |name| {
        const size = @field(sizes, name);
        if (comptime std.mem.eql(u8, name, "h")) {
            inline for (size) |n| total += @max(n, 64);
        } else total += @max(size, 64);
    }
    return total;
}
