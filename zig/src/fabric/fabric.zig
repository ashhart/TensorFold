//! TensorFold's fabric: MCDMA's link and KV handoff protocols, one- and two-sided collectives, costs and verbs.
const std = @import("std");

pub const layout = @import("layout.zig");
pub const words = @import("words.zig");
pub const mailbox = @import("mailbox.zig");
pub const relink = @import("relink.zig");
pub const registration = @import("registration.zig");
pub const wire = @import("wire.zig");
pub const manifest = @import("manifest.zig");
pub const frames = @import("frames.zig");
pub const producer = @import("producer.zig");
pub const decoder = @import("decoder.zig");
pub const rdma = @import("rdma.zig");
pub const mcdma_abi = @import("mcdma_abi.zig");
pub const mcdma = @import("mcdma.zig");
pub const packet = @import("packet.zig");
pub const collective = @import("collective.zig");
pub const cost = @import("cost.zig");
pub const verbs = @import("verbs");
pub const verbs_abi = verbs.abi;
pub const verbs_qp = @import("verbs_qp.zig");
pub const verbs_link = @import("verbs_link.zig");
pub const sendrecv = @import("sendrecv.zig");
pub const sendrecv_fake = @import("sendrecv_fake.zig");
pub const prefill_source = @import("prefill_source.zig");
pub const hybrid_state = @import("hybrid_state.zig");
pub const hybrid_export = @import("hybrid_export.zig");
pub const crc32 = @import("crc32.zig");
pub const fake = @import("fake.zig");
pub const link_fake = @import("link_fake.zig");

pub const Rdma = rdma.Rdma;
pub const Channel = collective.Channel;
pub const PrefillSource = prefill_source.PrefillSource;

test {
    std.testing.refAllDecls(@This());
}
