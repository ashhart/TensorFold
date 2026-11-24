//! The leader step header (T2 of #2560): one 64-byte header and its optional sections for every step protocol.
const header = @import("header.zig");
const message = @import("message.zig");
pub const section = @import("section.zig");

pub const magic = header.magic;
pub const version = header.version;
pub const header_size = header.size;
pub const Kind = header.Kind;
pub const Mode = header.Mode;
pub const Header = header.Header;
pub const Context = header.Context;
pub const encodeHeader = header.encodeHeader;
pub const decodeHeader = header.decodeHeader;
pub const expect = header.expect;

pub const Id = section.Id;
pub const Op = section.Op;
pub const Stream = section.Stream;
pub const Sampling = section.Sampling;
pub const Patch = section.Patch;
pub const Copy = section.Copy;
pub const SnapOp = section.SnapOp;
pub const Snapshot = section.Snapshot;
pub const RowEntry = section.RowEntry;
pub const Section = section.Section;

pub const Message = message.Message;
pub const Limits = message.Limits;
pub const Error = message.Error;
pub const Step = message.Step;
pub const size = message.size;
pub const write = message.write;
pub const encode = message.encode;
pub const decode = message.decode;

test {
    _ = header;
    _ = section;
    _ = message;
    _ = @import("message_test.zig");
    _ = @import("fixtures_test.zig");
}
