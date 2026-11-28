//! Compatibility names for the shared process-resident reader.
const resident = @import("sys").process_resident;
pub const Snapshot = resident.Snapshot;
pub const parse = resident.parse;
pub const read = resident.read;

test "process resident: server aliases preserve shared types and function identity" {
    const t = @import("std").testing;
    try t.expect(Snapshot == resident.Snapshot);
    try t.expectEqual(&resident.read, &read);
    try t.expectEqual(&resident.parse, &parse);
}
