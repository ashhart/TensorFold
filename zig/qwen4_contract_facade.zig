//! The contract fixture object shares one core owner for all contract types.
const owner = @import("core").contract;
pub const spec = owner.spec;
pub const gates = owner.gates;
pub const rows = owner.rows;
