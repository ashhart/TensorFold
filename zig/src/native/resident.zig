//! The native hosts share the Metal runtime's bounded residency owner.
const mtl = @import("metal");
pub const Resident = mtl.resident.Resident;
pub const Held = mtl.resident.Held;
