//! Startup cache capacity and policy metadata shared by native engines and discovery.
pub const PromptCacheCapacity = struct { budget_bytes: u64, minimum_state_bytes: u64 };

pub const PromptCachePlan = struct {
    source: enum { physical_footprint, explicit, metal_working_set },
    budget_bytes: u64,
    explicit_budget: bool = false,
    over_cap: ?bool = null,
    ram_bytes: ?u64 = null,
    ready_footprint_bytes: ?u64 = null,
    cap_bytes: ?u64 = null,
    room_bytes: ?u64 = null,
    margin_bytes: ?u64 = null,
};
