//! Memory operations: High-level read/write/query interface.

use tle_vsa::HyperVector;

/// Types of memory operations.
#[derive(Clone, Debug)]
pub enum MemoryOp {
    /// Store a new fact (role, filler, importance).
    Store {
        role: HyperVector,
        filler: HyperVector,
        importance: f32,
    },
    /// Retrieve a filler by role.
    Retrieve { role: HyperVector },
    /// Forget a specific fact.
    Forget {
        role: HyperVector,
        filler: HyperVector,
    },
    /// Query: check if a fact exists (similarity-based).
    Query {
        role: HyperVector,
        expected_filler: HyperVector,
    },
}
