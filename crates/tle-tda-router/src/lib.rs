//! # TDA Mapper Topological Router
//!
//! Replaces probabilistic softmax gating in MoE with deterministic
//! topological routing based on the Mapper algorithm from TDA.
//!
//! ## Algorithm
//!
//! 1. **Filter Function**: Project latent vectors onto a scalar axis
//!    (e.g., norm, first principal component, geometric energy)
//! 2. **Covering**: Divide filter range into overlapping intervals
//! 3. **Clustering**: Within each interval, cluster vectors by proximity
//! 4. **Graph Construction**: Build Reeb graph connecting clusters
//! 5. **Routing**: Assign vector to the node (expert) corresponding
//!    to its topological neighborhood
//!
//! This is entirely deterministic: same input always routes to same node.

pub mod filter;
pub mod mapper;
pub mod reeb_graph;
pub mod router;

pub use filter::FilterFunction;
pub use mapper::MapperConfig;
pub use reeb_graph::{ReebGraph, ReebNode, NodeType};
pub use router::TopologicalRouter;
