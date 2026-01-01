//! Server benchmarks for reasonkit-web
//!
//! These benchmarks measure HTTP server performance including
//! request handling latency and throughput.

use criterion::{criterion_group, criterion_main, Criterion};

fn server_benchmark(_c: &mut Criterion) {
    // TODO: Implement server benchmarks once the server is fully implemented
}

criterion_group!(benches, server_benchmark);
criterion_main!(benches);
