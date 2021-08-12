use rush::lshlib;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_stable_hash_function(c: &mut Criterion) {
    let hashfn = lshlib::StableHashFunction::rand(64, 768);
    let v = vec![1.0; 768];
    c.bench_function("hash 768", |b| b.iter(|| hashfn.hash(&v)));
}

/*
fn bench_query_to_lsh_db(c: &mut Criterion) {
    let mut lsh_db = lshlib::LocalitySensitiveHashDatabase::new(16, 64, 768);
    let v = vec![1.0; 768];
    c.bench_function("lsh_db 16-64-768", |b| {
        b.iter(|| lsh_db.query(&v) )
    });
}
*/
criterion_group!(stable_hash_benches, bench_stable_hash_function);
//criterion_group!(lsh_db_benches, bench_query_to_lsh_db);
criterion_main!(stable_hash_benches);
