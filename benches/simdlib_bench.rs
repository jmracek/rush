use rush::simdlib::*;
use rush::lshlib::*;
use criterion::{criterion_group, criterion_main, Criterion};
use itertools::zip_eq;

fn l2(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    zip_eq(x.iter(), y.iter()).
        fold(0.0, |acc, (x, y)| acc + (x - y).powf(2.0)).
        sqrt()
}

fn bench_standard_l2_distance(c: &mut Criterion) {
    c.bench_function(
        "d768 l2 dist", 
        |b| {
            let w = vec![0f32; 65536];
            let v = vec![0f32; 65536];
            b.iter(|| l2(&w, &v))
        }
    );
}

fn bench_simd_f32x4_l2_distance(c: &mut Criterion) {
    c.bench_function(
        "d768 l2 f32x4 dist", 
        |b| {
            let x = SimdVecImpl::<f32x4, 16384>::new();
            let y = SimdVecImpl::<f32x4, 16384>::new();
            b.iter(|| (&x).distance(&y))
        }
    );
}

fn bench_simd_f32x8_l2_distance(c: &mut Criterion) {
    c.bench_function(
        "d768 l2 f32x8 dist", 
        |b| {
            let x = SimdVecImpl::<f32x8, 8192>::new();
            let y = SimdVecImpl::<f32x8, 8192>::new();
            b.iter(|| (&x).distance(&y))
        }
    );
}


criterion_group!(vector_distance_benches, 
                 bench_standard_l2_distance, 
                 bench_simd_f32x4_l2_distance,
                 bench_simd_f32x8_l2_distance);
criterion_main!(vector_distance_benches);
