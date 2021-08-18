use rush::simd::*;
use rush::simd::vec::SimdVecImpl;
use rush::simd::avx::f32x8;
use rush::simd::sse::f32x4;
use rush::lsh::vector::Vector;
use itertools::zip_eq;
use std::time::{Duration, Instant};

fn l2(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    zip_eq(x.iter(), y.iter()).
        fold(0.0, |acc, (x, y)| {
            let delta = x - y;
            acc + delta * delta
        }).
        sqrt()
}

pub fn main() {
    /*
    let x = SimdVecImpl::<f32x4, 192>::new();
    let y = SimdVecImpl::<f32x4, 192>::new();
    */ 
    let x = SimdVecImpl::<f32x8, 96>::new();
    let y = SimdVecImpl::<f32x8, 96>::new();
    
    for _ in (0..10000) {
        x.distance(&y);
    }

    let start = Instant::now();
    let d = x.distance(&y);
    let duration = start.elapsed();

    println!("{:?}", duration);
    let w = vec![1f32; 768];
    let v = vec![0f32; 768];
    let start_l2 = Instant::now();
    let d_l2 = l2(&w, &v);
    let duration_l2 = start_l2.elapsed();
    println!("{:?}", duration_l2);
    println!("{}", d_l2);
    println!("{}", d);
}
