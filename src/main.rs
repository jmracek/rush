use rush::simdlib::*;
use rush::lshlib::*;
use itertools::zip_eq;
use std::time::{Duration, Instant};

fn l2(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    zip_eq(x.iter(), y.iter()).
        fold(0.0, |acc, (x, y)| acc + (x - y).powf(2.0)).
        sqrt()
}
use std::mem::align_of_val;
pub fn main() {
    let x = SimdVecImpl::<f32x4, 192>::new();
    let y = SimdVecImpl::<f32x4, 192>::new();
    /*
    let x = SimdVecImpl::<f32x8, 96>::new();
    let y = SimdVecImpl::<f32x8, 96>::new();
    */ 
    

    let start = Instant::now();
    let d = x.distance(&y);
    let duration = start.elapsed();

    println!("{:?}", duration);
    /*
    let w = vec![0f32; 768];
    let v = vec![0f32; 768];
    let d = l2(&w, &v);
    */
    println!("{}", d);
}
