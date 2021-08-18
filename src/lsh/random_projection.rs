use crate::lsh::vector::Vector;
use rand::Rng;

struct UnitVector<T: Vector>(T); 

pub struct RandomProjection<T: Vector> {
    proj: UnitVector<T>
}

impl<T: Vector<DType=f32>> RandomProjection<T> {
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let random_vector = (0..dim).
            map(|_| rng.gen_range(-1f32..1f32)).
            collect::<T>();
        let norm = T::dot(&random_vector, &random_vector).sqrt();
        let random_unit = random_vector / norm;
        RandomProjection {
            proj: UnitVector(random_unit)
        }
    }

    pub fn hash(&self, v: &T) -> u64 {
        if T::dot(&self.proj.0, v) > 0f32 { 1u64 } else { 0u64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
