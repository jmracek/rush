use crate::lsh::vector::Vector;
use std::mem::size_of;
use std::slice;
use std::fmt;
use std::marker::PhantomData;
use std::io::Write;
use rand::Rng;
use byteorder::{ByteOrder, LittleEndian};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::de;

#[derive(Debug)]
struct UnitVector<T> where 
    T: Vector,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType> 
{
    u: T
} 

#[derive(Debug)]
pub struct RandomProjection<T> 
where
    T: Vector,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    proj: UnitVector<T>
}

impl<T> RandomProjection<T> 
where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let random_vector = (0..dim).
            map(|_| rng.gen_range(-1f32..1f32)).
            collect::<T>();
        let norm = T::dot(&random_vector, &random_vector).sqrt();
        let random_unit = random_vector / norm;
        RandomProjection {
            proj: UnitVector {u: random_unit}
        }
    }

    pub fn hash(&self, v: &T) -> u64 {
        if T::dot(&self.proj.u, v) > 0f32 { 1u64 } else { 0u64 }
    }
}


impl<T> Serialize for RandomProjection<T> 
where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer
    {
        let mut buf = Vec::<u8>::new();
        let element_count = self.proj.u.dimension() as u32;
        
        buf.write(&element_count.to_le_bytes());
        
        for element in &self.proj.u {
            buf.write(&element.to_le_bytes());
        }
        
        serializer.serialize_bytes(buf.as_slice())
    }
}

struct RPVisitor<T> 
where
    T: Vector,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    marker: PhantomData<fn() -> RandomProjection<T>>
}

impl<T> RPVisitor<T> 
where
    T: Vector,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    fn new() -> Self { 
        RPVisitor{ 
            marker: PhantomData 
        } 
    }
}

impl<'de, T> de::Visitor<'de> for RPVisitor<T> where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    type Value = RandomProjection<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "a byte array where the first four bytes yield the dimension, \
                           and the remainder are the little endian bytes of the normalized \
                           projection vector")
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where E: de::Error
    {
        let mut dim_buf = [0u8; 4];
        dim_buf.copy_from_slice(&v[0..4]);
        let dim_u32 = u32::from_le_bytes(dim_buf);
        let dim = dim_u32 as usize;

        let expected_byte_count = size_of::<<T as Vector>::DType>() * dim;

        if expected_byte_count != v.len() - 4 {
            return Err(de::Error::invalid_length(v.len(), &self))
        }
        
        let mut float_elts = Vec::<f32>::new();
        float_elts.reserve(dim);
 
        byteorder::LittleEndian::read_f32_into_unchecked(v, float_elts.as_mut_slice());
        let unit = float_elts.into_iter().collect::<T>();
        let norm = T::dot(&unit, &unit).sqrt();
        
        if (norm - 1f32).abs() > 1e-5 {
            Err(de::Error::invalid_value(de::Unexpected::Bytes(v), &self))
        }
        else {
            Ok(RandomProjection{
                proj: UnitVector {u: unit}
            })
        }
    }

    fn visit_seq<A>(self, mut v: A) -> Result<Self::Value, A::Error>
        where A: de::SeqAccess<'de>
    {
        let mut buf = [0u8; 4];
        let mut buf_dtype = [0u8; size_of::<<T as Vector>::DType>()];

        buf[0] = v.next_element::<u8>().unwrap().unwrap();
        buf[1] = v.next_element::<u8>().unwrap().unwrap();
        buf[2] = v.next_element::<u8>().unwrap().unwrap();
        buf[3] = v.next_element::<u8>().unwrap().unwrap();

        let dim_u32 = u32::from_le_bytes(buf);
        let dim = dim_u32 as usize;

        let expected_byte_count = size_of::<<T as Vector>::DType>() * dim;
        let mut byte_count: usize = 0;
        
        let mut float_elts = Vec::<f32>::new();
        float_elts.reserve(dim);

        while let Ok(Some(element)) = v.next_element::<u8>() {
            byte_count += 1;
            let lane_idx = (byte_count - 1) % size_of::<<T as Vector>::DType>();
            buf_dtype[lane_idx] = element;

            if lane_idx == size_of::<<T as Vector>::DType>() - 1 {
                float_elts.push(<T as Vector>::DType::from_le_bytes(buf_dtype));
            }

        }
 
        let unit = float_elts.into_iter().collect::<T>();
        let norm = T::dot(&unit, &unit).sqrt();
        
        if (norm - 1f32).abs() > 1e-5 {
            Err(de::Error::invalid_value(de::Unexpected::Seq, &self))
        }
        else {
            Ok(RandomProjection{
                proj: UnitVector {u: unit}
            })
        }
    }
}

impl<'de, T> Deserialize<'de> for RandomProjection<T> 
where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de>
    {
        deserializer.deserialize_bytes(RPVisitor::<T>::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::vec::SimdVecImpl;
    use crate::simd::sse::f32x4;
    use serde_json;

    #[test]
    fn test_create_random_projection() {
        let rp = RandomProjection::<SimdVecImpl<f32x4, 8>>::new(8 * 4);
        let norm = (&rp.proj.u).dot(&rp.proj.u).sqrt();
        assert!((norm - 1f32).abs() < 1e-5);
    }
    
    #[test]
    fn test_random_projection_hash() {
    }
    
    #[test]
    fn test_rp_ser_de() {
        let rp = RandomProjection::<SimdVecImpl<f32x4, 8>>::new(8 * 4);
        let ser = serde_json::to_string(&rp).unwrap();
        let de: RandomProjection<SimdVecImpl<f32x4, 8>> = serde_json::from_str(&ser).unwrap();
        assert_eq!(rp.proj.u, de.proj.u);
    }
}
