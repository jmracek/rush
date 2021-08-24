use crate::simd::sse::f32x4;
//TODO: f32x8

#[inline(always)]
fn getblock32(block: &f32x4, idx: u8) -> u32 {
    unsafe {
        let chunk_ptr = std::mem::transmute::<&f32x4, *const u32>(block);     
        let elt_ptr = chunk_ptr.offset(idx.into());
        *elt_ptr
    }
}

#[inline(always)]
fn getblock64(block: &f32x4, idx: u8) -> u64 {
    unsafe {
        let chunk_ptr = std::mem::transmute::<&f32x4, *const u64>(block);     
        let elt_ptr = chunk_ptr.offset(idx.into());
        *elt_ptr
    }
}

#[inline(always)]
fn fmix32(mut h: u32) -> u32 {
  h ^= h >> 16;
  h *= 0x85ebca6bu32;
  h ^= h >> 13;
  h *= 0xc2b2ae35u32;
  h ^= h >> 16;
  h
}

#[inline(always)]
fn fmix64(mut k: u64) -> u64 {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdu64;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53u64;
  k ^= k >> 33;
  k
}

#[inline(always)]
fn rotl32(x: u32, r: i8) -> u32 {
    (x << r) | (x >> (32 - r))
}

#[inline(always)]
fn rotl64(x: u64, r: i8) -> u64 {
  return (x << r) | (x >> (64 - r));
}

const C32: &'static [u32] = &[0x239b961b, 0xab0e9789, 0x38b34ae5, 0xa1e38b93];
const C64: &'static [u64] = &[0x87c37b91114253d5, 0x4cf5ad432745937f];

pub fn murmur3_x86_128(data: &[f32x4], seed: u32) -> u128 {
    let mut h: [u32; 4] = [seed; 4];
    let mut k = [0u32; 4];
    let len: u32 = (data.len() as u32) * 4 * 4; // Length of data, in bytes

    for block in data {
        k[0] = getblock32(block, 0);
        k[1] = getblock32(block, 1);
        k[2] = getblock32(block, 2);
        k[3] = getblock32(block, 3);
        
        k[0] *= C32[0]; k[0] = rotl32(k[0], 15); k[0] *= C32[1];
        h[0] ^= k[0];   h[0] = rotl32(h[0], 19); h[0] += h[1];
        h[0] = h[0] * 5 + 0x561ccd1bu32;
        
        k[1] *= C32[1]; k[1] = rotl32(k[1], 16); k[1] *= C32[2];
        h[1] ^= k[1];   h[1] = rotl32(h[1], 17); h[1] += h[2];
        h[1] = h[1] * 5 + 0x0bcaa747u32;
        
        k[2] *= C32[2]; k[2] = rotl32(k[2], 17); k[2] *= C32[3];
        h[2] ^= k[2];   h[2] = rotl32(h[2], 15); h[2] += h[3];
        h[2] = h[2] * 5 + 0x96cd1c35u32;
        
        k[3] *= C32[3]; k[3] = rotl32(k[3], 18); k[3] *= C32[0];
        h[3] ^= k[3];   h[3] = rotl32(h[3], 13); h[3] += h[0];
        h[3] = h[3] * 5 + 0x32ac3b17u32;
    }

    h[0] ^= len; h[1] ^= len; h[2] ^= len; h[3] ^= len;

    h[0] += h[1]; h[0] += h[2]; h[0] += h[3];
    h[1] += h[0]; h[2] += h[0]; h[3] += h[0];

    h[0] = fmix32(h[0]);
    h[1] = fmix32(h[1]);
    h[2] = fmix32(h[2]);
    h[3] = fmix32(h[3]);

    h[0] += h[1]; h[0] += h[2]; h[0] += h[3];
    h[1] += h[0]; h[2] += h[0]; h[3] += h[0];
    
    unsafe {
        let result_ptr = std::mem::transmute::<&[u32; 4], *const u128>(&h);
        *result_ptr
    }
}

pub fn murmur3_x64_128(data: &[f32x4], seed: u32) -> u128 {
    let mut h: [u64; 2] = [seed as u64; 2]; 
    let mut k: [u64; 2] = [0; 2];
    let len: u64 = (data.len() as u64) * 4 * 4; 

    for block in data {
        k[0] = getblock64(block, 0);
        k[1] = getblock64(block, 1);
        
        k[0] *= C64[0]; k[0] = rotl64(k[0], 31); k[0] *= C64[1]; h[0] ^= k[0];
        h[0] = rotl64(h[0], 27); h[0] += h[1]; h[0] = h[0] * 5 + 0x52dce729u64;

        k[1] *= C64[1]; k[1] = rotl64(k[1], 33); k[1] *= C64[0]; h[1] ^= k[1];
        h[1] = rotl64(h[1],31); h[1] += h[0]; h[1] = h[1] * 5 + 0x38495ab5u64;
    }

    h[0] ^= len; h[1] ^= len;
    h[0] += h[1];
    h[1] += h[0];

    h[0] = fmix64(h[0]);
    h[1] = fmix64(h[1]);

    h[0] += h[1];
    h[1] += h[0];

    unsafe {
        let result_ptr = std::mem::transmute::<&[u64; 2], *const u128>(&h);
        *result_ptr
    }
}

#[cfg(test)]
mod murmurhash_test {
    use super::*;
    use crate::simd::sse::*;
    
    #[test]
    fn test_mmh_x64_128_zero() {
        let data = [f32x4::default();1];
        // Cross reference with Python implementation from module mmh3
        assert_eq!(murmur3_x64_128(&data, 0), 239788907712657087838427770177223989462u128);
    }
}
