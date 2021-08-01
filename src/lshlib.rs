use itertools::zip_eq;
use std::collections::HashMap;
use std::vec::Vec;
use std::rc::Rc;
use rand::Rng;

pub trait Euclidian {
    fn vec(&self) -> &Vec<f32>;
    fn dimension(&self) -> usize;
}

impl Euclidian for Vec<f32> {
    fn dimension(&self) -> usize { self.len() }
    fn vec(&self) -> &Vec<f32> { &self }
}

impl<'a, T> Euclidian for &'a T where T: Euclidian {
    fn dimension(&self) -> usize { T::dimension(self) }
    fn vec(&self) -> &Vec<f32> { T::vec(self) }
}

impl<'a, T> Euclidian for &'a mut T where T: Euclidian {
    fn dimension(&self) -> usize { T::dimension(self) }
    fn vec(&self) -> &Vec<f32> { T::vec(self) }
}

fn euclidian_distance<T: Euclidian>(x: &T, y: &T) -> f32 {
    zip_eq(x.vec().iter(), y.vec().iter()).
        fold(0.0, |acc, (x, y)| acc + (x - y).powf(2.0)).
        sqrt()
}

struct UnitVector {
    data: Vec<f32>
}

impl UnitVector {
    fn rand(dimension: u16) -> UnitVector {
        let mut rng = rand::thread_rng();
        let random_vector = (0..dimension).
            map(|_| rng.gen_range(-1.0..1.0)).
            collect::<Vec<f32>>();
        let norm = random_vector.
            iter().
            fold(0.0, |acc, x| acc + x.powf(2f32)).
            sqrt();
        let data = random_vector.
            iter().
            map(|x| x / norm).
            collect::<Vec<f32>>();
        UnitVector{data}
    }
}

impl Euclidian for UnitVector {
    fn vec(&self) -> &Vec<f32> {
        &self.data
    }
    
    fn dimension(&self) -> usize {
        self.data.len()
    }
}

pub struct RandomProjection {
    v: UnitVector
}

impl RandomProjection {
    fn new(dimension: u16) -> RandomProjection {
        RandomProjection{ v: UnitVector::rand(dimension) }
    } 

    fn hash<T: Euclidian>(&self, v: &T) -> u8 {
        if v.dimension() != self.v.dimension() {
            panic!(
                "Attempted to compute hash when input dimension was {}, but RandomProjection has dimension {}", 
                v.dimension(),
                self.v.dimension()
            )
        }
        let dot_product: f32 = zip_eq(self.v.vec().iter(), v.vec().iter()).
            fold(0.0, |acc, (x, y)| acc + x * y);
        if dot_product > 0.0 { 1u8 } else { 0u8 }
    }
}

pub struct StableHashFunction {
    projections: Vec<RandomProjection>
}

impl StableHashFunction {
    pub fn rand(num_projections: u16, dimension: u16) -> StableHashFunction {
        let hash_functions = (0..num_projections).
            map(|_| RandomProjection::new(dimension)).
            collect::<Vec<RandomProjection>>();

        StableHashFunction{projections: hash_functions}
    }

    pub fn hash<T: Euclidian>(&self, v: &T) -> u64 {
        self.projections.
            iter().
            map(|proj| proj.hash(v)).
            enumerate().
            fold(0u64, |acc, (i, sgn)| acc | ((sgn as u64) << i))
    }
}

struct StableHashTable<T: Euclidian> {
    table: HashMap<u64, Vec<Rc<T>>>,
    hashfn: StableHashFunction
}

impl<T: Euclidian> StableHashTable<T> {
    fn new(num_projections: u16, dimension: u16) -> StableHashTable<T> {
        StableHashTable {
            table: HashMap::<u64, Vec<Rc<T>>>::new(),
            hashfn: StableHashFunction::rand(num_projections, dimension)
        }
    }

    fn insert(&mut self, item: Rc<T>) {
        let hash_key = self.hashfn.hash(&*item);
        if let Some(container) = self.table.get_mut(&hash_key) {
            container.push(item);
        }
        else {
            self.table.insert(hash_key, vec![item]);
        }
    }
    
    fn query<'a>(&'a self, item: &T) -> Option<&'a T> {
        match self.table.get(&self.hashfn.hash(item)) {
            None => None,
            Some(result_set) => {
                let best_match: Option<(usize, f32)> = 
                    result_set.
                        iter().
                        map(|q| euclidian_distance(&**q, item)).
                        enumerate().
                        fold(None, |acc, (idx, q)| {
                            match acc {
                                None => Some((idx, q)),
                                Some((min_idx, p)) => {
                                    if q < p { Some((idx, q)) } else { Some((min_idx, p)) }
                                }
                            }
                        });
                
                match best_match {
                    Some((idx_closest, _)) => Some(&*result_set[idx_closest]),
                    _ => None
                }
            }
        }
    }
}

pub struct LocalitySensitiveHashDatabase<T: Euclidian> {
    items: Vec<Rc<T>>,
    tables: Vec<StableHashTable<T>>
}

impl<T: Euclidian> LocalitySensitiveHashDatabase<T> {
    pub fn len(&self) -> usize {
        self.items.len()    
    }

    pub fn insert(&mut self, item: T) {
        self.items.push(Rc::new(item));
        if let Some(working_item) = self.items.last() {
            for table in self.tables.iter_mut() {
                table.insert(Rc::clone(&working_item));
            }
        }
    }
    
    pub fn new(replicas: u16, num_projections: u16, dimension: u16) -> Self {
        LocalitySensitiveHashDatabase {
            items: Vec::<Rc<T>>::new(),
            tables: (0..replicas).
                map(|_| StableHashTable::<T>::new(num_projections, dimension)).
                collect::<Vec<StableHashTable<T>>>()
        }
    }
    /* 
    pub fn query(&'a self, item: &T) -> Option<&'a T> {
        let result = 
            self.tables.
                iter().
                map(|table| table.query(&item)).
                fold(None, |acc, x| {
                    match acc {
                        None => {
                            match x {
                                None => None,
                                Some(candidate) => Some((candidate, euclidian_distance(candidate, item)))
                            }
                        },
                        Some((_, min_distance)) => {
                            match x {
                                Some(next_candidate) => {
                                    let current_distance = euclidian_distance(next_candidate, item);
                                    if  current_distance < min_distance {
                                        Some((next_candidate, current_distance))
                                    }
                                    else {
                                        acc
                                    }
                                },
                                None => acc
                            }
                        }
                    }
                });
        match result {
            Some((nearest_neighbour, _)) => Some(nearest_neighbour),
            _ => None
        }
    }*/
}

/*
TODO's:
    1. Implement ability to load existing LocalitySensitiveHashDatabase from an existing saved model.
    2. Implement the ability to query an LSH database
    3. Figure out how a convenient way to implement FromIterator<T> for LSHDatabase.  The problem is
       that in the from_iter function we would need to know the number of replicas, the dimension, and
       the number of projections to use (bits).  
    4. Implement the service layer for the LSH cache.



impl<'a, T: Euclidian> FromIterator<T> for LocalitySensitiveHashDatabase<'a, T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut db = LocalitySensitiveHashDatabase::<T>::new();
        for item in iter {
            db.insert(item);
        }
        db
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidian_distance() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 0.0];
        let v3 = vec![1.0, 1.0];
        assert_eq!(euclidian_distance(&v1, &v2), 1.0);
        assert_eq!(euclidian_distance(&v1, &v1), 0.0);
        assert_eq!(euclidian_distance(&v1, &v3), 1.0);
    }

    #[test]
    fn test_create_unit_vector() {
        let v = UnitVector::rand(10);
        assert_eq!(v.data.len(), 10);
        assert!(v.data.iter().fold(true, |acc, elt| acc && (&-1.0 < elt) && (elt < &1.0)));
        assert!((v.data.iter().fold(0f32, |acc, elt| acc + elt.powf(2.0)).sqrt() - 1.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_random_projection_hash() {
        let proj = RandomProjection::new(3);
        let v = vec![2.0,2.0,2.0];
        proj.hash(&v);
    }
    
    #[test]
    fn test_stable_hash_function() {
        let hashfn = StableHashFunction::rand(64, 768);
        assert_eq!(hashfn.projections.len(), 64);
    }
    
    #[test]
    fn test_stable_hash_table_insert() {
        let mut table = StableHashTable::new(64, 3);
        let rc = Rc::new(vec![1.0,2.0,3.0]);
        table.insert(rc);
        assert_eq!(table.table.len(), 1);
    }
    
    #[test]
    fn test_stable_hash_table_query() {
        let mut table = StableHashTable::new(4, 3);
        let rc = Rc::new(vec![1.0, 2.0, 3.0]);
        table.insert(rc);
        
        let exact_query = vec![1.0, 2.0, 3.0];
        let result = table.query(&exact_query);
        assert!(match result {
            Some(item) => (item == &exact_query),
            None => false
        });
         
        let item2 = vec![1.0, 2.0, 10.0]; 
        let rc2 = Rc::new(vec![1.0, 2.0, 10.0]);
        table.insert(rc2);
        let approx_query1 = vec![1.0, 2.0, 3.1];
        let approx_query2 = vec![1.0, 2.0, 11.0];
        let maybe_result1 = table.query(&approx_query1);
        let maybe_result2 = table.query(&approx_query2);
        assert!(match maybe_result1 {
            Some(item) => (item == &exact_query),
            None => false
        });
        assert!(match maybe_result2 {
            Some(item) => (item == &item2),
            None => false
        });
    }
    
    #[test]
    fn test_lsh_db_insert() {
        let mut lsh_db = LocalitySensitiveHashDatabase::new(16, 64, 768);
        let v1 = vec![1.0; 768];
        lsh_db.insert(v1);
        let len1 = lsh_db.len();
        assert_eq!(len1, 1);
        
        let mut v2 = vec![1.0; 768];
        v2[10] = 0.2;
        lsh_db.insert(v2);
        let len2 = lsh_db.len();
        assert_eq!(len2, 2);
    }
}


/*

From Wikipedia...

In the first step, we define a new family {\displaystyle {\mathcal {G}}}{\mathcal {G}} of hash functions g, where each function g is obtained by concatenating k functions {\displaystyle h_{1},...,h_{k}}h_1, ..., h_k from {\displaystyle {\mathcal {F}}}{\mathcal {F}}, i.e., {\displaystyle g(p)=[h_{1}(p),...,h_{k}(p)]}g(p) = [h_1(p), ..., h_k(p)]. In other words, a random hash function g is obtained by concatenating k randomly chosen hash functions from {\displaystyle {\mathcal {F}}}{\mathcal {F}}. The algorithm then constructs L hash tables, each corresponding to a different randomly chosen hash function g.

In the preprocessing step we hash all n points from the data set S into each of the L hash tables. Given that the resulting hash tables have only n non-zero entries, one can reduce the amount of memory used per each hash table to {\displaystyle O(n)}O(n) using standard hash functions.

Given a query point q, the algorithm iterates over the L hash functions g. For each g considered, it retrieves the data points that are hashed into the same bucket as q. The process is stopped as soon as a point within distance {\displaystyle cR}cR from q is found.


Summary:

Preprocessing:
1. Choose L hash functions at random from G (which are themselves concatenations of random hash functions from L)
2. Create L hash tables out of the N data points.

Query:
1. Given vector q, compute all L hashes and retrieve all the elements that hashed to g(q)
2. Among g_i(q), find the closest vector to q for all i, call it \hat{g}_i(q)
3. Find argmin_i d(\hat{g}_i(q), q)

*/

