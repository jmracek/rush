use std::future::Future;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, Semaphore, RwLock};
use std::sync::Arc;
use rand::Rng;
use rush::lsh::LocalitySensitiveHashDatabase;
use rush::simd::SimdVecImpl;
use rush::simd::f32x4;
use rush::net::*;


#[tokio::main]
pub async fn main() -> rush::Result<()> {
    let listener = TcpListener::bind("127.0.0.1:9090").await?;
    run_server(listener, tokio::signal::ctrl_c()).await;
    Ok(())
}

async fn run_server(listener: TcpListener, shutdown: impl Future) {
    let db_ptr = Arc::new(RwLock::new(LocalitySensitiveHashDatabase::new(32, 768)));
    let mut db = db_ptr.write().await;
    // Insert 10,000 random vectors
    let mut rng = rand::thread_rng();
    let dim = 768usize;
    println!("Inserting 10_000 random vectors into LSH DB...");

    for _ in 0..10_000 {
        let random_vector = (0..dim).
            map(|_| rng.gen_range(-1f32..1f32)).
            collect::<SimdVecImpl<f32x4, 192>>();
        match db.insert(random_vector) {
            Ok(_) => {},
            Err(_) => panic!("Error encountered while inserting to LSH DB.  This shouldn't have happened"),
        };
    }
    
    drop(db);
    
    println!("Database prepared!");
    let (notify_shutdown, _) = broadcast::channel(1);

    //let (shutdown_sig, _) = broadcast::channel(1);
    let max_connections = 255;
 
    let mut server = Listener {
        listener,
        database: db_ptr.clone(),
        connection_limiter: Arc::new(Semaphore::new(max_connections)),
        shutdown_signal: notify_shutdown
    };
    
    tokio::select! {
        result = server.run() => {
            if let Err(err) = result {
                println!("Oopsie woopsie! I made a fucky-wucky: {}", err);
            }
        }
        _ = shutdown => { 
            println!("Teehee! Bye bye!");
        }
    }
}
