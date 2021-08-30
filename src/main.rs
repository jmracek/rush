use std::future::Future;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Semaphore};
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
    let mut db = LocalitySensitiveHashDatabase::new(32, 768);
    
    // Insert 10,000 random vectors
    let mut rng = rand::thread_rng();
    let dim = 768usize;
    println!("Inserting 100 random vectors into LSH DB...");

    for _ in 0..100 {
        let random_vector = (0..dim).
            map(|_| rng.gen_range(-1f32..1f32)).
            collect::<SimdVecImpl<f32x4, 192>>();
        db.insert(random_vector);
    }
    
    let db_ptr = Arc::new(db); 

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
                println!("Oopsie woopsie! I made a fucky-wucky");
            }
        }
        _ = shutdown => { 
            println!("Teehee! Bye bye!");
        }
    }
    println!("Goodbye!");
}
