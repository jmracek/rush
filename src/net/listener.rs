use tokio::sync::{broadcast, Semaphore, RwLock};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::error;
use crate::net::{Connection, Database, Handler};
use crate::lsh::vector::Vector;

pub struct Listener<DB> 
where
    DB: Database + Sync + Send + 'static,
    DB::Item: Vector<DType=f32> + Sync + Send,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    pub database: Arc<RwLock<DB>>,
    pub listener: TcpListener,
    pub connection_limiter: Arc<Semaphore>,
    pub shutdown_signal: broadcast::Sender<()>,
}

impl<DB> Listener<DB>
where
    DB: Database + Sync + Send + 'static,
    DB::Item: Vector<DType=f32> + Sync + Send,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    pub async fn run(&mut self) -> crate::Result<()> {
        loop {
            self.connection_limiter.acquire().await.unwrap().forget();
            
            // TODO: Implement exponential backoff
            let (socket, _) = self.listener.accept().await.unwrap();
            
            let mut handler = Handler {
                database: self.database.clone(),
                connection: Connection::new(socket),
                connection_limiter: self.connection_limiter.clone()
                //shutdown: ShutdownSignal::new(self.shutdown.subscribe()),
            };

            tokio::spawn(async move {
                if let Err(err) = handler.run().await {
                    println!("ERROR: {:?}", err);
                    error!(cause = ?err, "error handling connection");
                }
            }); 
        }
    }
}
