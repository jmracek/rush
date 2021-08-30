use tokio::sync::{mpsc, Semaphore};
use tokio::task;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use crate::net::{Connection, Database, Frame, IndexedFrame, Command};
use crate::lsh::vector::Vector;

enum Mode {
    Stream = 0,
    Bulk = 1,
    Single = 2
}

impl Mode {
    fn from_u64(x: u64) -> Option<Self> {
        match x {
            0u64 => Some(Mode::Stream),
            1u64 => Some(Mode::Bulk),
            2u64 => Some(Mode::Single),
            _    => None
        }
    }
}

pub(crate) struct Handler<DB> 
where
    DB: Database + Sync + Send + 'static,
    DB::Item: Vector<DType=f32>,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    pub(crate) database: Arc<DB>,
    pub(crate) connection: Connection,
    pub(crate) connection_limiter: Arc<Semaphore>,
    //shutdown: ShutdownSignal,
}

/*

Message: 
    - Integer (mode)
    - One of three possitibilities 

STREAM:
Each message...
*3\r\n+GET\n\n+DATASET\r\n$N\r\n[u8 x N]\r\n  OR
*3\r\n+PUT\n\n+DATASET\r\n$N\r\n[u8 x N]\r\n
THEN ON COMPLETE:
$-1\r\n

SINGLE:
*3\r\n+GET\n\n+DATASET\r\n$[N: u32, f32xN]\r\n

BULK:
*N\r\n[ [*3\r\n+GET/PUT\r\n+DATASET\r\n$N\r\n[u8 x N]\r\n] : N times]

*/

impl<DB> Handler<DB> 
where
    DB: Database + Sync + Send + 'static,
    DB::Item: Vector<DType=f32> + Sync + Send,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{

    pub(crate) async fn run(&mut self) -> crate::Result<()> {
        let maybe_mode = match self.connection.read_frame().await? {
            Some(Frame::Integer(mode)) => Mode::from_u64(mode),
            None => return Ok(()),
            _ => return Err("protocol error; expected integer frame".into()), 
        };

        let mode = match maybe_mode {
            Some(mode) => mode,
            None => return Err("protocol error; expected mode in [0, 1, 2]".into()),
        };

        match mode {
            Mode::Stream => self.handle_stream().await,
            Mode::Bulk => self.handle_bulk().await,
            Mode::Single => self.handle_single().await,
        }
    }
    
    // This function awaits Array frames until receiving a Null frame, at which point we exit.
    async fn handle_stream(&mut self) -> crate::Result<()> {
        let (tx, mut rx) = mpsc::channel(64);
        let mut counter: usize = 0;

        while let Some(frame) = self.connection.read_frame().await? {
            let arr = match frame {
                Frame::Array(array) => array, 
                Frame::Null() => return Ok(()),
                _ => return Err("protocol error; streaming frames must be either Array or Null".into())
            };

            let cmd = Command::<DB>::parse(arr)?;

            let db = self.database.clone();
            let txx = tx.clone();
            let id = counter; 

            task::spawn(async move {
                cmd.execute(id, db, txx).await;
            });
            
            // Wait for the response we'll need to send
            if let Some(response) = rx.recv().await {
                self.connection.write_frame(response.get_frame()).await?;
            }

            counter += 1;

            /* When I implement piplelining, I can replace above with while let and dispatch an index to
               each subtask to remember the caller order */
        }

        Ok(())
    }

    async fn handle_bulk(&mut self) -> crate::Result<()> {
        let (tx, mut rx) = mpsc::channel(64);
        if let Some(frame) = self.connection.read_frame().await? {
            let frames = match frame {
                Frame::Array(array) => array, 
                _ => return Err("protocol error; bulk mode frame must be Array".into())
            };
            
            for (id, frame) in frames.into_iter().enumerate() {
                let array = match frame {
                    Frame::Array(array) => array,
                    _ => return Err("protocol error; bulk mode commands must be Array frames".into())
                };
                let cmd = Command::<DB>::parse(array)?;
                let db = self.database.clone();
                let txx = tx.clone();

                task::spawn(async move {
                    cmd.execute(id, db, txx).await;
                });
            }
            
            let mut responses = BinaryHeap::new();
            
            // Collect all the responses
            while let Some(response) = rx.recv().await {
                responses.push(Reverse(response))
            }
            
            // Sort them so the order agrees with the sender's order
            let mut resp = Vec::<Frame>::new();
            while let Some(Reverse(response)) = responses.pop() {
                resp.push(Frame::from(response)); 
            }
            // Write the response back to the sender 
            self.connection.write_frame(&Frame::Array(resp)).await?;
        }

        Ok(())
    }

    async fn handle_single(&mut self) -> crate::Result<()> {
        Ok(())
    }
}
