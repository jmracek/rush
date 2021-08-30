mod connection;
pub(crate) use connection::Connection;

mod frame;
pub(crate) use frame::Frame;
pub(crate) use frame::IndexedFrame;

mod handler;
pub(crate) use handler::Handler;

mod listener;
pub use listener::Listener;

pub trait Database {
    type Item;
    fn len(&self) -> usize;
    fn insert(&mut self, item: Self::Item);
    fn query<'a>(&'a self, item: &Self::Item) -> Option<&'a Self::Item>;
}

mod get;
use get::Get;

use std::sync::Arc;
use crate::lsh::vector::Vector;
use bytes::Bytes;

pub(crate) enum Command<DB: Database> 
where
    <DB as Database>::Item: Vector<DType=f32> + Send + Sync,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    Get(Get<DB>),
//    Put(Put<T>),
}

impl<DB: Database> Command<DB> 
where
    <DB as Database>::Item: Vector<DType=f32> + Send + Sync,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    pub(crate) async fn execute(self, id: usize, db: Arc<DB>, ch: tokio::sync::mpsc::Sender<IndexedFrame>) -> crate::Result<()> {
        match self {
            Command::Get(cmd) => cmd.execute(id, db, ch).await,
 //           Put(cmd) => cmd.execute(id, db, ch).await,
        }
    }

    fn parse(array: Vec<Frame>) -> crate::Result<Self> {
        let mut it = array.into_iter();
        
        let command_name = match it.next().unwrap() {
            Frame::Simple(cmd) => cmd.to_lowercase(),
            _ => return Err("protocol error; expected command name".into())
        };
        
        // For now, this doesn't matter.  Later I'm going to make it matter.
        let dataset = match it.next().unwrap() {
            Frame::Simple(ds) => ds.to_lowercase(),
            _ => return Err("protocol error; expected dataset name".into())
        };

        let blob = match it.next().unwrap() {
            Frame::Bulk(data) => data,
            _ => return Err("protocol error; expected dataset name".into())
        };

        let command = match &command_name[..] {
            "get" => Command::Get(Get::<DB>::from_blob(dataset, blob)),
            _ => return Err("parse error; unrecognized command".into()),
        };
        
        Ok(command)
    }
}
