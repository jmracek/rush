use tokio::sync::RwLock;
use std::sync::Arc;
use crate::lsh::vector::Vector;

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
    fn insert(&mut self, item: Self::Item) -> crate::Result<()>;
    fn query<'a>(&'a self, item: &Self::Item) -> Option<&'a Self::Item>;
}

mod get;
use get::Get;

mod put;
use put::Put;

// TODO: Implement ability to publish from S3 locations
//mod publish;
//use publish::Publish;

pub(crate) enum Command<DB: Database> 
where
    <DB as Database>::Item: Vector<DType=f32> + Send + Sync,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    Get(Get<DB>),
    Put(Put<DB>),
    //Publish(Publish<DB>),
}

impl<DB: Database> Command<DB> 
where
    <DB as Database>::Item: Vector<DType=f32> + Send + Sync,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    pub(crate) async fn execute(self, id: usize, db: Arc<RwLock<DB>>, ch: tokio::sync::mpsc::Sender<IndexedFrame>) -> crate::Result<()> {
        match self {
            Command::Get(cmd) => cmd.execute(id, db, ch).await,
            Command::Put(cmd) => cmd.execute(id, db, ch).await,
            //Command::Publish(cmd) => cmd.execute(id, db, ch).await,
        }
    }

    fn parse(array: Vec<Frame>) -> crate::Result<Self> {
        let mut it = array.into_iter();
        
        let command_name = match it.next().unwrap() {
            Frame::Simple(cmd) => cmd.to_lowercase(),
            _ => return Err("protocol error; expected command name".into())
        };
        
        // TODO: For now, this doesn't matter.  Later I'm going to make it matter.
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
            "put" => Command::Put(Put::<DB>::from_blob(dataset, blob)),
            //"publish" => Command::Publish(Publish::new(dataset, location)),
            _ => return Err("parse error; unrecognized command".into()),
        };
        
        Ok(command)
    }
}
