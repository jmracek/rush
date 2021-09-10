use tokio::sync::RwLock;
use std::sync::Arc;
use crate::net::{Frame, IndexedFrame, Database};
use bytes::Bytes;
use crate::lsh::vector::Vector;


pub(crate) struct Put<DB: Database> 
where
    DB::Item: Vector<DType=f32>,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    dataset: String,
    item: Option<DB::Item>
}

impl<DB: Database> Put<DB> 
where
    DB::Item: Vector<DType=f32>,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>,
{
    pub(crate) async fn execute(self, id: usize, db_ptr: Arc<RwLock<DB>>, tx: tokio::sync::mpsc::Sender<IndexedFrame>) 
        -> crate::Result<()> 
    {
        let item = self.item.unwrap(); 
        // We drop the write lock ASAP to keep the the locked segment tight.
        let mut db = db_ptr.write().await;
        let success = db.insert(item);
        drop(db); 

        let resp = match success { 
            Ok(_) => Frame::Simple("OK".into()),
            _ => Frame::Error("unknown error inserting to LSH database".into()),
        };

        tx.send(IndexedFrame::new(id, resp)).await?; 
        Ok(())
    }

    pub(crate) fn from_blob(dataset: String, _bytes: Bytes) -> Self {
        Put {
            dataset,
            item: Some(DB::Item::default())
        }
    }
}
