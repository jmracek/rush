use std::mem;
use std::sync::Arc;
use crate::net::{Frame, IndexedFrame, Database};
use bytes::{Bytes, BytesMut, BufMut};
use crate::lsh::vector::Vector;


pub(crate) struct Get<DB: Database> 
where
    DB::Item: Vector<DType=f32>,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>
{
    dataset: String,
    item: DB::Item
}

impl<DB: Database> Get<DB> 
where
    DB::Item: Vector<DType=f32>,
    for <'a> &'a DB::Item: IntoIterator<Item=<DB::Item as Vector>::DType>,
{
    pub(crate) async fn execute(self, id: usize, db: Arc<DB>, tx: tokio::sync::mpsc::Sender<IndexedFrame>) -> crate::Result<()> {
        let resp = if let Some(value) = db.query(&self.item) {
            let mut buf = BytesMut::with_capacity(mem::size_of::<f32>() * value.dimension());
            for elt in value {
                buf.put_f32_le(elt);
            }
            Frame::Bulk(Bytes::from(buf))
        }
        else {
            Frame::Null()
        };
        tx.send(IndexedFrame::new(id, resp)).await?; 
        Ok(())
    }

    pub(crate) fn from_blob(dataset: String, bytes: Bytes) -> Self {
        Get {
            dataset,
            item: DB::Item::default() 
        }
    }
}
