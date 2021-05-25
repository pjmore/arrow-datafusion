
use crate::physical_plan::ColumnarValue;
use crate::error::Result;
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
pub struct ColumnExpr;


impl ColumnExpr{
    pub fn evaluate(name: &str, batch: &RecordBatch)->Result<ColumnarValue>{
        Ok(ColumnarValue::Array(
            batch.column(batch.schema().index_of(&name)?).clone(),
        ))
    }
    pub fn nullable(name: &str, input_schema: &Schema) -> Result<bool> {
        Ok(input_schema.field_with_name(name)?.is_nullable())
    }
    pub fn data_type(name: &str, input_schema: &Schema )->Result<DataType>{
        Ok(input_schema
            .field_with_name(&name)?
            .data_type()
            .clone())
    }

}