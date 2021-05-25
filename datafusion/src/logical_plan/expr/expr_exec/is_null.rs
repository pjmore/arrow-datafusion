use crate::physical_plan::ColumnarValue;
use crate::error::Result;
use arrow::{
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use arrow::array::BooleanArray;
use crate::scalar::ScalarValue;
use crate::error::DataFusionError;
use std::sync::Arc;
use arrow::compute;


pub struct IsNullExpr;

impl IsNullExpr{
    pub fn evaluate(value: ColumnarValue)->Result<ColumnarValue>{
        match value {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(Arc::new(
                compute::is_null(array.as_ref())?,
            ))),
            ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(
                ScalarValue::Boolean(Some(scalar.is_null())),
            )),
        }
    }

    pub fn datatype()->DataType{
        DataType::Boolean
    }
}