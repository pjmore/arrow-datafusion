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

pub struct NotExpr;


impl NotExpr{
    pub fn evaluate(value: ColumnarValue)->Result<ColumnarValue>{
        match value {
            ColumnarValue::Array(array) => {
                let array =
                    array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "boolean_op failed to downcast array".to_owned(),
                            )
                        })?;
                Ok(ColumnarValue::Array(Arc::new(
                    arrow::compute::kernels::boolean::not(array)?,
                )))
            }
            ColumnarValue::Scalar(scalar) => {
                use std::convert::TryInto;
                let bool_value: bool = scalar.try_into()?;
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(
                    !bool_value,
                ))))
            }
        }
    }

    pub fn data_type()->DataType{
        DataType::Boolean
    }

}