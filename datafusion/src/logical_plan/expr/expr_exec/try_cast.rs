use crate::physical_plan::ColumnarValue;
use crate::error::Result;
use arrow::{compute::CastOptions, datatypes::{DataType, Schema}, record_batch::RecordBatch};
use arrow::array::BooleanArray;
use crate::scalar::ScalarValue;
use crate::error::DataFusionError;
use std::sync::Arc;
use arrow::array::{Int8Array, Int16Array, Int32Array, Int64Array, Float32Array, Float64Array}; 
use arrow::array::ArrayRef;
use arrow::compute::kernels;

pub struct TryCastExpr;



impl TryCastExpr{
    pub fn evaluate(value: ColumnarValue, cast_type: &DataType)->Result<ColumnarValue>{
        match value {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(kernels::cast::cast(
                &array,
                &cast_type,
            )?)),
            ColumnarValue::Scalar(scalar) => {
                let scalar_array = scalar.to_array();
                let cast_array = kernels::cast::cast(&scalar_array, &cast_type)?;
                let cast_scalar = ScalarValue::try_from_array(&cast_array, 0)?;
                Ok(ColumnarValue::Scalar(cast_scalar))
            }
        }
    }
    pub fn data_type(cast_type: &DataType)->DataType{
        cast_type.clone()
    }
}