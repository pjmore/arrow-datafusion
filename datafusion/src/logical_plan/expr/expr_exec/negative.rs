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
use arrow::compute::negate;
use arrow::array::{Int8Array, Int16Array, Int32Array, Int64Array, Float32Array, Float64Array}; 
use arrow::array::ArrayRef;

pub struct NegativeExpr;


/// Invoke a compute kernel on array(s)
macro_rules! compute_op {
    // invoke unary operator
    ($OPERAND:expr, $OP:ident, $DT:ident) => {{
        let operand = $OPERAND
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new($OP(&operand)?))
    }};
}


impl NegativeExpr{
    pub fn evaluate(value: ColumnarValue, sub_expr_name: &str)->Result<ColumnarValue>{
        match value {
            ColumnarValue::Array(array) => {
                let result: Result<ArrayRef> = match array.data_type() {
                    DataType::Int8 => compute_op!(array, negate, Int8Array),
                    DataType::Int16 => compute_op!(array, negate, Int16Array),
                    DataType::Int32 => compute_op!(array, negate, Int32Array),
                    DataType::Int64 => compute_op!(array, negate, Int64Array),
                    DataType::Float32 => compute_op!(array, negate, Float32Array),
                    DataType::Float64 => compute_op!(array, negate, Float64Array),
                    _ => Err(DataFusionError::Internal(format!(
                        "(- '{:?}') can't be evaluated because the expression's type is {:?}, not signed numeric",
                        sub_expr_name,
                        array.data_type(),
                    ))),
                };
                result.map(|a| ColumnarValue::Array(a))
            }
            ColumnarValue::Scalar(scalar) => {
                Ok(ColumnarValue::Scalar(scalar.arithmetic_negate()))
            }
        }
    }
}