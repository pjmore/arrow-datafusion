

use std::{sync::Arc};

use arrow::array::*;
use arrow::compute::kernels::arithmetic::{
    add, divide, divide_scalar, multiply, subtract,
};
use arrow::compute::kernels::boolean::{and_kleene, or_kleene};
use arrow::compute::kernels::comparison::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow::compute::kernels::comparison::{
    eq_scalar, gt_eq_scalar, gt_scalar, lt_eq_scalar, lt_scalar, neq_scalar,
};
use arrow::compute::kernels::comparison::{
    eq_utf8, gt_eq_utf8, gt_utf8, like_utf8, like_utf8_scalar, lt_eq_utf8, lt_utf8,
    neq_utf8, nlike_utf8, nlike_utf8_scalar,
};
use arrow::compute::kernels::comparison::{
    eq_utf8_scalar, gt_eq_utf8_scalar, gt_utf8_scalar, lt_eq_utf8_scalar, lt_utf8_scalar,
    neq_utf8_scalar,
};
use arrow::datatypes::{DataType, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use crate::scalar::ScalarValue;
use crate::error::{DataFusionError, Result};
use crate::physical_plan::{PhysicalExpr, ColumnarValue};
use crate::logical_plan::Operator;
use crate::physical_plan::expressions::coercion::{eq_coercion, order_coercion, string_coercion, numerical_coercion};
/// Invoke a compute kernel on a pair of binary data arrays
macro_rules! compute_utf8_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new(paste::expr! {[<$OP _utf8>]}(&ll, &rr)?))
    }};
}

/// Invoke a compute kernel on a data array and a scalar value
macro_rules! compute_utf8_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        if let ScalarValue::Utf8(Some(string_value)) = $RIGHT {
            Ok(Arc::new(paste::expr! {[<$OP _utf8_scalar>]}(
                &ll,
                &string_value,
            )?))
        } else {
            Err(DataFusionError::Internal(format!(
                "compute_utf8_op_scalar failed to cast literal value {}",
                $RIGHT
            )))
        }
    }};
}

/// Invoke a compute kernel on a data array and a scalar value
macro_rules! compute_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        use std::convert::TryInto;
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        // generate the scalar function name, such as lt_scalar, from the $OP parameter
        // (which could have a value of lt) and the suffix _scalar
        Ok(Arc::new(paste::expr! {[<$OP _scalar>]}(
            &ll,
            $RIGHT.try_into()?,
        )?))
    }};
}

/// Invoke a compute kernel on array(s)
macro_rules! compute_op {
    // invoke binary operator
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
    // invoke unary operator
    ($OPERAND:expr, $OP:ident, $DT:ident) => {{
        let operand = $OPERAND
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new($OP(&operand)?))
    }};
}

macro_rules! binary_string_array_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let result: Result<Arc<dyn Array>> = match $LEFT.data_type() {
            DataType::Utf8 => compute_utf8_op_scalar!($LEFT, $RIGHT, $OP, StringArray),
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for scalar operation on string array",
                other
            ))),
        };
        Some(result)
    }};
}

macro_rules! binary_string_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary operation on string arrays",
                other
            ))),
        }
    }};
}

/// Invoke a compute kernel on a pair of arrays
/// The binary_primitive_array_op macro only evaluates for primitive types
/// like integers and floats.
macro_rules! binary_primitive_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Int8 => compute_op!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op!($LEFT, $RIGHT, $OP, Float64Array),
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary operation on primitive arrays",
                other
            ))),
        }
    }};
}

/// Invoke a compute kernel on an array and a scalar
/// The binary_primitive_array_op_scalar macro only evaluates for primitive
/// types like integers and floats.
macro_rules! binary_primitive_array_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let result: Result<Arc<dyn Array>> = match $LEFT.data_type() {
            DataType::Int8 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op_scalar!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op_scalar!($LEFT, $RIGHT, $OP, Float64Array),
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for scalar operation on primitive array",
                other
            ))),
        };
        Some(result)
    }};
}

/// The binary_array_op_scalar macro includes types that extend beyond the primitive,
/// such as Utf8 strings.
macro_rules! binary_array_op_scalar {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let result: Result<Arc<dyn Array>> = match $LEFT.data_type() {
            DataType::Int8 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op_scalar!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op_scalar!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op_scalar!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op_scalar!($LEFT, $RIGHT, $OP, Float64Array),
            DataType::Utf8 => compute_utf8_op_scalar!($LEFT, $RIGHT, $OP, StringArray),
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                compute_op_scalar!($LEFT, $RIGHT, $OP, TimestampNanosecondArray)
            }
            DataType::Date32 => {
                compute_op_scalar!($LEFT, $RIGHT, $OP, Date32Array)
            }
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for scalar operation on dyn array",
                other
            ))),
        };
        Some(result)
    }};
}

/// The binary_array_op macro includes types that extend beyond the primitive,
/// such as Utf8 strings.
macro_rules! binary_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Int8 => compute_op!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op!($LEFT, $RIGHT, $OP, Float64Array),
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                compute_op!($LEFT, $RIGHT, $OP, TimestampNanosecondArray)
            }
            DataType::Date32 => {
                compute_op!($LEFT, $RIGHT, $OP, Date32Array)
            }
            DataType::Date64 => {
                compute_op!($LEFT, $RIGHT, $OP, Date64Array)
            }
            other => Err(DataFusionError::Internal(format!(
                "Data type {:?} not supported for binary operation on dyn arrays",
                other
            ))),
        }
    }};
}

/// Invoke a boolean kernel on a pair of arrays
macro_rules! boolean_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}

pub struct BinaryExpr;

fn common_binary_type(
    lhs_type: &DataType,
    op: &Operator,
    rhs_type: &DataType,
) -> Result<DataType> {
    // This result MUST be compatible with `binary_coerce`
    let result = match op {
        Operator::And | Operator::Or => match (lhs_type, rhs_type) {
            // logical binary boolean operators can only be evaluated in bools
            (DataType::Boolean, DataType::Boolean) => Some(DataType::Boolean),
            _ => None,
        },
        // logical equality operators have their own rules, and always return a boolean
        Operator::Eq | Operator::NotEq => eq_coercion(lhs_type, rhs_type),
        // "like" operators operate on strings and always return a boolean
        Operator::Like | Operator::NotLike => string_coercion(lhs_type, rhs_type),
        // order-comparison operators have their own rules
        Operator::Lt | Operator::Gt | Operator::GtEq | Operator::LtEq => {
            order_coercion(lhs_type, rhs_type)
        }
        // for math expressions, the final value of the coercion is also the return type
        // because coercion favours higher information types
        Operator::Plus | Operator::Minus | Operator::Divide | Operator::Multiply => {
            numerical_coercion(lhs_type, rhs_type)
        }
        Operator::Modulus => {
            return Err(DataFusionError::NotImplemented(
                "Modulus operator is still not supported".to_string(),
            ))
        }
    };

    // re-write the error message of failed coercions to include the operator's information
    match result {
        None => Err(DataFusionError::Plan(
            format!(
                "'{:?} {} {:?}' can't be evaluated because there isn't a common type to coerce the types to",
                lhs_type, op, rhs_type
            ),
        )),
        Some(t) => Ok(t)
    }
}


impl BinaryExpr{
    pub fn data_type(
        lhs_type: &DataType,
        op: &Operator,
        rhs_type: &DataType,
    ) -> Result<DataType> {
        // validate that it is possible to perform the operation on incoming types.
        // (or the return datatype cannot be infered)
        let common_type = common_binary_type(lhs_type, op, rhs_type)?;
    
        match op {
            // operators that return a boolean
            Operator::Eq
            | Operator::NotEq
            | Operator::And
            | Operator::Or
            | Operator::Like
            | Operator::NotLike
            | Operator::Lt
            | Operator::Gt
            | Operator::GtEq
            | Operator::LtEq => Ok(DataType::Boolean),
            // math operations return the same value as the common coerced type
            Operator::Plus | Operator::Minus | Operator::Divide | Operator::Multiply => {
                Ok(common_type)
            }
            Operator::Modulus => Err(DataFusionError::NotImplemented(
                "Modulus operator is still not supported".to_string(),
            )),
        }
    }
    


    pub fn evaluate_binop(left_value: ColumnarValue, op: &Operator, right_value: ColumnarValue, num_rows: usize)->Result<ColumnarValue>{
        let left_data_type = left_value.data_type();
        let right_data_type = right_value.data_type();
    
        if left_data_type != right_data_type {
            return Err(DataFusionError::Internal(format!(
                "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                op, left_data_type, right_data_type
            )));
        }
    
        let scalar_result = match (&left_value, &right_value) {
            (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar)) => {
                // if left is array and right is literal - use scalar operations
                match &op {
                    Operator::Lt => binary_array_op_scalar!(array, scalar.clone(), lt),
                    Operator::LtEq => {
                        binary_array_op_scalar!(array, scalar.clone(), lt_eq)
                    }
                    Operator::Gt => binary_array_op_scalar!(array, scalar.clone(), gt),
                    Operator::GtEq => {
                        binary_array_op_scalar!(array, scalar.clone(), gt_eq)
                    }
                    Operator::Eq => binary_array_op_scalar!(array, scalar.clone(), eq),
                    Operator::NotEq => {
                        binary_array_op_scalar!(array, scalar.clone(), neq)
                    }
                    Operator::Like => {
                        binary_string_array_op_scalar!(array, scalar.clone(), like)
                    }
                    Operator::NotLike => {
                        binary_string_array_op_scalar!(array, scalar.clone(), nlike)
                    }
                    Operator::Divide => {
                        binary_primitive_array_op_scalar!(array, scalar.clone(), divide)
                    }
                    // if scalar operation is not supported - fallback to array implementation
                    _ => None,
                }
            }
            (ColumnarValue::Scalar(scalar), ColumnarValue::Array(array)) => {
                // if right is literal and left is array - reverse operator and parameters
                match &op {
                    Operator::Lt => binary_array_op_scalar!(array, scalar.clone(), gt),
                    Operator::LtEq => {
                        binary_array_op_scalar!(array, scalar.clone(), gt_eq)
                    }
                    Operator::Gt => binary_array_op_scalar!(array, scalar.clone(), lt),
                    Operator::GtEq => {
                        binary_array_op_scalar!(array, scalar.clone(), lt_eq)
                    }
                    Operator::Eq => binary_array_op_scalar!(array, scalar.clone(), eq),
                    Operator::NotEq => {
                        binary_array_op_scalar!(array, scalar.clone(), neq)
                    }
                    // if scalar operation is not supported - fallback to array implementation
                    _ => None,
                }
            }
            (_, _) => None,
        };
    
        if let Some(result) = scalar_result {
            return result.map(|a| ColumnarValue::Array(a));
        }
    
        // if both arrays or both literals - extract arrays and continue execution
        let (left, right) = (
            left_value.into_array(num_rows),
            right_value.into_array(num_rows),
        );
    
        let result: Result<ArrayRef> = match &op {
            Operator::Like => binary_string_array_op!(left, right, like),
            Operator::NotLike => binary_string_array_op!(left, right, nlike),
            Operator::Lt => binary_array_op!(left, right, lt),
            Operator::LtEq => binary_array_op!(left, right, lt_eq),
            Operator::Gt => binary_array_op!(left, right, gt),
            Operator::GtEq => binary_array_op!(left, right, gt_eq),
            Operator::Eq => binary_array_op!(left, right, eq),
            Operator::NotEq => binary_array_op!(left, right, neq),
            Operator::Plus => binary_primitive_array_op!(left, right, add),
            Operator::Minus => binary_primitive_array_op!(left, right, subtract),
            Operator::Multiply => binary_primitive_array_op!(left, right, multiply),
            Operator::Divide => binary_primitive_array_op!(left, right, divide),
            Operator::And => {
                if left_data_type == DataType::Boolean {
                    boolean_op!(left, right, and_kleene)
                } else {
                    return Err(DataFusionError::Internal(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        op,
                        left.data_type(),
                        right.data_type()
                    )));
                }
            }
            Operator::Or => {
                if left_data_type == DataType::Boolean {
                    boolean_op!(left, right, or_kleene)
                } else {
                    return Err(DataFusionError::Internal(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        op, left_data_type, right_data_type
                    )));
                }
            }
            Operator::Modulus => Err(DataFusionError::NotImplemented(
                "Modulus operator is still not supported".to_string(),
            )),
        };
        result.map(|a| ColumnarValue::Array(a))
    }
}


