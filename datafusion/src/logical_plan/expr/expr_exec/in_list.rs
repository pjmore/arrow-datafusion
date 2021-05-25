use crate::physical_plan::ColumnarValue;
use crate::error::Result;
use arrow::{compute::CastOptions, datatypes::{DataType, Schema}, record_batch::RecordBatch};
use arrow::array::BooleanArray;
use crate::scalar::ScalarValue;
use crate::error::DataFusionError;
use std::sync::Arc;
use arrow::array::{Int8Array, Int16Array, Int32Array, Int64Array, Float32Array, Float64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array}; 
use arrow::array::ArrayRef;



pub struct InListExpr;

macro_rules! make_contains {
    ($ARRAY:expr, $LIST_VALUES:expr, $NEGATED:expr, $SCALAR_VALUE:ident, $ARRAY_TYPE:ident) => {{
        let array = $ARRAY.as_any().downcast_ref::<$ARRAY_TYPE>().unwrap();

        let mut contains_null = false;
        let values = $LIST_VALUES
            .iter()
            .flat_map(|expr| match expr {
                ColumnarValue::Scalar(s) => match s {
                    ScalarValue::$SCALAR_VALUE(Some(v)) => Some(*v),
                    ScalarValue::$SCALAR_VALUE(None) => {
                        contains_null = true;
                        None
                    }
                    ScalarValue::Utf8(None) => {
                        contains_null = true;
                        None
                    }
                    datatype => unimplemented!("Unexpected type {} for InList", datatype),
                },
                ColumnarValue::Array(_) => {
                    unimplemented!("InList does not yet support nested columns.")
                }
            })
            .collect::<Vec<_>>();

        Ok(ColumnarValue::Array(Arc::new(
            array
                .iter()
                .map(|x| {
                    let contains = x.map(|x| values.contains(&x));
                    match contains {
                        Some(true) => {
                            if $NEGATED {
                                Some(false)
                            } else {
                                Some(true)
                            }
                        }
                        Some(false) => {
                            if contains_null {
                                None
                            } else if $NEGATED {
                                Some(true)
                            } else {
                                Some(false)
                            }
                        }
                        None => None,
                    }
                })
                .collect::<BooleanArray>(),
        )))
    }};
}




impl InListExpr{
    pub fn evaluate(value: ColumnarValue, list_values: &[ColumnarValue], negated: bool)->Result<ColumnarValue>{
        let value_data_type = value.data_type();
        let array = match value {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array(),
        };

        match value_data_type {
            DataType::Float32 => {
                make_contains!(array, list_values, negated, Float32, Float32Array)
            }
            DataType::Float64 => {
                make_contains!(array, list_values, negated, Float64, Float64Array)
            }
            DataType::Int16 => {
                make_contains!(array, list_values, negated, Int16, Int16Array)
            }
            DataType::Int32 => {
                make_contains!(array, list_values, negated, Int32, Int32Array)
            }
            DataType::Int64 => {
                make_contains!(array, list_values, negated, Int64, Int64Array)
            }
            DataType::Int8 => {
                make_contains!(array, list_values, negated, Int8, Int8Array)
            }
            DataType::UInt16 => {
                make_contains!(array, list_values, negated, UInt16, UInt16Array)
            }
            DataType::UInt32 => {
                make_contains!(array, list_values, negated, UInt32, UInt32Array)
            }
            DataType::UInt64 => {
                make_contains!(array, list_values, negated, UInt64, UInt64Array)
            }
            DataType::UInt8 => {
                make_contains!(array, list_values, negated, UInt8, UInt8Array)
            }
            DataType::Boolean => {
                make_contains!(array, list_values, negated, Boolean, BooleanArray)
            }
            DataType::Utf8 => compare_utf8::<i32>(array, list_values, negated),
            DataType::LargeUtf8 => {
                compare_utf8::<i64>(array, list_values, negated)
            }
            datatype => {
                unimplemented!("InList does not support datatype {:?}.", datatype)
            }
        }

    }
}


use arrow::array::GenericStringArray;
use arrow::array::StringOffsetSizeTrait;
/// Compare for specific utf8 types
#[allow(clippy::unnecessary_wraps)]
fn compare_utf8<T: StringOffsetSizeTrait>(
    array: ArrayRef,
    list_values: &[ColumnarValue],
    negated: bool,
) -> Result<ColumnarValue> {
    let array = array
        .as_any()
        .downcast_ref::<GenericStringArray<T>>()
        .unwrap();

    let mut contains_null = false;
    let values = list_values
        .iter()
        .flat_map(|expr| match expr {
            ColumnarValue::Scalar(s) => match s {
                ScalarValue::Utf8(Some(v)) => Some(v.as_str()),
                ScalarValue::Utf8(None) => {
                    contains_null = true;
                    None
                }
                ScalarValue::LargeUtf8(Some(v)) => Some(v.as_str()),
                ScalarValue::LargeUtf8(None) => {
                    contains_null = true;
                    None
                }
                datatype => unimplemented!("Unexpected type {} for InList", datatype),
            },
            ColumnarValue::Array(_) => {
                unimplemented!("InList does not yet support nested columns.")
            }
        })
        .collect::<Vec<&str>>();

    Ok(ColumnarValue::Array(Arc::new(
        array
            .iter()
            .map(|x| {
                let contains = x.map(|x| values.contains(&x));
                match contains {
                    Some(true) => {
                        if negated {
                            Some(false)
                        } else {
                            Some(true)
                        }
                    }
                    Some(false) => {
                        if contains_null {
                            None
                        } else if negated {
                            Some(true)
                        } else {
                            Some(false)
                        }
                    }
                    None => None,
                }
            })
            .collect::<BooleanArray>(),
    )))
}