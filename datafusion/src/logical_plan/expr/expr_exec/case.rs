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

pub struct CaseExpr;

impl CaseExpr {
    pub fn case_when_with_expr<T: std::borrow::Borrow<dyn PhysicalExpr>>( expr: Option<Arc<dyn PhysicalExpr>>, when_then_expr: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)>, else_expr: Option<Arc<dyn PhysicalExpr>>);


    /// This function evaluates the form of CASE that matches an expression to fixed values.
    ///
    /// CASE expression
    ///     WHEN value THEN result
    ///     [WHEN ...]
    ///     [ELSE result]
    /// END
    fn case_when_with_expr(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let return_type = self.when_then_expr[0].1.data_type(&batch.schema())?;
        let expr = self.expr.as_ref().unwrap();
        let base_value = expr.evaluate(batch)?;
        let base_type = expr.data_type(&batch.schema())?;
        let base_value = base_value.into_array(batch.num_rows());

        // start with the else condition, or nulls
        let mut current_value: Option<ArrayRef> = if let Some(e) = &self.else_expr {
            Some(e.evaluate(batch)?.into_array(batch.num_rows()))
        } else {
            Some(new_null_array(&return_type, batch.num_rows()))
        };

        // walk backwards through the when/then expressions
        for i in (0..self.when_then_expr.len()).rev() {
            let i = i as usize;

            let when_value = self.when_then_expr[i].0.evaluate(batch)?;
            let when_value = when_value.into_array(batch.num_rows());

            let then_value = self.when_then_expr[i].1.evaluate(batch)?;
            let then_value = then_value.into_array(batch.num_rows());

            // build boolean array representing which rows match the "when" value
            let when_match = array_equals(&base_type, when_value, base_value.clone())?;

            current_value = Some(if_then_else(
                &when_match,
                then_value,
                current_value.unwrap(),
                &return_type,
            )?);
        }

        Ok(ColumnarValue::Array(current_value.unwrap()))
    }

    /// This function evaluates the form of CASE where each WHEN expression is a boolean
    /// expression.
    ///
    /// CASE WHEN condition THEN result
    ///      [WHEN ...]
    ///      [ELSE result]
    /// END
    fn case_when_no_expr(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let return_type = self.when_then_expr[0].1.data_type(&batch.schema())?;

        // start with the else condition, or nulls
        let mut current_value: Option<ArrayRef> = if let Some(e) = &self.else_expr {
            Some(e.evaluate(batch)?.into_array(batch.num_rows()))
        } else {
            Some(new_null_array(&return_type, batch.num_rows()))
        };

        // walk backwards through the when/then expressions
        for i in (0..self.when_then_expr.len()).rev() {
            let i = i as usize;

            let when_value = self.when_then_expr[i].0.evaluate(batch)?;
            let when_value = when_value.into_array(batch.num_rows());
            let when_value = when_value
                .as_ref()
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("WHEN expression did not return a BooleanArray");

            let then_value = self.when_then_expr[i].1.evaluate(batch)?;
            let then_value = then_value.into_array(batch.num_rows());

            current_value = Some(if_then_else(
                &when_value,
                then_value,
                current_value.unwrap(),
                &return_type,
            )?);
        }

        Ok(ColumnarValue::Array(current_value.unwrap()))
    }
}
