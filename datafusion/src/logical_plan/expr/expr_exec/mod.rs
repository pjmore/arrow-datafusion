use crate::{error::{Result, DataFusionError}, physical_plan::window_functions::BuiltInWindowFunction};
use super::super::Operator;
use super::Expr;
use super::ExprRewriter;
use std::fmt;
use std::sync::Arc;
use crate::execution::context::ExecutionProps;

use aggregates::{AccumulatorFunctionImplementation, StateTypeFunction};
use arrow::{compute::{CastOptions, can_cast_types}, datatypes::DataType};
use arrow::record_batch::RecordBatch;

use crate::logical_plan::{DFField, DFSchema, DFSchemaRef};
use crate::physical_plan::{
    aggregates, expressions::binary_operator_data_type, functions, udf::ScalarUDF,
    window_functions,
};
use crate::{physical_plan::udaf::AggregateUDF, scalar::ScalarValue};
use crate::physical_plan::functions::{ReturnTypeFunction, ScalarFunctionImplementation, Signature};
use std::collections::HashSet;
use crate::physical_plan::ColumnarValue;

use crate::physical_plan::expressions::coercion;

mod binary;
mod cast;
mod column;
mod not;
mod is_not_null;
mod in_list;
mod is_null;
mod negative;
mod try_cast;



///This function inserts any implicit expressions that used to be placed into a physical expression
/// BinaryExpr inserts TryCastExpr to coerce columns to correct type

pub struct ExprExecPreparation<'sch>{
    all_schema: Vec<&'sch DFSchemaRef>,
    execution_props: &'sch ExecutionProps,
}
pub struct ExecutableExpr(Expr);

fn todo_err()->Result<ColumnarValue>{
    Err(DataFusionError::Internal("PLACEHOLDER ERROR".to_string()))
}

impl ExecutableExpr{
    pub fn new<'sch>(expr: &Expr, all_schema: Vec<&'sch DFSchemaRef>, execution_props: &'sch ExecutionProps)->Result<ExecutableExpr>{
        let mut rewriter = ExprExecPreparation{all_schema, execution_props};
        let rewritten_expr = expr.clone().rewrite(&mut rewriter)?;
        Ok(ExecutableExpr(rewritten_expr))
    }

    fn evaluate_unchecked(unchecked_expr: &Expr, batch: &RecordBatch)->Result<ColumnarValue>{
        match unchecked_expr{
            Expr::Alias(expr, _) => ExecutableExpr::evaluate_unchecked(&expr, batch),
            Expr::Column(name) => column::ColumnExpr::evaluate(&name, batch),
            Expr::ScalarVariable(_) => {
                todo!("Wat do here?")
            }
            Expr::Literal(s) => Ok(ColumnarValue::Scalar(s.clone())),
            Expr::BinaryExpr { left, op, right } => {
                let left_value = ExecutableExpr::evaluate_unchecked(&left, batch)?;
                let right_value = ExecutableExpr::evaluate_unchecked(&right, batch)?;
                binary::BinaryExpr::evaluate_binop(left_value, op, right_value, batch.num_rows())
            }
            Expr::Not(expr) => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                not::NotExpr::evaluate(value)
            }
            Expr::IsNotNull(expr) => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                is_not_null::IsNotNullExpr::evaluate(value)
            }
            Expr::IsNull(expr) => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                is_null::IsNullExpr::evaluate(value)
            }
            Expr::Negative(expr) => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                negative::NegativeExpr::evaluate(value, "PLACEHOLDER")
            }
            Expr::Case { expr, when_then_expr, else_expr } => {
                todo!()
                //cast::CastExpr::evaluate(&expr, &when_then_expr, &else_expr)
            }
            Expr::Cast { expr, data_type } => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                cast::CastExpr::evaluate(value, data_type, &CastOptions{safe: false})
            }
            Expr::TryCast { expr, data_type } => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                try_cast::TryCastExpr::evaluate(value, data_type)
            }
            Expr::ScalarFunction { fun, args } => {
                let arg_values = args.iter().map(|arg| ExecutableExpr::evaluate_unchecked(arg, batch)).collect::<Result<Vec<_>>>()?;
                let fun_impl = crate::physical_plan::functions::make_function_impl(&fun, todo!())?;
                fun_impl(&arg_values)
            }
            Expr::ScalarUDF { fun, args } => {
                let arg_values = args.iter().map(|arg| ExecutableExpr::evaluate_unchecked(arg, batch)).collect::<Result<Vec<_>>>()?;
                (&fun.fun)(&arg_values)
            }
            Expr::AggregateFunction { fun, args, distinct } => {
                todo_err()                 
            }
            Expr::WindowFunction { fun, args } => todo_err(),
            Expr::AggregateUDF { fun, args } => todo_err() ,
            Expr::InList { expr, list, negated } => {
                let value = ExecutableExpr::evaluate_unchecked(&expr, batch)?;
                let list_values = list.iter().map(|item| ExecutableExpr::evaluate_unchecked(&item, batch)).collect::<Result<Vec<_>>>()?;
                in_list::InListExpr::evaluate(value, &list_values, *negated)
            }
            Expr::Sort { .. } => Err(DataFusionError::Internal("Found Sort expression in executable expression, this should never happen.".to_string())),
            Expr::Between { .. } => Err(DataFusionError::Internal("Found Between expression in executable expression, this should always be replaced by the sub expression expr >= low && expr <= high".to_string())),
            Expr::Wildcard => Err(DataFusionError::Internal("Found Wildcard Expression in Executable expression, this should never happen.".to_string())),
        }
    }
    fn nullable(expr: &Expr, input_schema: &arrow::datatypes::Schema)->Result<bool>{
        fn nullable_todo_err()->Result<bool>{
            Err(DataFusionError::Internal("PLACEHOLDER NULLABILITY FUNC".to_string()))
        }
        match expr{
            Expr::Alias(expr, _) => ExecutableExpr::nullable(&expr, input_schema),
            Expr::Column(name) => column::ColumnExpr::nullable(&name, input_schema),
            Expr::ScalarVariable(_) => Ok(false),
            Expr::Literal(s) => Ok(s.is_null()), 
            Expr::BinaryExpr { left, op, right } => {
                Ok(ExecutableExpr::nullable(&left, input_schema)? || ExecutableExpr::nullable(&right, input_schema)?)
            }
            Expr::Not(expr) => ExecutableExpr::nullable(&expr, input_schema),
            Expr::IsNotNull(_) => Ok(false),
            Expr::IsNull(_) => Ok(true),
            Expr::Negative(expr) => ExecutableExpr::nullable(&expr, input_schema),
            Expr::Between { expr, negated, low, high } => ExecutableExpr::nullable(&expr, input_schema),
            Expr::Case { expr, when_then_expr, else_expr } => nullable_todo_err(),
            Expr::Cast { expr, data_type } => ExecutableExpr::nullable(&expr, input_schema),
            Expr::TryCast { expr, data_type } => Ok(true),
            Expr::Sort { expr, asc, nulls_first } => nullable_todo_err(),
            Expr::ScalarFunction { .. }|
            Expr::ScalarUDF { .. } |
            Expr::AggregateFunction { .. } |
            Expr::WindowFunction { .. } |
            Expr::AggregateUDF { .. } => nullable_todo_err(),
            Expr::InList { expr, list, negated } =>ExecutableExpr::nullable(&expr, input_schema),
            Expr::Wildcard => nullable_todo_err(),
        }
    }
    fn data_type(expr: &Expr, input_schema: &arrow::datatypes::Schema)->Result<DataType>{
        fn data_type_todo_err()->Result<DataType>{
            Err(DataFusionError::Internal("PLACEHOLDER".to_string()))
        }
        match expr{
            Expr::Alias(expr,__) => ExecutableExpr::data_type(&expr, input_schema),
            Expr::Column(name) => column::ColumnExpr::data_type(&name, input_schema),
            Expr::ScalarVariable(_) => Ok(DataType::Utf8),
            Expr::Literal(s) => Ok(s.get_datatype()),
            Expr::BinaryExpr { left, op, right } => {
                let lhs_dt = ExecutableExpr::data_type(&left, input_schema)?;
                let rhs_dt = ExecutableExpr::data_type(&right, input_schema)?;
                binary_operator_data_type(&lhs_dt, op, &rhs_dt)
            }
            Expr::Not(_) => Ok(DataType::Boolean),
            Expr::IsNotNull(_) => Ok(DataType::Boolean),
            Expr::IsNull(_) => Ok(DataType::Boolean),
            Expr::Negative(expr) => ExecutableExpr::data_type(&expr, input_schema),
            Expr::Between { expr, negated, low, high } => Ok(DataType::Boolean),
            Expr::Case { .. } => data_type_todo_err(),
            Expr::Cast { expr, data_type } => Ok(data_type.clone()),
            Expr::TryCast { expr, data_type } => Ok(data_type.clone()),
            Expr::Sort { expr, asc, nulls_first } => unreachable!("Should not be able to get here"),
            Expr::ScalarFunction { fun, args } => {
                let arg_types = args.iter().map(|arg| ExecutableExpr::data_type(arg, input_schema)).collect::<Result<Vec<_>>>()?;
                crate::physical_plan::functions::return_type(fun, &arg_types)
            }
            Expr::ScalarUDF { ..}|
            Expr::AggregateFunction { .. }|
            Expr::WindowFunction { .. } |
            Expr::AggregateUDF { .. } => data_type_todo_err(),
            Expr::InList { expr, list, negated } => Ok(DataType::Boolean),
            Expr::Wildcard => data_type_todo_err(),
        }
    }
}

impl fmt::Display for ExecutableExpr{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}
impl fmt::Debug for ExecutableExpr{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.0)
    }
}


use crate::physical_plan::PhysicalExpr;
impl PhysicalExpr for ExecutableExpr{
    fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }

    fn data_type(&self, input_schema: &arrow::datatypes::Schema) -> Result<DataType> {
        ExecutableExpr::data_type(&self.0, input_schema)
    }

    fn nullable(&self, input_schema: &arrow::datatypes::Schema) -> Result<bool> {
        ExecutableExpr::nullable(&self.0, input_schema)
    }

    fn evaluate(&self, batch: &arrow::record_batch::RecordBatch) -> Result<crate::physical_plan::ColumnarValue> {
        ExecutableExpr::evaluate_unchecked(&self.0, batch)
    }
}

impl<'sch> ExprExecPreparation<'sch>{
    fn get_datatype(&self, expr: &Expr)->Result<DataType>{
        for schema in self.all_schema.iter(){
            match expr.get_type(*schema){
                Ok(datatype)=> return Ok(datatype),
                _=>(),
            }
        }
        Err(DataFusionError::Internal(format!("Could not find the datatype for the expression {:?} with the schemas: {:?}", expr, self.all_schema)))
    }


    fn coerce_function_types(&self, signature: &Signature, args: &mut [Expr])->Result<()>{
        if args.is_empty(){
            return Ok(())
        }
        let current_types =    args
            .iter()
            .map(|expr| self.get_datatype(expr))
            .collect::<Result<Vec<_>>>()?;

        let new_types = crate::physical_plan::type_coercion::data_types(&current_types, signature)?;
        for (arg, (old_data_type, new_data_type)) in args.iter_mut().zip(current_types.into_iter().zip(new_types.into_iter())){
            if old_data_type != new_data_type{
                self.coerce_expr(arg, &old_data_type, new_data_type)?;
            }
        }
        Ok(())
    }

    fn coerce_expr(&self, expr: &mut Expr, from_type: &DataType, to_type: DataType)->Result<()>{
        let mut temp_replacement_expr = Expr::Wildcard;
        if !can_cast_types(&from_type, &to_type){
            return Err(DataFusionError::Internal(format!(
                "Unsupported CAST from {:?} to {:?}",
                from_type,to_type
            )))
        }
        std::mem::swap(expr, &mut temp_replacement_expr);
        *expr = Expr::TryCast{
            expr: Box::new(temp_replacement_expr),
            data_type: to_type
        };
        Ok(())
    }
}


impl<'sch> ExprRewriter for ExprExecPreparation<'sch>{
    fn mutate(&mut self, expr: Expr) -> Result<Expr> {
        println!("Recieved expression :{:?} ", expr);
        let expr = match expr{
            Expr::BinaryExpr {mut left, op, mut right } => {
                let left_datatype = self.get_datatype(left.as_ref())?;
                println!("The left datatype was {}", left_datatype);
                let right_datatype = self.get_datatype(right.as_ref())?;
                println!("The right datatype was {}", left_datatype);
                let common_datatype = crate::physical_plan::expressions::common_binary_type(&left_datatype, &op, &right_datatype)?;
                if left_datatype != common_datatype{
                    self.coerce_expr(&mut left, &left_datatype, common_datatype.clone())?;
                }
                if right_datatype != common_datatype{
                    self.coerce_expr(&mut right, &right_datatype, common_datatype)?;
                };
                Expr::BinaryExpr{left, op, right}
            }
            Expr::Not(expr) => {
                let data_type = self.get_datatype(&expr)?;
                if data_type != DataType::Boolean{
                    return Err(DataFusionError::Internal(format!(
                        "NOT '{:?}' can't be evaluated because the expression's type is {:?}, not boolean",
                        expr, data_type,
                    )))
                }
                Expr::Not(expr)
            }
            
            Expr::Negative(expr) => {
                let data_type = self.get_datatype(&expr)?;
                if !coercion::is_signed_numeric(&data_type) {
                    return Err(DataFusionError::Internal(
                        format!(
                            "(- '{:?}') can't be evaluated because the expression's type is {:?}, not signed numeric",
                            expr, data_type,
                        ),
                    ));
                }
                Expr::Negative(expr)
            }
            Expr::Between { expr, negated, low, high } => {
                todo!("Implement Between expression preparation")
            }
            Expr::Case { expr, when_then_expr, else_expr } => {
                todo!()
            }
            Expr::Cast{expr: cast_arg, data_type: to_type}=> {
                let from_type = self.get_datatype(&cast_arg)?;
                //If the datatypes are the same remove the currently visited node
                if from_type == to_type{
                    *cast_arg
                }else if can_cast_types(&from_type, &to_type){
                    Expr::Cast{expr: cast_arg, data_type: to_type}
                } else {
                    return Err(DataFusionError::Internal(format!(
                        "Unsupported CAST from {:?} to {:?}",
                        cast_arg, to_type
                    )));
                }
            }
            Expr::Sort { .. } => {
                return Err(DataFusionError::Internal("Sort expressions are not valid within expresions which are meant to be executed".to_string()));
            }
            Expr::ScalarFunction { fun, mut args } => {
                let signature = crate::physical_plan::functions::signature(&fun);
                self.coerce_function_types(&signature, &mut args)?;
                Expr::ScalarFunction { fun, args }
            }
            Expr::ScalarUDF { fun, mut args } => {
                self.coerce_function_types(&fun.signature, &mut args)?;
                Expr::ScalarUDF { fun, args }
            }
            Expr::AggregateFunction { fun,mut  args, distinct } => {
                let signature = crate::physical_plan::aggregates::signature(&fun);
                self.coerce_function_types(&signature, &mut args)?;
                Expr::AggregateFunction { fun, args, distinct} 
            }
            Expr::WindowFunction { fun, mut args } => {
                let signature = crate::physical_plan::window_functions::signature(&fun);
                self.coerce_function_types(&signature, &mut args)?;
                Expr::WindowFunction { fun, args }
            }
            Expr::AggregateUDF { fun, mut args } => {
                self.coerce_function_types(&fun.signature, &mut args)?;
                Expr::AggregateUDF{fun ,args}
            }
            Expr::TryCast{expr, data_type: to_type}  => {
                let from_type = self.get_datatype(expr.as_ref())?;
                if from_type == to_type{
                    *expr
                }else if arrow::compute::kernels::cast::can_cast_types(&from_type, &to_type){
                    Expr::TryCast{expr, data_type: to_type}
                }else{
                    return Err(DataFusionError::Internal(format!(
                        "Unsupported CAST from {:?} to {:?}",
                        from_type,to_type
                    )))
                }
            }
            Expr::Wildcard => {
                return Err(DataFusionError::Internal("Wildcard expressions are not valid within executable expressions".to_string()))
            }
            //Expressions that do not require any implicityly inserted expressions, currently these are:
            // Alias
            // Column
            // ScalarVariable
            // Literal
            // IsNotNull
            // IsNull
            // InList
            passthrough_exprs => passthrough_exprs,
        };
        println!("output expression: {:?}\n\n", expr);
        Ok(expr)
    } 

  
}

