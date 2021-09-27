// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Expression simplification optimizer.
//! Rewrites expressions using equivalence rules and the egg optimization library   
#![allow(dead_code)]
#![allow(unused_variables)]
use std::convert::TryFrom;
use std::fmt::Display;
use std::str::FromStr;
use std::vec;
use arrow::datatypes::{DataType, IntervalUnit, TimeUnit};
use log::debug;
use crate::error::DataFusionError;
use crate::physical_plan::expressions::{BinaryExpr, CastExpr, common_binary_type};
use crate::physical_plan::functions::{BuiltinScalarFunction, FunctionVolatility};
use crate::physical_plan::udf::ScalarUDF;
use crate::physical_plan::aggregates::AggregateFunction;
use crate::physical_plan::udaf::AggregateUDF;
use std::rc::Rc;

use crate::{
    logical_plan::LogicalPlan, optimizer::optimizer::OptimizerRule, scalar::ScalarValue,
};
use crate::{logical_plan::Operator, optimizer::utils};

use crate::error::Result as DFResult;
use crate::execution::context::ExecutionProps;
use crate::logical_plan::Expr;

use egg::{rewrite as rw, *};

/// Tokomak optimization rule
pub struct Tokomak{}
impl Tokomak {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

#[allow(unused_variables)]
fn exec_prop_rules(udf_registry:  Rc<UDFRegistry>,udaf_registry: Rc<UDAFRegistry>, rules: &mut Vec<Rewrite<TokomakExpr,()>>){
    let a:Var = "?a".parse().unwrap();
    let b: Var = "?b".parse().unwrap();
    let c: Var = "?c".parse().unwrap();
    let d: Var = "?d".parse().unwrap();
    let x: Var ="?x".parse().unwrap();
    let func: Var = "?func".parse().unwrap();
    let args: Var = "?args".parse().unwrap();
    rules.push(rw!("const-call-udf-scalar"; "(call_udf ?func ?args) "=>{ ConstScalarUDFCallApplier{udf_registry, udaf_registry, func, args}}));
}

fn all_rules(udf_registry:  Rc<UDFRegistry>,udaf_registry: Rc<UDAFRegistry>)->Vec<Rewrite<TokomakExpr, ()>>{
    let mut rules = rules();
    exec_prop_rules(udf_registry, udaf_registry, &mut rules);
    rules
}


fn rules() -> Vec<Rewrite<TokomakExpr, ()>> {
    let a:Var = "?a".parse().unwrap();
    let b: Var = "?b".parse().unwrap();
    let c: Var = "?c".parse().unwrap();
    let d: Var = "?d".parse().unwrap();
    let x: Var ="?x".parse().unwrap();
    let func: Var = "?func".parse().unwrap();
    let args: Var = "?args".parse().unwrap();
    return vec![
        rw!("commute-add"; "(+ ?x ?y)" => "(+ ?y ?x)"),
        rw!("commute-mul"; "(* ?x ?y)" => "(* ?y ?x)"),
        rw!("commute-and"; "(and ?x ?y)" => "(and ?y ?x)"),
        rw!("commute-or"; "(or ?x ?y)" => "(or ?y ?x)"),
        rw!("commute-eq"; "(= ?x ?y)" => "(= ?y ?x)"),
        rw!("commute-neq"; "(<> ?x ?y)" => "(<> ?y ?x)"),
        rw!("converse-gt"; "(> ?x ?y)" => "(< ?y ?x)"),
        rw!("converse-gte"; "(>= ?x ?y)" => "(<= ?y ?x)"),
        rw!("converse-lt"; "(< ?x ?y)" => "(> ?y ?x)"),
        rw!("converse-lte"; "(<= ?x ?y)" => "(>= ?y ?x)"),
        rw!("add-0"; "(+ ?x 0)" => "?x"),
        rw!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("minus-0"; "(- ?x 0)" => "?x"),
        rw!("mul-1"; "(* ?x 1)" => "?x"),
        rw!("div-1"; "(/ ?x 1)" => "?x"),
        rw!("dist-and-or"; "(or (and ?a ?b) (and ?a ?c))" => "(and ?a (or ?b ?c))"),
        rw!("dist-or-and"; "(and (or ?a ?b) (or ?a ?c))" => "(or ?a (and ?b ?c))"),
        rw!("not-not"; "(not (not ?x))" => "?x"),
        rw!("or-same"; "(or ?x ?x)" => "?x"),
        rw!("and-same"; "(and ?x ?x)" => "?x"),
        rw!("and-true"; "(and true ?x)"=> "?x"),
        rw!("0-minus"; "(- 0 ?x)"=> "(negative ?x)"),
        rw!("and-false"; "(and false ?x)"=> "false"),
        rw!("or-false"; "(or false ?x)"=> "?x"),
        rw!("or-true"; "(or true ?x)"=> "true"),

        //rw!("inlist-union"; "(")

        
        

        rw!("between-same"; "(between ?e ?a ?a)"=> "(= ?e ?a)"),
        rw!("expand-between"; "(between ?e ?a ?b)" => "(and (>= ?e ?a) (<= ?e ?b))"),// May not be correct with nulls
        rw!("between_inverted-same"; "(between_inverted ?e ?a ?a)" => "(<> ?e ?a)" ),// May not be correct with nulls
        rw!("expand-between_inverted"; "(between_inverted ?e ?a ?b)" => "(and (< ?e ?a) (> ?e ?b))"),
        rw!("between_inverted-not-between"; "(between_inverted ?e ?a ?b)" => "(not (between ?e ?a ?b))"),

        rw!("const-prop-between"; "(and (>= ?e ?a) (<= ?e ?b))"=> "false" if gt_var(a,b)),
        rw!("const-prop-between_inverted"; "(and (< ?e ?a) (> ?e ?b))" => "true" if gt_var(a,b)),

        
        rw!("between-or-union"; "(or (between ?x ?a ?b) (between ?x ?c ?d))" => { BetweenMergeApplier{
            common_comparison: x,
            lhs_lower: a,
            lhs_upper: b,
            rhs_upper: d,
            rhs_lower: c,
        }}),
        //[ 1, 2, null , 3 ,4]
        //[ 1, 2, null , 3 ,4]
        rw!("const-prop-binop-col-eq"; "(= ?a ?a)" => "(is_not_null ?a)"), //May not be true with nullable columns because NULL != NULL
        rw!("const-prop-binop-col-neq"; "(<> ?a ?a)" => "false"), //May not be true with nullable columns because NULL != NULL
        rw!("const-prop-binop-eq"; "(= ?a ?b)"=>{ const_binop(a,b, Operator::Eq)} if can_perform_const_binary_op(a,b, Operator::Eq)),
        rw!("const-prop-binop-neq"; "(<> ?a ?b)"=>{ const_binop(a,b, Operator::NotEq)} if can_perform_const_binary_op(a,b, Operator::NotEq)),

        rw!("const-prop-binop-gt"; "(> ?a ?b)"=>{ const_binop(a,b, Operator::Gt)} if can_perform_const_binary_op(a,b, Operator::Gt)),
        rw!("const-prop-binop-gte"; "(>= ?a ?b)"=>{ const_binop(a,b, Operator::GtEq)} if can_perform_const_binary_op(a,b, Operator::GtEq)),

        rw!("const-prop-binop-lt"; "(< ?a ?b)"=>{ const_binop(a,b, Operator::Lt)} if can_perform_const_binary_op(a,b, Operator::Lt)),
        rw!("const-prop-binop-lte"; "(<= ?a ?b)"=>{ const_binop(a,b, Operator::LtEq)} if can_perform_const_binary_op(a,b, Operator::LtEq)),


        rw!("const-prop-binop-add"; "(+ ?a ?b)"=>{ const_binop(a,b, Operator::Plus) } if can_convert_both_to_scalar_value(a,b) ),
        rw!("const-prop-binop-sub"; "(- ?a ?b)" => { const_binop(a, b, Operator::Minus)} if can_convert_both_to_scalar_value(a,b)),
        rw!("const-prop-binop-mul"; "(* ?a ?b)" => { const_binop(a,b, Operator::Multiply) } if can_convert_both_to_scalar_value(a,b)),
        rw!("const-prop-binop-div"; "(/ ?a ?b)" =>{  const_binop(a,b, Operator::Divide)} if can_convert_both_to_scalar_value(a,b)),

        rw!("const-cast"; "(cast ?a ?b)" =>{ ConstCastApplier{value: a, cast_type: b}} if can_convert_to_scalar_value(a) ),
        rw!("const-call-scalar"; "(call ?func ?args)"=>{ ConstCallApplier{func, args}} if is_immutable_scalar_builtin_function(func) ),
    ];
}



struct ConstScalarUDFCallApplier{
    pub udf_registry:  Rc<UDFRegistry>,
    pub udaf_registry: Rc<UDAFRegistry>,
    pub func: Var,
    pub args: Var,

}

impl ConstScalarUDFCallApplier{
    fn get_scalar_func(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst)->Option<Arc<ScalarUDF>>{
        let name_id = get_scalar_udf_name(self.func, egraph, eclass, subst)?;
        let scalar_udf = self.udf_registry.get(name_id.as_str())?.clone();
        match scalar_udf.volatility{
            FunctionVolatility::Immutable => Some(scalar_udf),
            _ => None,
        } 
    }

    fn apply_one_opt(&self,egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst )->Option<Id>{
        let udf = self.get_scalar_func(egraph, eclass, subst)?;
        let arg_ids = to_literal_list(self.args, egraph, eclass, subst)?;
        let mut func_args = Vec::with_capacity(arg_ids.len());
        for id in arg_ids{
            let scalar = egraph[*id].nodes.iter()
            .flat_map(|expr|->Result<ScalarValue, DataFusionError>{ expr.try_into()} )
            .nth(0)?;
            func_args.push(scalar);
        }
        let coerced_args = coerce_func_args(&func_args, &udf.signature).ok()?;
        let col_val_args = coerced_args.into_iter().map(|a| ColumnarValue::Scalar(a)).collect::<Vec<_>>();
        let ret_val = (&udf.fun)(&col_val_args).ok()?;
        let scalar_ret_val = ret_val.try_into_scalar()?;
        let tokomak_ret_val = scalar_ret_val.try_into().ok()?;
        Some(egraph.add(tokomak_ret_val))
        
    }

}



fn get_scalar_udf_name(var:Var, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst)->Option<Symbol>{
    egraph[subst[var]].nodes.iter().map(|expr| match expr{
        TokomakExpr::ScalarUDF(UDFName(f))=> {
            Some(f.clone())
        },
        _ => None
    }).find(|e| e.is_some()).flatten()
}

fn get_udaf_name(var: Var, egraph: &mut TokomakEGraph, eclass: Id, subst: &Subst)->Option<Symbol>{
    egraph[subst[var]].nodes.iter().map(|expr| match expr{
        TokomakExpr::AggregateUDF(UDFName(f))=> {
            Some(f.clone())
        },
        _ => None
    }).find(|e| e.is_some()).flatten()
}

impl<'a> Applier<TokomakExpr, ()> for ConstScalarUDFCallApplier{
    fn apply_one(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        match self.apply_one_opt(egraph, eclass, subst){
            Some(id)=>vec![id],
            None => Vec::new(),
        }
    }
}



fn can_call_scalar_builtin_const(func: Var, args: Var)->impl Fn(&mut EGraph<TokomakExpr, ()>,  Id,  &Subst)->bool{
    let is_immut = is_immutable_scalar_builtin_function(func);
    let lit_args= args_are_literal(args);
    move |egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst|->bool{
        is_immut(egraph,id, subst) && lit_args(egraph,id, subst)
    }
}


fn is_immutable_scalar_builtin_function(func: Var)->impl Fn(&mut EGraph<TokomakExpr, ()>,  Id,  &Subst)->bool{
   move |egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst|->bool{
       fetch_as_immutable_builtin_scalar_func(func, egraph, id, subst).is_some()
   }
}

fn args_are_literal(args: Var)->impl Fn(&mut EGraph<TokomakExpr, ()>,  Id,  &Subst)->bool{
    move |egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst|->bool{
        to_literal_list(args, egraph, id, subst).is_some()
    }
}

struct ConstCallApplier{
    pub func: Var,
    pub args: Var,
}
/* 
fn is_const_list(var: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    move |egraph: &mut TokomakEGraph, id: Id, subst: &Subst|->bool{

    }
}

*/

fn is_literal(expr: Id, egraph: & TokomakEGraph, id: Id, subst: &Subst)->bool{
    let res = egraph[expr].nodes.iter().flat_map(|expr|->Result<ScalarValue, DataFusionError>{ expr.try_into()} ).nth(0);
    if res.is_none(){
        return false;
    }
    true
}

fn to_literal_list<'a>(var: Var, egraph: &'a TokomakEGraph, id: Id, subst: &Subst)->Option<&'a [Id]>{
    for expr in egraph[subst[var]].nodes.iter(){
        match expr{
            TokomakExpr::List(ref l)=>{
                if l.iter().all(|expr|  is_literal(*expr,egraph, id, subst)){
                    return Some(l.as_slice());
                }
            }
            _=>(),
        } 
    }
    None
}
use core::fmt::Debug;
trait LoggingOkay<T: Debug>{
    fn lok(self)->Option<T>;
}

impl<T: Debug, E: Debug> LoggingOkay<T> for Result<T, E>{
    fn lok(self)->Option<T> {
        match self{
            Ok(v) => {
                println!("Found value: {:#?}", v);
                Some(v)
            },
            Err(e) => {
                println!("Found error: {:#?}", e);
                None
            },
        }
    }
}

use crate::physical_plan::ColumnarValue;
impl ConstCallApplier{
    fn apply_one_opt(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst)->Option<Vec<Id>>{
        let fun = fetch_as_immutable_builtin_scalar_func(self.func, egraph, eclass,subst)?;
        println!("Found the function: {}", fun);
        let args = to_literal_list(self.args,egraph, eclass, subst)?;
        let num_args = args.len();

        let mut func_args = Vec::with_capacity(num_args);
        for id in args{
            let scalar = egraph[*id].nodes.iter()
            .flat_map(|expr|->Result<ScalarValue, DataFusionError>{ expr.try_into()} )
            .nth(0)
            .map(|s| s)?;
            func_args.push(scalar);
        }
        println!("The arguments to the function are {:#?}", func_args);
        let func_args = coerce_func_args(&func_args, &crate::physical_plan::functions::signature(&fun)).ok()?;
        let func_args = func_args.into_iter().map(|s| ColumnarValue::Scalar(s)).collect::<Vec<ColumnarValue>>();
        let arg_types = func_args.iter().map(|arg| arg.data_type()).collect::<Vec<DataType>>();
        let return_ty = crate::physical_plan::functions::return_type(&fun, arg_types.as_slice()).ok()?;
        println!("The return type is: {}", return_ty);
        let fun_impl = crate::physical_plan::functions::create_immutable_impl(&fun).ok()?;
        println!("Construct func impl");
        let result:ColumnarValue = (&fun_impl)(&func_args).lok()?;
        println!("Got columnar result");
        let val: ScalarValue = result.try_into_scalar()?;
        println!("got scalarValue result");

        let expr: TokomakExpr = val.try_into().lok()?;
        println!("got TokomakExpr result result");

        Some(vec![egraph.add(expr)]) 
    }
}




impl Applier<TokomakExpr, ()> for ConstCallApplier{
    fn apply_one(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        println!("Optimizing: {:?}",egraph[eclass]);
        match self.apply_one_opt(egraph, eclass, subst){
            Some(s) =>s,
            None => Vec::new(),
        }
    }
}




fn is_scalar_convertible(var: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst)-> bool{
    conditional_result_to_bool(move |egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst|->Result<(), DataFusionError>{
        let _ = convert_to_scalar_value(var, egraph, id, subst)?;
        Ok(())
    })
}

fn can_const_cast(value: Var, ty: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst)-> bool{
    conditional_result_to_bool(move | egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst|->Result<(), DataFusionError>{
        let _ = convert_to_scalar_value(value, egraph, id, subst)?;
        let _ = convert_to_tokomak_type(ty, egraph, id, subst)?;
        Ok(())
    })
}

struct ConstCastApplier{
    value: Var,
    cast_type: Var,
}

fn convert_to_tokomak_type(var: Var, egraph: &mut EGraph<TokomakExpr, ()>, _id: Id, subst: &Subst)->Result<TokomakDataType, DataFusionError>{
    let expr: Option<&TokomakExpr> = egraph[subst[var]].nodes.iter().find(|e| matches!(e, TokomakExpr::Type(_)));
    let expr = expr.map(|e| match e{
        TokomakExpr::Type(t) => t.clone(),
        _ => panic!("Found TokomakExpr which was not TokomakExpr::Type, found {:?}", e),
    });
    expr.ok_or_else(|| DataFusionError::Internal("Could not convert to TokomakType".to_string()))
}

impl ConstCastApplier{
    fn apply_res(&self, egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst)-> Option<TokomakExpr>{
        println!("Applying const cast");
        let value = convert_to_scalar_value(self.value, egraph, id, subst).lok()?;
        let ty = convert_to_tokomak_type(self.cast_type, egraph, id, subst).lok()?;
        let ty: DataType = ty.into();
        let val = CastExpr::cast_scalar_default_options(value,&ty).lok()?;
        val.try_into().lok()
    }
}
impl Applier<TokomakExpr, ()> for ConstCastApplier{
    fn apply_one(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        let cast_const_res = self.apply_res(egraph, eclass, subst);
        let cast_const_val = match cast_const_res{
            Some(v)=> v,
            None=> return Vec::new()
        };
        vec![egraph.add(cast_const_val)]
    }
}

fn conditional_result_to_bool<T>(c:impl Fn(&mut TokomakEGraph, Id, &Subst) -> Result<T, DataFusionError>)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    move |egraph: &mut TokomakEGraph, id: Id, subst: &Subst| -> bool{
        let res = c(egraph, id, subst);
        match res{
            Err(_)=> {
                //debug!(e);
                return false;
            }
            Ok(_)=> return true,
        }
    }
}


struct BetweenMergeApplier{
    pub common_comparison: Var,
    pub lhs_lower: Var, 
    pub lhs_upper: Var, 
    pub rhs_lower: Var, 
    pub rhs_upper: Var
}

impl BetweenMergeApplier{
    fn try_merge(&self,egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst )->Result<(TokomakExpr, TokomakExpr), DataFusionError>{
        let lhs_low = convert_to_scalar_value(self.lhs_lower, egraph, id, subst)?;
        let lhs_high = convert_to_scalar_value(self.lhs_upper, egraph, id, subst)?;
        let rhs_low = convert_to_scalar_value(self.rhs_lower, egraph, id, subst)?;
        let rhs_high = convert_to_scalar_value(self.rhs_upper, egraph, id, subst)?;

        //Check if one is contained within another
        let rhs_high_in_lhs = gte(rhs_high.clone(), lhs_low.clone())? && lte(rhs_high.clone(), lhs_high.clone())?;
        let rhs_low_in_lhs = gte(rhs_low.clone(), lhs_low.clone())? && lte(rhs_low.clone(), lhs_high.clone())?;
        let is_overlap = rhs_high_in_lhs || rhs_low_in_lhs;
        if is_overlap{
            let new_lower = min(lhs_low, rhs_low)?;
            let new_high = max(lhs_high, rhs_high)?;
            return Ok((new_lower.try_into()?,new_high.try_into()?))
        }
        Err(DataFusionError::Internal(String::new()))
    }
}

impl Applier<TokomakExpr, ()> for BetweenMergeApplier{
    fn apply_one(&self, egraph: &mut EGraph<TokomakExpr, ()>, eclass: Id, subst: &Subst) -> Vec<Id> {
        let (lower, upper) = match self.try_merge(egraph, eclass, subst){
            Ok(new_range)=>new_range,
            Err(_) => return Vec::new(),
        };
        let lower_id = egraph.add(lower);
        let upper_id = egraph.add(upper);
        let common_compare = egraph[subst[self.common_comparison]].id;
        let new_between = TokomakExpr::Between([common_compare, lower_id, upper_id]);
        let new_between_id = egraph.add(new_between);
        vec![new_between_id]
    }
}





fn min(x: ScalarValue, y: ScalarValue)->Result<ScalarValue, DataFusionError>{
    let ret_val = if lt(x.clone(), y.clone())?{
        x
    }else{
        y
    };
    Ok(ret_val)
}

fn max(x: ScalarValue, y: ScalarValue)->Result<ScalarValue, DataFusionError>{
    let ret_val = if gt(x.clone(), y.clone())?{
        x
    }else{
        y
    };
    Ok(ret_val)
}

fn lt(lhs: ScalarValue, rhs:ScalarValue)->Result<bool, DataFusionError>{
    let res = scalar_binop(lhs, rhs, &Operator::Lt)?;
    res.try_into()
}

fn lte(lhs: ScalarValue, rhs:ScalarValue)->Result<bool, DataFusionError>{
    let res = scalar_binop(lhs, rhs, &Operator::LtEq)?;
    res.try_into()
}

fn gt(lhs: ScalarValue, rhs:ScalarValue)->Result<bool, DataFusionError>{
    let res = scalar_binop(lhs, rhs, &Operator::Gt)?;
    res.try_into()
}

fn gte(lhs: ScalarValue, rhs:ScalarValue)->Result<bool, DataFusionError>{
    let res = scalar_binop(lhs, rhs, &Operator::GtEq)?;
    res.try_into()
}

fn cast(val: ScalarValue, cast_type: &DataType)->Result<ScalarValue, DataFusionError>{
    CastExpr::cast_scalar_default_options(val, cast_type)
}


fn scalar_binop(lhs: ScalarValue, rhs:ScalarValue, op: &Operator)->Result<ScalarValue, DataFusionError>{
    let (lhs, rhs) = binop_type_coercion(lhs, rhs, &op)?;
    let res = BinaryExpr::evaluate_values(&op, crate::physical_plan::ColumnarValue::Scalar(lhs),crate::physical_plan::ColumnarValue::Scalar(rhs),1)?;
    match res{
        crate::physical_plan::ColumnarValue::Scalar(s)=> Ok(s),
        crate::physical_plan::ColumnarValue::Array(arr)=>{
            if arr.len() == 1 {
                ScalarValue::try_from_array(&arr, 0)
            }else{
                Err(DataFusionError::Internal(format!("Could not convert an array of length {} to a scalar", arr.len())))
            }
        }
    }
}

fn binop_type_coercion(mut lhs: ScalarValue,mut rhs:ScalarValue, op: &Operator)->Result<(ScalarValue, ScalarValue), DataFusionError>{
    let lhs_dt = lhs.get_datatype();
    let rhs_dt = rhs.get_datatype();
    let common_datatype = common_binary_type(&lhs_dt, &op, &rhs_dt)?;
    if lhs_dt != common_datatype{
        lhs = cast(lhs, &common_datatype)?;
    }
    if rhs_dt != common_datatype{
        rhs = cast(rhs, &common_datatype)?;
    }
    Ok((lhs, rhs))
}


fn gt_var(lhs: Var, rhs: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    move |egraph: &mut TokomakEGraph, id: Id, subst: &Subst| -> bool{
        if !can_convert_both_to_scalar_value(lhs, rhs)(egraph, id, subst){
            return false;
        }
        let lt = const_binop(lhs, rhs, Operator::Gt);
        let t_expr = match lt.evaluate(egraph,id, subst){
            Some(s) => s,
            None => return false,
        };
        match t_expr{
            TokomakExpr::Boolean(v)=> return v,
            _ => return false,
        }

    }
}


fn can_convert_both_to_scalar_value(lhs: Var, rhs: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    move |egraph: &mut TokomakEGraph, _id: Id, subst: &Subst| -> bool{
        let lexpr = egraph[subst[lhs]].nodes.iter().find(|e| e.can_convert_to_scalar_value());
        let rexpr = egraph[subst[rhs]].nodes.iter().find(|e| e.can_convert_to_scalar_value());
        lexpr.is_some() && rexpr.is_some()
    }
}





fn const_binop(lhs: Var, rhs: Var, op: Operator)->ConstBinop<impl  Fn (TokomakExpr, TokomakExpr)->Option<TokomakExpr>, impl Fn(&&TokomakExpr)->bool>{
    ConstBinop(lhs, rhs, |e| e.can_convert_to_scalar_value(), move |l, r|->Option<TokomakExpr>{
        let mut lhs : ScalarValue = l.try_into().ok()?;
        let mut rhs: ScalarValue = r.try_into().ok()?;
        let ldt = lhs.get_datatype();
        let rdt = rhs.get_datatype();
        println!(" pre cast {:?} {:?} {:?}", lhs, op, rhs);
        if ldt != rdt{
            let common_dt = common_binary_type(&ldt, &op, &rdt).ok()?;
            if ldt != common_dt {
                lhs = CastExpr::cast_scalar_default_options(lhs, &common_dt).ok()?;
            }
            if rdt != common_dt{
                rhs = CastExpr::cast_scalar_default_options(rhs, &common_dt).ok()?;
            }
            println!(" after cast {:?} {:?} {:?}", lhs, op, rhs);
        }
        
        
        let res = BinaryExpr::evaluate_values(&op, crate::physical_plan::ColumnarValue::Scalar(lhs), crate::physical_plan::ColumnarValue::Scalar(rhs), 1).unwrap();
        let scalar = match res{
            crate::physical_plan::ColumnarValue::Array(arr) =>{
                
                if arr.len() != 1{
                    println!("Found array as result of scalar literal math");
                    return None;
                }
                ScalarValue::try_from_array(&arr, 0).ok()?
            },
            crate::physical_plan::ColumnarValue::Scalar(s) => s,
        };
        let t_expr: TokomakExpr = scalar.try_into().unwrap();
        Some(t_expr)
    })
}

struct ConstBinop<F: Fn (TokomakExpr, TokomakExpr)->Option<TokomakExpr>, M: Fn(&&TokomakExpr)->bool + 'static>(pub Var, pub Var, pub M, pub F);

impl<F: Fn (TokomakExpr, TokomakExpr)->Option<TokomakExpr>, M: Fn(&&TokomakExpr)->bool> ConstBinop<F, M>{
    fn evaluate(&self, egraph: &mut EGraph<TokomakExpr, ()>, _: Id, subst: &Subst)->Option<TokomakExpr>{
        let lhs = egraph[subst[self.0]].nodes.iter().find(|expr| self.2(expr));
        let lhs = match lhs{
            Some(v)=>v,
            None => return None,
        }.clone();
        let rhs =  egraph[subst[self.1]].nodes.iter().find(|expr| self.2(expr));
        let rhs = match rhs{
            Some(v)=>v,
            None => return None,
        }.clone();
        self.3(lhs, rhs)

    }
}

impl<F: Fn (TokomakExpr, TokomakExpr)->Option<TokomakExpr>, M: Fn(&&TokomakExpr)->bool> Applier<TokomakExpr, ()> for ConstBinop<F, M>{
    fn apply_one(&self, egraph: &mut EGraph<TokomakExpr, ()>, id: Id, subst: &Subst) -> Vec<Id> {
        match self.evaluate(egraph, id, subst){
            Some(e) => vec![egraph.add(e)],
            None => Vec::new(),
        }
    }
}



struct ConstSub(pub Var, pub Var);
impl Applier<TokomakExpr, ()> for ConstSub{
    fn apply_one(&self, egraph: &mut TokomakEGraph, _eclass: Id, subst: &Subst) -> Vec<Id> {
        let lhs =  egraph[subst[self.0]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let lhs = match lhs{
            Some(v)=>v,
            None => return Vec::new(),
        }.clone();
        let rhs =  egraph[subst[self.1]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let rhs = match rhs{
            Some(v)=>v,
            None => return Vec::new(),
        }.clone();
        match (lhs, rhs){
            (TokomakExpr::Int64(l), TokomakExpr::Int64(r)) => vec![egraph.add(TokomakExpr::Int64(l - r))],
            (TokomakExpr::Float64(l), TokomakExpr::Float64(r)) => vec![egraph.add(TokomakExpr::Float64(l - r))],
            _ => Vec::new(),
        }
    }

    

    fn vars(&self) -> Vec<Var> {
        vec![self.0.clone(), self.1.clone()]
    }
}

type TokomakAnalysis=();
type TokomakEGraph = EGraph<TokomakExpr, TokomakAnalysis>;
impl TokomakExpr{
    fn can_convert_to_scalar_value(&self)->bool{
        matches!(self, TokomakExpr::Boolean(_) |
        TokomakExpr::UInt8(_)|
        TokomakExpr::UInt16(_)|
        TokomakExpr::UInt32(_)|
        TokomakExpr::UInt64(_)|
        TokomakExpr::Int8(_)|
        TokomakExpr::Int16(_)|
        TokomakExpr::Int32(_)|
        TokomakExpr::Int64(_)|
        TokomakExpr::Date32(_)|
        TokomakExpr::Date64(_)|
        TokomakExpr::Float32(_)|
        TokomakExpr::Float64(_)|
        TokomakExpr::Utf8(_)|
        TokomakExpr::TimestampSecond(_)|
        TokomakExpr::TimestampMillisecond(_)|
        TokomakExpr::TimestampMicrosecond(_)|
        TokomakExpr::TimestampNanosecond(_)|
        TokomakExpr::IntervalYearMonth(_)|
        TokomakExpr::IntervalDayTime(_))
    }
}

fn can_perform_const_binary_op(lhs: Var, rhs: Var, op: Operator)->impl Fn(&mut TokomakEGraph, Id, &Subst)-> bool{
    move |egraph:  &mut TokomakEGraph, id: Id, subst: &Subst| -> bool{
        let lhs_lit = convert_to_scalar_value(lhs, egraph, id, subst);
        if lhs_lit.is_err(){
            return false;
        }
        let lhs_lit = lhs_lit.unwrap();

        let rhs_lit = convert_to_scalar_value(rhs, egraph, id, subst);
        if rhs_lit.is_err(){
            return false;
        }
        let rhs_lit = rhs_lit.unwrap();
        let res  = binop_type_coercion(lhs_lit, rhs_lit, &op);
        if res.is_err(){
            return false;
        }
        return true;
    }
}

fn convert_to_scalar_value(var: Var, egraph: &mut TokomakEGraph, _id: Id, subst: &Subst)->Result<ScalarValue, DataFusionError>{
    let expr = egraph[subst[var]].nodes.iter().find(|e| e.can_convert_to_scalar_value()).ok_or_else(|| DataFusionError::Internal("Could not find a node that was convertable to scalar value".to_string()))?;
    expr.try_into()
}

fn fetch_as_builtin_scalar_func(var: Var, egraph: &mut TokomakEGraph, id: Id, subst: &Subst)->Option<BuiltinScalarFunction>{
    egraph[subst[var]].nodes.iter().map(|expr| match expr{
        TokomakExpr::ScalarBuiltin(f)=> Some(f.clone()),
        _ => None
    }).find(|e| e.is_some()).flatten()
}

fn fetch_as_immutable_builtin_scalar_func(var: Var, egraph: &mut TokomakEGraph, id: Id, subst: &Subst)-> Option<BuiltinScalarFunction>{
    egraph[subst[var]].nodes.iter().map(|expr| match expr{
        TokomakExpr::ScalarBuiltin(f)=> {
            if f.function_volatility() == FunctionVolatility::Immutable{
                Some(f.clone())
            }else{
                None
            }
        },
        _ => None
    }).find(|e| e.is_some()).flatten()
}
use crate::physical_plan::functions::Signature;
fn coerce_func_args(
    values: &[ScalarValue],
    signature: &Signature,
) -> Result<Vec<ScalarValue>, DataFusionError> {
    if values.is_empty() {
        return Ok(vec![]);
    }

    let current_types = (values)
        .iter()
        .map(|e| e.get_datatype() )
        .collect::<Vec<DataType>>();

    let new_types = crate::physical_plan::functions::data_types(&current_types, signature)?;

    (&values)
        .iter()
        .enumerate()
        .map(|(i, value)| cast(value.clone(), &new_types[i]))
        .collect::<Result<Vec<_>, DataFusionError>>()
}


#[allow(dead_code)]
fn can_convert_to_scalar_value(var: Var)-> impl Fn(&mut TokomakEGraph, Id, &Subst)-> bool{
    move |egraph: &mut TokomakEGraph, _id: Id, subst: &Subst| -> bool{
        let expr = egraph[subst[var]].nodes.iter().find(|e| e.can_convert_to_scalar_value());
        expr.is_some()
    }
}



#[allow(dead_code)]
fn is_const_math_propable(lhs:Var, rhs: Var)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    move  |egraph: &mut TokomakEGraph, _, subst: &Subst| -> bool{
        let lhs_ty = egraph[subst[lhs]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let lhs_ty = match lhs_ty{
            Some(expr)=> expr,
            None => return false,
        };
        let disc = std::mem::discriminant(lhs_ty);
        let rhs_ty = egraph[subst[rhs]].nodes.iter().find(|expr| std::mem::discriminant(*expr) == disc);
        match rhs_ty{
            Some(_)=> (),
            None => return false,
        };
        return true;
    }
}
#[allow(dead_code)]
fn is_const_addable(lhs: &'static str, rhs: &'static str)->impl Fn(&mut TokomakEGraph, Id, &Subst) -> bool{
    let lhs:Var = lhs.parse().unwrap();
    let rhs:Var = rhs.parse().unwrap();
    move  |egraph: &mut TokomakEGraph, _, subst: &Subst| -> bool{
        let lhs_ty = egraph[subst[lhs]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let lhs_ty = match lhs_ty{
            Some(expr)=> expr,
            None => return false,
        };
        let disc = std::mem::discriminant(lhs_ty);
        let rhs_ty = egraph[subst[rhs]].nodes.iter().find(|expr| std::mem::discriminant(*expr) == disc);
        match rhs_ty{
            Some(_)=> (),
            None => return false,
        };
        return true;
    }
}
struct ConstAdd(pub Var, pub Var);
impl Applier<TokomakExpr, TokomakAnalysis> for ConstAdd{
    fn apply_one(&self, egraph: &mut TokomakEGraph, _eclass: Id, subst: &Subst) -> Vec<Id> {
        let lhs =  egraph[subst[self.0]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let lhs = match lhs{
            Some(v)=>v,
            None => return Vec::new(),
        }.clone();
        let rhs =  egraph[subst[self.1]].nodes.iter().find(|expr| matches!(expr, TokomakExpr::Float64(_)|TokomakExpr::Int64(_)));
        let rhs = match rhs{
            Some(v)=>v,
            None => return Vec::new(),
        }.clone();
        match (lhs, rhs){
            (TokomakExpr::Int64(l), TokomakExpr::Int64(r)) => vec![egraph.add(TokomakExpr::Int64(l + r))],
            (TokomakExpr::Float64(l), TokomakExpr::Float64(r)) => vec![egraph.add(TokomakExpr::Float64(l + r))],
            _ => Vec::new(),
        }
    }

    

    fn vars(&self) -> Vec<Var> {
        vec![self.0.clone(), self.1.clone()]
    }

    
}



define_language! {
    #[derive(Copy)]
    enum TokomakDataType {
        "date32" = Date32,
        "date64" = Date64,
        "bool" = Boolean,
        "int8" = Int8,
        "int16" =Int16,
        "int32" =Int32,
        "int64" =Int64,
        "uint8" =UInt8,
        "uint16" =UInt16,
        "uint32" =UInt32,
        "uint64" =UInt64,
        "float16" =Float16,
        "float32" =Float32,
        "float64" =Float64,
        "utf8"=Utf8,
        "largeutf8"=LargeUtf8,
        "time(s)"=TimestampSecond,
        "time(ms)"=TimestampMillisecond,
        "time(us)"=TimestampMicrosecond,
        "time(ns)"=TimestampNanosecond,
        "interval(yearmonth)"=IntervalYearMonth,
        "interval(daytime)"=IntervalDayTime,
    } 
}

impl Display for TokomakDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl Into<DataType> for &TokomakDataType{
    fn into(self)->DataType{
        let v = *self;
        v.into()
    }
}

impl Into<DataType> for TokomakDataType{
    fn into(self)->DataType{
        match self{
            TokomakDataType::Date32 => DataType::Date32,
            TokomakDataType::Date64 => DataType::Date64,
            TokomakDataType::Boolean => DataType::Boolean,
            TokomakDataType::Int8 => DataType::Int8,
            TokomakDataType::Int16 => DataType::Int16,
            TokomakDataType::Int32 => DataType::Int32,
            TokomakDataType::Int64 => DataType::Int64,
            TokomakDataType::UInt8 => DataType::UInt8,
            TokomakDataType::UInt16 => DataType::UInt16,
            TokomakDataType::UInt32 => DataType::UInt32,
            TokomakDataType::UInt64 => DataType::UInt64,
            TokomakDataType::Float16 => DataType::Float16,
            TokomakDataType::Float32 => DataType::Float32,
            TokomakDataType::Float64 => DataType::Float64,
            TokomakDataType::Utf8 => DataType::Utf8,
            TokomakDataType::LargeUtf8 => DataType::LargeUtf8,
            TokomakDataType::TimestampSecond => DataType::Timestamp(TimeUnit::Second, None),
            TokomakDataType::TimestampMillisecond => DataType::Timestamp(TimeUnit::Millisecond, None),
            TokomakDataType::TimestampMicrosecond => DataType::Timestamp(TimeUnit::Microsecond, None),
            TokomakDataType::TimestampNanosecond => DataType::Timestamp(TimeUnit::Nanosecond, None),
            TokomakDataType::IntervalYearMonth => DataType::Interval(IntervalUnit::YearMonth),
            TokomakDataType::IntervalDayTime => DataType::Interval(IntervalUnit::DayTime),
        }
    }
}

impl TryFrom<DataType> for TokomakDataType{
    type Error = DataFusionError;
    fn try_from(val: DataType) -> Result<Self, Self::Error> {
        Ok(match val{
             DataType::Date32 => TokomakDataType::Date32,
             DataType::Date64 => TokomakDataType::Date64,
             DataType::Boolean => TokomakDataType::Boolean,
             DataType::Int8 => TokomakDataType::Int8,
             DataType::Int16 => TokomakDataType::Int16,
             DataType::Int32 => TokomakDataType::Int32,
             DataType::Int64 => TokomakDataType::Int64,
             DataType::UInt8 => TokomakDataType::UInt8,
             DataType::UInt16 => TokomakDataType::UInt16,
             DataType::UInt32 => TokomakDataType::UInt32,
             DataType::UInt64 => TokomakDataType::UInt64,
             DataType::Float16 => TokomakDataType::Float16,
             DataType::Float32 => TokomakDataType::Float32,
             DataType::Float64 => TokomakDataType::Float64,
             DataType::Utf8 => TokomakDataType::Utf8,
             DataType::LargeUtf8 => TokomakDataType::LargeUtf8,
             DataType::Timestamp(TimeUnit::Second, None) => TokomakDataType::TimestampSecond,
             DataType::Timestamp(TimeUnit::Millisecond, None) => TokomakDataType::TimestampMillisecond,
             DataType::Timestamp(TimeUnit::Microsecond, None) => TokomakDataType::TimestampMicrosecond,
             DataType::Timestamp(TimeUnit::Nanosecond, None) => TokomakDataType::TimestampNanosecond,
             DataType::Interval(IntervalUnit::YearMonth) => TokomakDataType::IntervalYearMonth,
             DataType::Interval(IntervalUnit::DayTime) => TokomakDataType::IntervalDayTime,
             _ => return Err(DataFusionError::Internal(format!("The data type {} is invalid as a tokomak datatype", val))),
        })
    }
}

impl FromStr for TokomakDataType {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "date32" => Ok(TokomakDataType::Date32),
            "date64" => Ok(TokomakDataType::Date64),
            "bool" => Ok(TokomakDataType::Boolean),
            "int8" => Ok(TokomakDataType::Int8),
            "int16" => Ok(TokomakDataType::Int16),
            "int32" => Ok(TokomakDataType::Int32),
            "int64" => Ok(TokomakDataType::Int64),
            "uint8" => Ok(TokomakDataType::UInt8),
            "uint16" => Ok(TokomakDataType::UInt16),
            "uint32" => Ok(TokomakDataType::UInt32),
            "uint64" => Ok(TokomakDataType::UInt64),
            "float16" => Ok(TokomakDataType::Float16),
            "float32" => Ok(TokomakDataType::Float32),
            "float64" => Ok(TokomakDataType::Float64),
            "utf8"=> Ok(TokomakDataType::Utf8),
            "largeutf8"=> Ok(TokomakDataType::LargeUtf8),
            "time(s)"=> Ok(TokomakDataType::TimestampSecond),
            "time(ms)"=> Ok(TokomakDataType::TimestampMillisecond),
            "time(us)"=> Ok(TokomakDataType::TimestampMicrosecond),
            "time(ns)"=> Ok(TokomakDataType::TimestampNanosecond),
            "interval(yearmonth)"=> Ok(TokomakDataType::IntervalYearMonth),
            "interval(daytime)"=> Ok(TokomakDataType::IntervalDayTime),
            _ => Err(DataFusionError::Internal("Parsing string as TokomakDataType failed".to_string()))
        }
    }
}
use ordered_float::OrderedFloat;






use std::convert::TryInto;
impl TryInto<ScalarValue> for &TokomakExpr{
    type Error=DataFusionError;
    fn try_into(self) -> Result<ScalarValue, Self::Error> {
        Ok(match self{
            TokomakExpr::Boolean(v) => ScalarValue::Boolean(Some(*v)),
            TokomakExpr::UInt8(v) => ScalarValue::UInt8(Some(*v)),
            TokomakExpr::UInt16(v) => ScalarValue::UInt16(Some(*v)),
            TokomakExpr::UInt32(v) => ScalarValue::UInt32(Some(*v)),
            TokomakExpr::UInt64(v) => ScalarValue::UInt64(Some(*v)),
            TokomakExpr::Int8(v) => ScalarValue::Int8(Some(*v)),
            TokomakExpr::Int16(v) => ScalarValue::Int16(Some(*v)),
            TokomakExpr::Int32(v) => ScalarValue::Int32(Some(*v)),
            TokomakExpr::Int64(v) => ScalarValue::Int64(Some(*v)),
            TokomakExpr::Date32(v) => ScalarValue::Date32(Some(*v)),
            TokomakExpr::Date64(v) => ScalarValue::Date64(Some(*v)),
            TokomakExpr::Float32(v) => ScalarValue::Float32(Some(v.0)),
            TokomakExpr::Float64(v) => ScalarValue::Float64(Some(v.0)),
            TokomakExpr::Utf8(v) => ScalarValue::Utf8(Some(v.clone())),
            TokomakExpr::TimestampSecond(v) => ScalarValue::TimestampSecond(Some(*v)),
            TokomakExpr::TimestampMillisecond(v) => ScalarValue::TimestampMillisecond(Some(*v)),
            TokomakExpr::TimestampMicrosecond(v) => ScalarValue::TimestampMicrosecond(Some(*v)),
            TokomakExpr::TimestampNanosecond(v) => ScalarValue::TimestampNanosecond(Some(*v)),
            TokomakExpr::IntervalYearMonth(v) => ScalarValue::IntervalYearMonth(Some(*v)),
            TokomakExpr::IntervalDayTime(v) => ScalarValue::IntervalDayTime(Some(*v)),
            _ => return Err(DataFusionError::Internal(format!("The given tokomak expression is not valid as a scalar value: {:?}", self)))
        })
    }
}

impl TryInto<ScalarValue> for TokomakExpr{
    type Error = DataFusionError;
    fn try_into(self) -> Result<ScalarValue, Self::Error> {
        Ok(match self{
            TokomakExpr::Utf8(v) => ScalarValue::Utf8(Some(v)),
            _ => (&self).try_into()?
        })
    }
}

impl TryFrom<&ScalarValue> for TokomakExpr{
    type Error = DataFusionError;
    fn try_from(value: &ScalarValue)->Result<TokomakExpr, Self::Error>{
        Ok(match value{
            ScalarValue::Boolean(Some(v)) =>TokomakExpr::Boolean(*v),
            ScalarValue::Float32(Some(v)) =>TokomakExpr::Float32(OrderedFloat::from(*v)),
            ScalarValue::Float64(Some(v)) =>TokomakExpr::Float64(OrderedFloat::from(*v)),
            ScalarValue::Int8(Some(v)) =>TokomakExpr::Int8(*v),
            ScalarValue::Int16(Some(v)) =>TokomakExpr::Int16(*v),
            ScalarValue::Int32(Some(v)) =>TokomakExpr::Int32(*v),
            ScalarValue::Int64(Some(v)) =>TokomakExpr::Int64(*v),
            ScalarValue::UInt8(Some(v)) =>TokomakExpr::UInt8(*v),
            ScalarValue::UInt16(Some(v)) =>TokomakExpr::UInt16(*v),
            ScalarValue::UInt32(Some(v)) =>TokomakExpr::UInt32(*v),
            ScalarValue::UInt64(Some(v)) =>TokomakExpr::UInt64(*v),
            ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) =>TokomakExpr::Utf8(v.to_string()),
            ScalarValue::Date32(Some(v)) =>TokomakExpr::Date32(*v),
            ScalarValue::Date64(Some(v)) =>TokomakExpr::Date64(*v),
            ScalarValue::TimestampSecond(Some(v)) =>TokomakExpr::TimestampSecond(*v),
            ScalarValue::TimestampMillisecond(Some(v)) =>TokomakExpr::TimestampMillisecond(*v),
            ScalarValue::TimestampMicrosecond(Some(v)) =>TokomakExpr::TimestampMicrosecond(*v),
            ScalarValue::TimestampNanosecond(Some(v)) =>TokomakExpr::TimestampNanosecond(*v),
            ScalarValue::IntervalYearMonth(Some(v)) =>TokomakExpr::IntervalYearMonth(*v),
            ScalarValue::IntervalDayTime(Some(v)) =>TokomakExpr::IntervalDayTime(*v),
            _ => return Err(DataFusionError::Internal(format!("The scalar value {:?} is an invalid tokomak expression", value)))
        })
    }
}



impl TryFrom<ScalarValue> for TokomakExpr{
    type Error = DataFusionError;

    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        Ok(match value{
            ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => TokomakExpr::Utf8(v),
            _ => (&value).try_into()? 
        })
    }
}


define_language! {
    /// Supported expressions in ExprSimplifier
    enum TokomakExpr {
        "+" = Plus([Id; 2]),
        "-" = Minus([Id; 2]),
        "*" = Multiply([Id; 2]),
        "/" = Divide([Id; 2]),
        "%" = Modulus([Id; 2]),
        "not" = Not(Id),
        "or" = Or([Id; 2]),
        "and" = And([Id; 2]),
        "=" = Eq([Id; 2]),
        "<>" = NotEq([Id; 2]),
        "<" = Lt([Id; 2]),
        "<=" = LtEq([Id; 2]),
        ">" = Gt([Id; 2]),
        ">=" = GtEq([Id; 2]),
            

        "is_not_null" = IsNotNull(Id),
        "is_null" = IsNull(Id),
        "negative" = Negative(Id),
        "between" = Between([Id; 3]),
        "between_inverted" = BetweenInverted([Id; 3]),
        "like" = Like([Id; 2]),
        "not_like" = NotLike([Id; 2]),
        "in_list" = InList([Id; 2]),
        "not_in_list" = NotInList([Id; 2]),
        "list" = List(Vec<Id>),
        //ScalarValue types 

        Boolean(bool),
        UInt8(u8),
        UInt16(u16),
        UInt32(u32),
        UInt64(u64),
        Int8(i8),
        Int16(i16),
        Int32(i32),
        Int64(i64),
        Date32(i32),
        Date64(i64),
        Float32(OrderedFloat<f32>),
        Float64(OrderedFloat<f64>),
        Type(TokomakDataType),
        ScalarBuiltin(BuiltinScalarFunction),
        AggregateBuiltin(AggregateFunction),
        //THe fist expression for all of the function call types must be the corresponding function type
        //For UDFs this is a string, which is looked up in the ExecutionProps
        //The last expression must be a List and is the arguments for the function.
        "call" = ScalarBuiltinCall([Id; 2]),
        "call_udf"=ScalarUDFCall([Id; 2]),
        //For the aggregate builtin functions the second parameter is a boolean for the distinct modifier
        //Given an arguments list in the variable ?args 
        //Calling MAX(distinct column) would have the form
        // (call_agg true max ?args)
        //While calling max(column) would be
        // (call_agg false max ?args)
        "call_agg" = AggregateBuiltinCall([Id; 3]),
        "call_udaf" = AggregateUDFCall([Id; 2]),
        
        ScalarUDF(UDFName),
        AggregateUDF(UDFName),
        Column(Symbol),
        Utf8(String),
        TimestampSecond(i64),
        TimestampMillisecond(i64),
        TimestampMicrosecond(i64),
        TimestampNanosecond(i64),
        IntervalYearMonth(i32),
        IntervalDayTime(i64),
        
        
        // cast id as expr. Type is encoded as symbol
        "cast" = Cast([Id; 2]),
        "try_cast" = TryCast([Id;2]),
    }
}
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct UDFName(pub Symbol);

impl Display for UDFName{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl FromStr for UDFName{
    type Err=DataFusionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let first_char = s.chars().nth(0).ok_or(DataFusionError::Internal("Zero length udf naem".to_string()))?;
        if first_char == '?' || first_char.is_numeric(){
            return Err(DataFusionError::Internal("Found ? or number as first char".to_string()));
        }  
        Ok(UDFName(Symbol::from_str(s).unwrap()))
    }
}
use std::collections::HashMap;
use std::sync::Arc;

type UDFRegistry = HashMap<String, Arc<ScalarUDF>>;
type UDAFRegistry = HashMap<String, Arc<AggregateUDF>>;


fn add_list(rec_expr: &mut RecExpr<TokomakExpr>,udf_registry: &mut UDFRegistry, udaf_registry: &mut UDAFRegistry, exprs: Vec<Expr>)->Option<Id>{
    let list = exprs.into_iter().map(|expr| to_tokomak_expr(rec_expr, udf_registry, udaf_registry, expr)).collect::<Option<Vec<Id>>>()?;
    Some(rec_expr.add(TokomakExpr::List(list)))
}

fn to_tokomak_expr(rec_expr: &mut RecExpr<TokomakExpr>, udf_registry: &mut UDFRegistry, udaf_registry: &mut HashMap<String, Arc<AggregateUDF>>, expr: Expr) -> Option<Id> {
    match expr {
        Expr::BinaryExpr { left, op, right } => {
            let left = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *left)?;
            let right = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *right)?;
            let binary_expr = match op {
                Operator::Eq => TokomakExpr::Eq,
                Operator::NotEq => TokomakExpr::NotEq,
                Operator::Lt => TokomakExpr::Lt,
                Operator::LtEq => TokomakExpr::LtEq,
                Operator::Gt => TokomakExpr::Gt,
                Operator::GtEq => TokomakExpr::GtEq,
                Operator::Plus => TokomakExpr::Plus,
                Operator::Minus => TokomakExpr::Minus,
                Operator::Multiply => TokomakExpr::Multiply,
                Operator::Divide => TokomakExpr::Divide,
                Operator::Modulus => TokomakExpr::Modulus,
                Operator::And => TokomakExpr::And,
                Operator::Or => TokomakExpr::Or,
                Operator::Like => TokomakExpr::Like,
                Operator::NotLike => TokomakExpr::NotLike,
            };
            Some(rec_expr.add(binary_expr([left, right])))
        }
        Expr::Column(c) => Some(rec_expr.add(TokomakExpr::Column(Symbol::from(c)))),
        Expr::Literal(ScalarValue::Int64(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Int64(x)))
        }
        Expr::Literal(ScalarValue::Utf8(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Utf8(x)))
        }
        Expr::Literal(ScalarValue::LargeUtf8(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Utf8(x)))
        }
        Expr::Literal(ScalarValue::Boolean(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Boolean(x)))
        }
        Expr::Literal(ScalarValue::Date32(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Date32(x)))
        }
        Expr::Literal(ScalarValue::Date64(Some(x))) => {
            Some(rec_expr.add(TokomakExpr::Date64(x)))
        }
        Expr::Not(expr) => {
            let e = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *expr)?;
            Some(rec_expr.add(TokomakExpr::Not(e)))
        }
        Expr::IsNull(expr) => {
            let e = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *expr)?;
            Some(rec_expr.add(TokomakExpr::IsNull(e)))
        }
        Expr::IsNotNull(expr) => {
            let e = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *expr)?;
            Some(rec_expr.add(TokomakExpr::IsNotNull(e)))
        }
        Expr::Negative(expr) => {
            let e = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *expr)?;
            Some(rec_expr.add(TokomakExpr::Negative(e)))
        }
        Expr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let e = to_tokomak_expr(rec_expr,  udf_registry, udaf_registry, *expr)?;
            let low = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *low)?;
            let high = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *high)?;
            if negated {
                Some(rec_expr.add(TokomakExpr::BetweenInverted([e, low, high])))
            } else {
                Some(rec_expr.add(TokomakExpr::Between([e, low, high])))
            }
        }

        Expr::Cast {
            expr,
            data_type
        } => {
            let ty = data_type.try_into().ok()?;
            let e = to_tokomak_expr(rec_expr, udf_registry, udaf_registry, *expr)?;
            let t = rec_expr.add(TokomakExpr::Type(ty));

            Some(rec_expr.add(TokomakExpr::Cast([e, t])))
        }
        Expr::TryCast { expr, data_type } => {
            let ty: TokomakDataType = data_type.try_into().ok()?;
            let e = to_tokomak_expr(rec_expr,  udf_registry, udaf_registry, *expr)?;
            let t = rec_expr.add(TokomakExpr::Type(ty));
            Some(rec_expr.add(TokomakExpr::TryCast([e,t])))
        },
        Expr::ScalarFunction { fun, args } => {
            let fun_id = rec_expr.add(TokomakExpr::ScalarBuiltin(fun));
            let args_id = add_list(rec_expr,  udf_registry, udaf_registry,args)?;
            Some(rec_expr.add(TokomakExpr::ScalarBuiltinCall([fun_id, args_id])))
        },
        Expr::Alias(expr, _)=>to_tokomak_expr(rec_expr,  udf_registry, udaf_registry, *expr),
        Expr::InList { expr, list, negated } => {
            let val_expr = to_tokomak_expr(rec_expr,  udf_registry, udaf_registry, *expr)?;
            let list_id = add_list(rec_expr,  udf_registry, udaf_registry,list)?;
            Some(match negated{
                false => rec_expr.add(TokomakExpr::InList([val_expr, list_id])),
                true => rec_expr.add(TokomakExpr::NotInList([val_expr, list_id])),
            })
        },
        Expr::AggregateFunction{fun, args, distinct }=>{
            let agg_expr = TokomakExpr::AggregateBuiltin(fun);
            let fun_id = rec_expr.add(agg_expr);
            let args_id = add_list(rec_expr, udf_registry, udaf_registry,args)?;
            let distinct_id = rec_expr.add(TokomakExpr::Boolean(distinct));
            Some(rec_expr.add(TokomakExpr::AggregateBuiltinCall([fun_id, distinct_id, args_id])))
        },
        //Expr::Case { expr, when_then_expr, else_expr } => todo!(),
        //Expr::Sort { expr, asc, nulls_first } => todo!(),
        Expr::ScalarUDF { fun, args } => {
            let args_id = add_list(rec_expr, udf_registry, udaf_registry,args)?;
            let fun_name: Symbol = fun.name.clone().into();
            let fun_name_id = rec_expr.add(TokomakExpr::ScalarUDF(UDFName(fun_name)));
            Some(rec_expr.add(TokomakExpr::ScalarUDFCall([fun_name_id, args_id])))
        },
        //Expr::WindowFunction { fun, args } => todo!(),
        Expr::AggregateUDF { fun, args } => {
            let args_id = add_list(rec_expr, udf_registry, udaf_registry,args)?;
            let fun_name: Symbol = fun.name.clone().into();//Symbols are leaked at this point in time. Maybe different solution is required.
            let fun_name_id = rec_expr.add(TokomakExpr::AggregateUDF(UDFName(fun_name)));
            Some(rec_expr.add(TokomakExpr::AggregateUDFCall([fun_name_id, args_id])))
        },
        //Expr::Wildcard => todo!(),
        //Expr::ScalarVariable(_) => todo!(),
        
        // not yet supported
        e => {
            debug!("Expression not yet supported in tokomak optimizer {:?}", e);
            None
        },
    }
}

fn to_exprs(rec_expr: &RecExpr<TokomakExpr>, udf_registry:&mut UDFRegistry, udaf_registry: &mut UDAFRegistry, id: Id) -> Expr {
    let refs = rec_expr.as_ref();
    let index: usize = id.into();
    match refs[index] {
        TokomakExpr::Plus(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry, ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Plus,
                right: Box::new(r),
            }
        }
        TokomakExpr::Minus(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Minus,
                right: Box::new(r),
            }
        }
        TokomakExpr::Divide(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Divide,
                right: Box::new(r),
            }
        }
        TokomakExpr::Modulus(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Modulus,
                right: Box::new(r),
            }
        }
        TokomakExpr::Not(id) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,id);
            Expr::Not(Box::new(l))
        }
        TokomakExpr::IsNotNull(id) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,id);
            Expr::IsNotNull(Box::new(l))
        }
        TokomakExpr::IsNull(id) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,id);
            Expr::IsNull(Box::new(l))
        }
        TokomakExpr::Negative(id) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,id);
            Expr::Negative(Box::new(l))
        }

        TokomakExpr::Between([expr, low, high]) => {
            let left = to_exprs(&rec_expr, udf_registry, udaf_registry,expr);
            let low_expr = to_exprs(&rec_expr, udf_registry, udaf_registry,low);
            let high_expr = to_exprs(&rec_expr, udf_registry, udaf_registry,high);

            Expr::Between {
                expr: Box::new(left),
                negated: false,
                low: Box::new(low_expr),
                high: Box::new(high_expr),
            }
        }
        TokomakExpr::BetweenInverted([expr, low, high]) => {
            let left = to_exprs(&rec_expr, udf_registry, udaf_registry,expr);
            let low_expr = to_exprs(&rec_expr, udf_registry, udaf_registry,low);
            let high_expr = to_exprs(&rec_expr, udf_registry, udaf_registry,high);

            Expr::Between {
                expr: Box::new(left),
                negated: false,
                low: Box::new(low_expr),
                high: Box::new(high_expr),
            }
        }
        TokomakExpr::Multiply(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Multiply,
                right: Box::new(r),
            }
        }
        TokomakExpr::Or(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Or,
                right: Box::new(r),
            }
        }
        TokomakExpr::And(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::And,
                right: Box::new(r),
            }
        }
        TokomakExpr::Eq(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Eq,
                right: Box::new(r),
            }
        }
        TokomakExpr::NotEq(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::NotEq,
                right: Box::new(r),
            }
        }
        TokomakExpr::Lt(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Lt,
                right: Box::new(r),
            }
        }
        TokomakExpr::LtEq(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::LtEq,
                right: Box::new(r),
            }
        }
        TokomakExpr::Gt(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Gt,
                right: Box::new(r),
            }
        }
        TokomakExpr::GtEq(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::GtEq,
                right: Box::new(r),
            }
        }
        TokomakExpr::Like(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::Like,
                right: Box::new(r),
            }
        }
        TokomakExpr::NotLike(ids) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[0]);
            let r = to_exprs(&rec_expr, udf_registry, udaf_registry,ids[1]);

            Expr::BinaryExpr {
                left: Box::new(l),
                op: Operator::NotLike,
                right: Box::new(r),
            }
        }

        TokomakExpr::Column(col) => Expr::Column(col.to_string()),
        TokomakExpr::Cast([e, ty]) => {
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,e);
            let index:usize = ty.into();
            let dt = match &refs[index] {
                TokomakExpr::Type(s) => s,
                _ => panic!("Second argument of cast should be type")
            };
            let dt: DataType = dt.into();

            Expr::Cast { expr: Box::new(l), data_type: dt}
        }
        TokomakExpr::Type(_) => {
            panic!("Type should only be part of expression")
        }
        TokomakExpr::InList([val, list])=>{
            let l = to_exprs(&rec_expr, udf_registry, udaf_registry,val);
            let list_idx: usize = list.into();
            let list = &refs[list_idx];
            let list = match list{
                TokomakExpr::List(ref l) => l.iter().map(|i| to_exprs(&rec_expr,udf_registry, udaf_registry,*i )).collect::<Vec<Expr>>(),
                _ => panic!("The second id in TokomakExpr::InList was not a list"),
            };
            Expr::InList{
              list,
              expr: Box::new(l), 
              negated: false, 
            }
        }
        TokomakExpr::NotInList([val, list])=>{
            let l = to_exprs(&rec_expr,  udf_registry, udaf_registry,val);
            let list_idx: usize = list.into();
            let list = &refs[list_idx];
            let list = match list{
                TokomakExpr::List(ref l) => l.iter().map(|i| to_exprs(&rec_expr,udf_registry, udaf_registry,*i )).collect::<Vec<Expr>>(),
                _ => panic!("The second id in TokomakExpr::InList was not a list"),
            };
            Expr::InList{
              list,
              expr: Box::new(l), 
              negated: true, 
            }
        }
        TokomakExpr::List(_) => panic!("TokomakExpr::List should only ever be a child expr and should be handled by the parent expression"),
        TokomakExpr::Boolean(v) => Expr::Literal( ScalarValue::Boolean(Some(v))),
        TokomakExpr::UInt8(v) => Expr::Literal( ScalarValue::UInt8(Some(v))),
        TokomakExpr::UInt16(v) => Expr::Literal( ScalarValue::UInt16(Some(v))),
        TokomakExpr::UInt32(v) => Expr::Literal( ScalarValue::UInt32(Some(v))),
        TokomakExpr::UInt64(v) => Expr::Literal( ScalarValue::UInt64(Some(v))),
        TokomakExpr::Int8(v) => Expr::Literal( ScalarValue::Int8(Some(v))),
        TokomakExpr::Int16(v) => Expr::Literal( ScalarValue::Int16(Some(v))),
        TokomakExpr::Int32(v) => Expr::Literal( ScalarValue::Int32(Some(v))),
        TokomakExpr::Int64(v) => Expr::Literal( ScalarValue::Int64(Some(v))),
        TokomakExpr::Date32(v) => Expr::Literal( ScalarValue::Date32(Some(v))),
        TokomakExpr::Date64(v) => Expr::Literal( ScalarValue::Date64(Some(v))),
        TokomakExpr::Float32(v) => Expr::Literal( ScalarValue::Float32(Some(v.0))),
        TokomakExpr::Float64(v) => Expr::Literal( ScalarValue::Float64(Some(v.0))),
        TokomakExpr::Utf8(ref v) => Expr::Literal( ScalarValue::Utf8(Some(v.clone()))),
        TokomakExpr::TimestampSecond(v) => Expr::Literal( ScalarValue::TimestampSecond(Some(v))),
        TokomakExpr::TimestampMillisecond(v) => Expr::Literal( ScalarValue::TimestampMillisecond(Some(v))),
        TokomakExpr::TimestampMicrosecond(v) => Expr::Literal( ScalarValue::TimestampMicrosecond(Some(v))),
        TokomakExpr::TimestampNanosecond(v) => Expr::Literal( ScalarValue::TimestampNanosecond(Some(v))),
        TokomakExpr::IntervalYearMonth(v) => Expr::Literal( ScalarValue::IntervalYearMonth(Some(v))),
        TokomakExpr::IntervalDayTime(v) => Expr::Literal( ScalarValue::IntervalDayTime(Some(v))),
        TokomakExpr::ScalarBuiltin(_) => panic!("ScalarBuiltin should only be part of an expression"),
        TokomakExpr::ScalarBuiltinCall([fun, args]) => {
            let fun_idx: usize = fun.into();
            let fun = match &refs[fun_idx]{
                TokomakExpr::ScalarBuiltin(f)=>f,
                f => panic!("Expected a function in the first position, found {:?}", f),
            };
            let args_idx: usize = args.into();
            let arg_ids = match &refs[args_idx]{
                TokomakExpr::List(args)=> args,
                e => panic!("Expected a list a function arguments, found: {:?}", e),
            };
            let args = arg_ids.iter().map(|expr| to_exprs(rec_expr, udf_registry, udaf_registry,*expr)).collect::<Vec<_>>();
            Expr::ScalarFunction{
                fun:fun.clone(),
                args,
            }
        },
        TokomakExpr::TryCast(_) => todo!(),
        TokomakExpr::AggregateBuiltin(_) => todo!(),
        TokomakExpr::ScalarUDFCall([name_id, args_id]) => {
            let args_idx: usize = args_id.into();
            let args = &refs[args_idx];
            let args = match args{
                TokomakExpr::List(ref l) => l.iter().map(|i| to_exprs(&rec_expr,udf_registry, udaf_registry,*i )).collect::<Vec<Expr>>(),
                _ => panic!("The second id in TokomakExpr::InList was not a list"),
            };
            let name_idx: usize = name_id.into();
            let name = &refs[name_idx];
            let name = match name{
                TokomakExpr::ScalarUDF(sym)=> sym.0.to_string(),
                e => panic!("Found a non ScalarUDF node in the first position of ScalarUdf: {:#?}",e),
            };
            //TODO switch to lookup
            let fun = udf_registry.get(&name).expect(format!("could not find the scalar udf \"{}\"", name).as_str()).clone();
            Expr::ScalarUDF{
                fun,
                args    
            }
        },
        TokomakExpr::AggregateBuiltinCall(_) => todo!(),
        TokomakExpr::AggregateUDFCall(_) => todo!(),
        TokomakExpr::ScalarUDF(_) => todo!(),
        TokomakExpr::AggregateUDF(_) => todo!(),
        
        
    }
}

impl OptimizerRule for Tokomak {
    fn optimize(
        &self,
        plan: &LogicalPlan,
        props: &ExecutionProps,
    ) -> DFResult<LogicalPlan> {
        let mut udf_registry = HashMap::new();
        let mut udaf_registry = HashMap::new();
        let inputs = plan.inputs();
        let new_inputs: Vec<LogicalPlan> = inputs
            .iter()
            .map(|plan| self.optimize(plan, props))
            .collect::<DFResult<Vec<_>>>()?;
        // optimize all expressions individual (for now)
        let mut exprs = vec![];
        for expr in plan.expressions().iter() {
            let rec_expr = &mut RecExpr::default();
            let tok_expr = to_tokomak_expr(rec_expr, &mut udf_registry, &mut udaf_registry, expr.clone());
            match tok_expr {
                None => exprs.push(expr.clone()),
                Some(_expr) => {
                    let runner = Runner::<TokomakExpr, (), ()>::default()
                        .with_expr(rec_expr)
                        .run(&rules());

                    let mut extractor = Extractor::new(&runner.egraph, AstSize);
                    let (_, best_expr) = extractor.find_best(runner.roots[0]);
                    let start = best_expr.as_ref().len() - 1;
                    exprs.push(to_exprs(&best_expr, &mut udf_registry, &mut udaf_registry,start.into()).clone());
                }
            }
        }

        utils::from_plan(plan, &exprs, &new_inputs)
    }

    fn name(&self) -> &str {
        "tokomak"
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use egg::Runner;

    use crate::arrow::array::ArrayRef;
    use crate::arrow::array::Float64Array;
    use std::sync::Arc;
    fn create_pow_udf(name: &str, v: FunctionVolatility)->ScalarUDF{
        let pow = |args: &[ArrayRef]| {
            // in DataFusion, all `args` and output are dynamically-typed arrays, which means that we need to:
            // 1. cast the values to the type we want
            // 2. perform the computation for every element in the array (using a loop or SIMD) and construct the result
    
            // this is guaranteed by DataFusion based on the function's signature.
            assert_eq!(args.len(), 2);
    
            // 1. cast both arguments to f64. These casts MUST be aligned with the signature or this function panics!
            let base = &args[0]
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("cast failed");
            let exponent = &args[1]
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("cast failed");
    
            // this is guaranteed by DataFusion. We place it just to make it obvious.
            assert_eq!(exponent.len(), base.len());
    
            // 2. perform the computation
            let array = base
                .iter()
                .zip(exponent.iter())
                .map(|(base, exponent)| {
                    match (base, exponent) {
                        // in arrow, any value can be null.
                        // Here we decide to make our UDF to return null when either base or exponent is null.
                        (Some(base), Some(exponent)) => Some(base.powf(exponent)),
                        _ => None,
                    }
                })
                .collect::<Float64Array>();
    
            // `Ok` because no error occurred during the calculation (we should add one if exponent was [0, 1[ and the base < 0 because that panics!)
            // `Arc` because arrays are immutable, thread-safe, trait objects.
            Ok(Arc::new(array) as ArrayRef)
        };
        // the function above expects an `ArrayRef`, but DataFusion may pass a scalar to a UDF.
        // thus, we use `make_scalar_function` to decorare the closure so that it can handle both Arrays and Scalar values.
        let pow = crate::physical_plan::functions::make_scalar_function(pow);
    
        // Next:
        // * give it a name so that it shows nicely when the plan is printed
        // * declare what input it expects
        // * declare its return type
        let pow = crate::logical_plan::create_udf(
            name,
            // expects two f64
            vec![DataType::Float64, DataType::Float64],
            // returns f64
            Arc::new(DataType::Float64),
            pow,
            Some(v)
        );
        pow
    
    }

    #[test]
    fn test_add_0() {
        let expr = "(+ 0 (x))".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
            .with_expr(&expr)
            .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        assert_eq!(format!("{}", best_expr), "x")
    }

    #[test]
    fn test_dist_and_or() {
        let expr = "(or (or (and (= 1 2) foo) (and (= 1 2) bar)) (and (= 1 2) boo))"
            .parse()
            .unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
            .with_expr(&expr)
            .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);

        assert_eq!(
            format!("{}", best_expr),
            "(and (= 1 2) (or boo (or foo bar)))"
        )
    }

    #[test]
    fn test_const_prop_add_int(){
        let expr = "(+ 1 2)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);

        assert_eq!(
            format!("{}", best_expr),
            "3"
        ) 
    }

    #[test]
    fn test_const_prop_add_float(){
        let expr = "(+ 1.5 2.0)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        

        assert_eq!(
            format!("{}", best_expr),
            "3.5"
        ) 
    }

    #[test]
    fn test_const_prop_sub_int(){
        let expr = "(- 5 2)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);

        assert_eq!(
            format!("{}", best_expr),
            "3"
        ) 
    }

    #[test]
    fn test_const_prop_sub_float(){
        let expr = "(- 5.0 1.5)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        

        assert_eq!(
            format!("{}", best_expr),
            "3.5"
        ) 
    }

    #[test]
    fn test_between_same(){
        let expr = "(between ?x ?same ?same)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        

        assert_eq!(
            format!("{}", best_expr),
            "(= ?x ?same)"
        ) 
    }

    #[test]
    fn test_between_inverted_same(){
        let expr = "(between_inverted ?x ?same ?same)".parse().unwrap();
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());

        let mut extractor = Extractor::new(&runner.egraph, AstSize);

        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        

        assert_eq!(
            format!("{}", best_expr),
            "(<> ?x ?same)"
        ) 
    }

    fn rewrite_expr(expr: &str)->Result<String, Box<dyn std::error::Error>>{
        let expr: RecExpr<TokomakExpr> = expr.parse()?;
        println!("unoptomized expr {:?}", expr);
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules());
        let mut extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        println!("optimized expr {:?}", best_expr);
        Ok(format!("{}", best_expr))
    }

    fn rewrite_expr_with_registries(udf_reg:Rc<UDFRegistry>, udaf_reg: Rc<UDAFRegistry>, expr: &str)->Result<String, Box<dyn std::error::Error>>{
        let expr: RecExpr<TokomakExpr> = expr.parse()?;
        println!("unoptomized expr {:?}", expr);
        let rules = all_rules(udf_reg, udaf_reg);
        let runner = Runner::<TokomakExpr, (), ()>::default()
        .with_expr(&expr)
        .run(&rules);
        let mut extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best_expr) = extractor.find_best(runner.roots[0]);
        println!("optimized expr {:?}", best_expr);
        Ok(format!("{}", best_expr))
    }

    #[test]
    fn test_const_prop()->Result<(), Box<dyn std::error::Error>>{
        let pow = create_pow_udf("pow",FunctionVolatility::Immutable);
        let pow_vol = create_pow_udf("pow_vol", FunctionVolatility::Volatile);
         
        let udf_reg = vec![pow, pow_vol].into_iter().map(|f| (f.name.clone(), Arc::new(f))).collect::<HashMap<String, Arc<ScalarUDF>>>();
        let udaf_reg = vec![].into_iter().map(|f: AggregateUDF| (f.name.clone(), Arc::new(f))).collect::<UDAFRegistry>();
        let udf_reg = Rc::new(udf_reg);
        let udaf_reg = Rc::new(udaf_reg);
        let expr_expected: Vec<(&'static str, &'static str)> =  vec![ 
            ("(between ?x 10 4)", "false"),
            ("(between_inverted ?x 10 4)", "true"),
            ("(or (between ?x 0 10) (between ?x 9 20))", "(between ?x 0 20)"),
            ("(cast (cast 2  utf8) float32)", "2"),
            ("(* 4 (call abs (list -1)))", "4"),
            ("(call_udf pow (list (+ 2.5 1.5) (- 1 1.5)))","0.5"), //Test udf inlining
            ("(call_udf pow_vol (list (+ 2.5 1.5) (- 1 1.5)))", "(call_udf pow_vol (list 4 -0.5))")
        ];

        for (expr, expected) in expr_expected{
            let rewritten_expr = rewrite_expr_with_registries( udf_reg.clone(), udaf_reg.clone(), expr)?;
            assert_eq!(rewritten_expr, expected); 
        } 
        Ok(())

    }

    


}
 