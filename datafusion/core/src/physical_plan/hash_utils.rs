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

//! Functionality used both on logical and physical plans

use crate::error::{DataFusionError, Result};
use ahash::{CallHasher, RandomState};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, DecimalArray,
    DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, LargeStringArray, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array, PrimitiveArray, GenericStringArray, StringOffsetSizeTrait,
};
use arrow::datatypes::{
    ArrowDictionaryKeyType, ArrowNativeType, DataType, Int16Type, Int32Type, Int64Type,
    Int8Type, TimeUnit, UInt16Type, UInt32Type, UInt64Type, UInt8Type, ArrowPrimitiveType, Date64Type, Date32Type, Float32Type, Float64Type, TimestampSecondType, TimestampMillisecondType, TimestampMicrosecondType, TimestampNanosecondType,
};
use std::sync::Arc;

// Combines two hashes into one hash
#[inline]
fn combine_hashes(l: u64, r: u64) -> u64 {
    let hash = (17 * 37u64).wrapping_add(l);
    hash.wrapping_mul(37).wrapping_add(r)
}


fn hash_decimal128<'a>(
    array: &ArrayRef,
    random_state: &RandomState,
    hashes_buffer: &'a mut [u64],
    mul_col: bool,
) {
    let array = array.as_any().downcast_ref::<DecimalArray>().unwrap();
    if array.null_count() == 0 {
        if mul_col {
            for (i, hash) in hashes_buffer.iter_mut().enumerate() {
                *hash =
                    combine_hashes(i128::get_hash(&array.value(i), random_state), *hash);
            }
        } else {
            for (i, hash) in hashes_buffer.iter_mut().enumerate() {
                *hash = i128::get_hash(&array.value(i), random_state);
            }
        }
    } else if mul_col {
        for (i, hash) in hashes_buffer.iter_mut().enumerate() {
            if !array.is_null(i) {
                *hash =
                    combine_hashes(i128::get_hash(&array.value(i), random_state), *hash);
            }
        }
    } else {
        for (i, hash) in hashes_buffer.iter_mut().enumerate() {
            if !array.is_null(i) {
                *hash = i128::get_hash(&array.value(i), random_state);
            }
        }
    }
}

macro_rules! hash_array {
    ($array_type:ident, $column: ident, $ty: ident, $hashes: ident, $random_state: ident, $multi_col: ident) => {
        let array = $column.as_any().downcast_ref::<$array_type>().unwrap();
        if array.null_count() == 0 {
            if $multi_col {
                for (i, hash) in $hashes.iter_mut().enumerate() {
                    *hash = combine_hashes(
                        $ty::get_hash(&array.value(i), $random_state),
                        *hash,
                    );
                }
            } else {
                for (i, hash) in $hashes.iter_mut().enumerate() {
                    *hash = $ty::get_hash(&array.value(i), $random_state);
                }
            }
        } else {
            if $multi_col {
                for (i, hash) in $hashes.iter_mut().enumerate() {
                    if !array.is_null(i) {
                        *hash = combine_hashes(
                            $ty::get_hash(&array.value(i), $random_state),
                            *hash,
                        );
                    }
                }
            } else {
                for (i, hash) in $hashes.iter_mut().enumerate() {
                    if !array.is_null(i) {
                        *hash = $ty::get_hash(&array.value(i), $random_state);
                    }
                }
            }
        }
    };
}

pub(crate) fn hash_decimal128_new<'a, const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: u32, num_to_process: u32){
    assert!(num_to_process <= N);
    let array = column.as_any().downcast_ref::<DecimalArray>().unwrap();
    assert!(start_idx + num_to_process <= array.len());
    if array.null_count() == 0 {
        let mut arr_idx = start_idx;
        for offset in 0..num_to_process{
            if arr_idx >= array.len(){
                //Since the decimal array doesn't have value_unchecked use hint instead
                unsafe{std::hint::unreachable_unchecked()};
            }
            let value = array.value(arr_idx);
            let new_hash = i128::get_hash(&value, random_state);
            *unsafe{hashes.get_unchecked_mut(offset)} = if FIRST_COL{
                new_hash
            }else{
                combine_hashes(new_hash, *unsafe{hashes.get_unchecked(offset)})
            };
            arr_idx +=1;
        }
    } else {
        let mut arr_idx = start_idx;
        for offset in 0..num_to_process{
            if arr_idx >= array.len(){
                //Since the decimal array doesn't have value_unchecked use hint instead
                unsafe{std::hint::unreachable_unchecked()};
            }
            //If this is the first column set the index to 0 to avoid reusing hash values already in the buffer 
            if array.is_null(arr_idx){
                if FIRST_COL{
                    *unsafe{hashes.get_unchecked_mut(offset)} =  0;
                }
                continue;
            }
            let value = array.value(arr_idx);
            let new_hash = i128::get_hash(&value, random_state);
            *unsafe{hashes.get_unchecked_mut(offset)} = if FIRST_COL{
                new_hash
            }else{
                combine_hashes(new_hash, *unsafe{hashes.get_unchecked(offset)})
            };
            arr_idx +=1;
        }
    }
}


pub (crate) fn hash_array<ARR: 'static, IsNull: Fn(&ARR, usize)->bool + 'static + Copy, GetHash: Fn(&ARR, &RandomState, usize)->u64, const N: usize, const FIRST_COL: bool>(column: &dyn Array,hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize, is_null: IsNull, get_hash: GetHash){
    let null_count = column.null_count();
    let arr = column.as_any().downcast_ref::<ARR>().unwrap();
    assert!(num_to_process <= N);
    assert!(start_idx + num_to_process <= column.len());
    if null_count == 0{
        for offset in 0..num_to_process{
            *unsafe{hashes.get_unchecked_mut(offset)} = {
                let new_hash = get_hash(arr, random_state, start_idx + offset);
                if FIRST_COL{
                    new_hash
                }else{
                    combine_hashes(new_hash, *unsafe{hashes.get_unchecked(offset)})
                }
            }
        }
    }else{
        for offset in 0..num_to_process{
            let idx = start_idx + offset;
            let old_hash = *unsafe{hashes.get_unchecked(offset)};
            //If first column is null then fill with 0
            *unsafe{hashes.get_unchecked_mut(offset)} = if FIRST_COL && is_null(arr, idx){
                0
            }else if is_null(arr, idx){
                //If other column is null do not modify hash. Done this way to encourage conditional move instructions
                old_hash
            }else{
                //If value is not null calculate has normally
                let new_hash = get_hash(arr,random_state, start_idx + offset);
                if FIRST_COL{
                    new_hash
                }else{
                    combine_hashes(new_hash, old_hash)
                }
            };
        }
    }
}

pub (crate) fn hash_array_boolean<const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize){
    hash_array::<BooleanArray, _, _, N, FIRST_COL>(column, hashes, random_state, start_idx, num_to_process, 
        |arr, idx|  arr.is_null(idx),
        |arr, r, idx| {
            let value = unsafe{arr.value_unchecked(idx)} as u8;
            u8::get_hash(&value,r)
        }
    )
}

pub (crate) fn hash_array_utf8<StringIndexType: StringOffsetSizeTrait,const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize){
    hash_array::<GenericStringArray<StringIndexType>, _, _, N, FIRST_COL>(column, hashes, random_state, start_idx, num_to_process, 
        |arr, idx|  arr.is_null(idx),
        |arr, r, idx| {
            let value = unsafe{arr.value_unchecked(idx)};
            str::get_hash(&value,r)
        }
    )
}

pub(crate) fn hash_array_primitive<T: ArrowPrimitiveType, const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize)
where T::Native: std::hash::Hash{
    hash_array_native::<T, _, N, FIRST_COL>(column, hashes, random_state, start_idx, num_to_process, |val, r| <T as ArrowPrimitiveType>::Native::get_hash(&val, r))
}

pub (crate) fn hash_f64_array<const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize){
    hash_array_native::<Float64Type,_, N, FIRST_COL>(column, hashes, random_state, start_idx, num_to_process, |val, r| {
        let v = u64::from_ne_bytes(val.to_ne_bytes());
        let hash = <UInt64Type as ArrowPrimitiveType>::Native::get_hash(&v, r);
        hash
    })
}

pub (crate) fn hash_f32_array<const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize){
    hash_array_native::<Float32Type,_, N, FIRST_COL>(column, hashes, random_state, start_idx, num_to_process, |val, r| {
        let v = u32::from_ne_bytes(val.to_ne_bytes());
        let hash = <UInt32Type as ArrowPrimitiveType>::Native::get_hash(&v, r);
        hash
    })
}

pub (crate) fn hash_array_native<T: ArrowPrimitiveType, HashVal: Fn(T::Native, &RandomState)->u64 +'static + Copy,const N: usize, const FIRST_COL: bool>(column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: usize, num_to_process: usize, hash_value: HashVal ){
   
    
    let array = column.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    assert!(num_to_process <= N);
    
    assert!(start_idx + num_to_process <= array.len());
    if array.null_count() == 0{
        let values = array.values();
        for offset in 0..num_to_process{
            *unsafe{hashes.get_unchecked_mut(offset)} = {
                //SAFETY: Since num_to_process > offset and array.len() >= start_idx + num_to_process this will always be withing bounds
                let value = *unsafe{values.get_unchecked(start_idx + offset)};
                let new_hash = hash_value(value, random_state);
                if FIRST_COL{
                    new_hash
                }else{
                    combine_hashes(new_hash, *unsafe{hashes.get_unchecked(offset)})
                }
            }
        }
    }else{
        for offset in 0..num_to_process{
            let idx = start_idx + offset;
            let old_hash = *unsafe{hashes.get_unchecked(offset)};
            //If first column is null then fill with 0
            *unsafe{hashes.get_unchecked_mut(offset)} = if FIRST_COL && array.is_null(idx){
                0
            }else if array.is_null(idx){
                //If other column is null do not modify hash. Done this way to encourage conditional move instructions
                old_hash
            }else{
                //If value is not null calculate has normally
                let value = unsafe{array.value_unchecked(idx)};
                let new_hash = hash_value(value, random_state);
                if FIRST_COL{
                    new_hash
                }else{
                    combine_hashes(new_hash, old_hash)
                }
            };
        }
    }
}






macro_rules! hash_array_primitive {
    ($array_type:ident, $column: ident, $ty: ident, $hashes: ident, $random_state: ident, $multi_col: ident) => {
        let array = $column.as_any().downcast_ref::<$array_type>().unwrap();
        let values = array.values();

        if array.null_count() == 0 {
            if $multi_col {
                for (hash, value) in $hashes.iter_mut().zip(values.iter()) {
                    *hash = combine_hashes($ty::get_hash(value, $random_state), *hash);
                }
            } else {
                for (hash, value) in $hashes.iter_mut().zip(values.iter()) {
                    *hash = $ty::get_hash(value, $random_state)
                }
            }
        } else {
            if $multi_col {
                for (i, (hash, value)) in
                    $hashes.iter_mut().zip(values.iter()).enumerate()
                {
                    if !array.is_null(i) {
                        *hash =
                            combine_hashes($ty::get_hash(value, $random_state), *hash);
                    }
                }
            } else {
                for (i, (hash, value)) in
                    $hashes.iter_mut().zip(values.iter()).enumerate()
                {
                    if !array.is_null(i) {
                        *hash = $ty::get_hash(value, $random_state);
                    }
                }
            }
        }
    };
}

macro_rules! hash_array_float {
    ($array_type:ident, $column: ident, $ty: ident, $hashes: ident, $random_state: ident, $multi_col: ident) => {
        let array = $column.as_any().downcast_ref::<$array_type>().unwrap();
        let values = array.values();

        if array.null_count() == 0 {
            if $multi_col {
                for (hash, value) in $hashes.iter_mut().zip(values.iter()) {
                    *hash = combine_hashes(
                        $ty::get_hash(
                            &$ty::from_le_bytes(value.to_le_bytes()),
                            $random_state,
                        ),
                        *hash,
                    );
                }
            } else {
                for (hash, value) in $hashes.iter_mut().zip(values.iter()) {
                    *hash = $ty::get_hash(
                        &$ty::from_le_bytes(value.to_le_bytes()),
                        $random_state,
                    )
                }
            }
        } else {
            if $multi_col {
                for (i, (hash, value)) in
                    $hashes.iter_mut().zip(values.iter()).enumerate()
                {
                    if !array.is_null(i) {
                        *hash = combine_hashes(
                            $ty::get_hash(
                                &$ty::from_le_bytes(value.to_le_bytes()),
                                $random_state,
                            ),
                            *hash,
                        );
                    }
                }
            } else {
                for (i, (hash, value)) in
                    $hashes.iter_mut().zip(values.iter()).enumerate()
                {
                    if !array.is_null(i) {
                        *hash = $ty::get_hash(
                            &$ty::from_le_bytes(value.to_le_bytes()),
                            $random_state,
                        );
                    }
                }
            }
        }
    };
}

/// Hash the values in a dictionary array
fn create_hashes_dictionary<K: ArrowDictionaryKeyType>(
    array: &ArrayRef,
    random_state: &RandomState,
    hashes_buffer: &mut [u64],
    multi_col: bool,
) -> Result<()> {
    let dict_array = array.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();

    // Hash each dictionary value once, and then use that computed
    // hash for each key value to avoid a potentially expensive
    // redundant hashing for large dictionary elements (e.g. strings)
    let dict_values = Arc::clone(dict_array.values());
    let mut dict_hashes = vec![0; dict_values.len()];
    create_hashes(&[dict_values], random_state, &mut dict_hashes)?;

    // combine hash for each index in values
    if multi_col {
        for (hash, key) in hashes_buffer.iter_mut().zip(dict_array.keys().iter()) {
            if let Some(key) = key {
                let idx = key
                    .to_usize()
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "Can not convert key value {:?} to usize in dictionary of type {:?}",
                            key, dict_array.data_type()
                        ))
                    })?;
                *hash = combine_hashes(dict_hashes[idx], *hash)
            } // no update for Null, consistent with other hashes
        }
    } else {
        for (hash, key) in hashes_buffer.iter_mut().zip(dict_array.keys().iter()) {
            if let Some(key) = key {
                let idx = key
                    .to_usize()
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "Can not convert key value {:?} to usize in dictionary of type {:?}",
                            key, dict_array.data_type()
                        ))
                    })?;
                *hash = dict_hashes[idx]
            } // no update for Null, consistent with other hashes
        }
    }
    Ok(())
}

/// Test version of `create_hashes` that produces the same value for
/// all hashes (to test collisions)
///
/// See comments on `hashes_buffer` for more details
#[cfg(feature = "force_hash_collisions")]
pub fn create_hashes<'a>(
    _arrays: &[ArrayRef],
    _random_state: &RandomState,
    hashes_buffer: &'a mut Vec<u64>,
) -> Result<&'a mut Vec<u64>> {
    for hash in hashes_buffer.iter_mut() {
        *hash = 0
    }
    return Ok(hashes_buffer);
}

pub (crate) fn create_hashes_new<const N: usize>(arrays: &[ArrayRef], random_state: &RandomState, hashes_buffer: &mut[u64; N], start_idx: u32, num_to_process: u32)->Result<()>{
    hash_column::<N, true>(&arrays[0], random_state, hashes_buffer, start_idx, num_to_process)?;
    for col in arrays.iter().skip(1){
        hash_column::<N, false>(col, random_state, hashes_buffer, start_idx, num_to_process)?;
    }
    Ok(())
}

fn hash_column<const N: usize, const FIRST_COL: bool>(col: &ArrayRef, random_state: &RandomState, hashes_buffer: &mut[u64; N], start_idx: u32, num_to_process: u32)->Result<()>{
    // combine hashes with `combine_hashes` if we have more than 1 column
    let col = col.as_ref();
    if num_to_process > N{
        return Err(DataFusionError::Internal(format!("Attempted to process more than {} elements at once, exceeded buffer length {}.", num_to_process, N)));
    }
    if num_to_process > N{
        unsafe{std::hint::unreachable_unchecked()};
    }
    match col.data_type() {
        DataType::Decimal(_, _) => hash_decimal128_new::<N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt8 => hash_array_primitive::<UInt8Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt16 =>  hash_array_primitive::<UInt16Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt32 =>  hash_array_primitive::<UInt32Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt64 =>  hash_array_primitive::<UInt64Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int8 =>  hash_array_primitive::<Int8Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int16 =>  hash_array_primitive::<Int16Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int32 =>  hash_array_primitive::<Int32Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int64 =>  hash_array_primitive::<Int64Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Float32 =>  hash_f32_array::<N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Float64 =>  hash_f64_array::<N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Second, None) => hash_array_primitive::<TimestampSecondType, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Millisecond, None) =>  hash_array_primitive::<TimestampMillisecondType, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Microsecond, None) =>  hash_array_primitive::<TimestampMicrosecondType, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Nanosecond, _) =>  hash_array_primitive::<TimestampNanosecondType, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Date32 =>  hash_array_primitive::<Date32Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Date64 =>  hash_array_primitive::<Date64Type, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Boolean => hash_array_boolean::<N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Utf8 => hash_array_utf8::<i32, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::LargeUtf8 => hash_array_utf8::<i64, N, FIRST_COL>(col, hashes_buffer, random_state, start_idx, num_to_process),
        /*DataType::Dictionary(index_type, _) => match **index_type {
            DataType::Int8 => {
                create_hashes_dictionary::<Int8Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::Int16 => {
                create_hashes_dictionary::<Int16Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::Int32 => {
                create_hashes_dictionary::<Int32Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::Int64 => {
                create_hashes_dictionary::<Int64Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::UInt8 => {
                create_hashes_dictionary::<UInt8Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::UInt16 => {
                create_hashes_dictionary::<UInt16Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::UInt32 => {
                create_hashes_dictionary::<UInt32Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            DataType::UInt64 => {
                create_hashes_dictionary::<UInt64Type>(
                    col,
                    random_state,
                    hashes_buffer,
                    multi_col,
                )?;
            }
            _ => {
                return Err(DataFusionError::Internal(format!(
                    "Unsupported dictionary type in hasher hashing: {}",
                    col.data_type(),
                )))
            }
        },*/
        _ => {
            // This is internal because we should have caught this before.
            return Err(DataFusionError::Internal(format!(
                "Unsupported data type in hasher: {}",
                col.data_type()
            )));
        }
    }   
    Ok(()) 
}

/// Creates hash values for every row, based on the values in the
/// columns.
///
/// The number of rows to hash is determined by `hashes_buffer.len()`.
/// `hashes_buffer` should be pre-sized appropriately
#[cfg(not(feature = "force_hash_collisions"))]
pub fn create_hashes<'a>(
    arrays: &[ArrayRef],
    random_state: &RandomState,
    hashes_buffer: &'a mut Vec<u64>,
) -> Result<&'a mut Vec<u64>> {
    // combine hashes with `combine_hashes` if we have more than 1 column
    let multi_col = arrays.len() > 1;

    for col in arrays {
        match col.data_type() {
            DataType::Decimal(_, _) => {
                hash_decimal128(col, random_state, hashes_buffer, multi_col);
            }
            DataType::UInt8 => {
                hash_array_primitive!(
                    UInt8Array,
                    col,
                    u8,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::UInt16 => {
                hash_array_primitive!(
                    UInt16Array,
                    col,
                    u16,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::UInt32 => {
                hash_array_primitive!(
                    UInt32Array,
                    col,
                    u32,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::UInt64 => {
                hash_array_primitive!(
                    UInt64Array,
                    col,
                    u64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Int8 => {
                hash_array_primitive!(
                    Int8Array,
                    col,
                    i8,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Int16 => {
                hash_array_primitive!(
                    Int16Array,
                    col,
                    i16,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Int32 => {
                hash_array_primitive!(
                    Int32Array,
                    col,
                    i32,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Int64 => {
                hash_array_primitive!(
                    Int64Array,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Float32 => {
                hash_array_float!(
                    Float32Array,
                    col,
                    u32,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Float64 => {
                hash_array_float!(
                    Float64Array,
                    col,
                    u64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Timestamp(TimeUnit::Second, None) => {
                hash_array_primitive!(
                    TimestampSecondArray,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Timestamp(TimeUnit::Millisecond, None) => {
                hash_array_primitive!(
                    TimestampMillisecondArray,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                hash_array_primitive!(
                    TimestampMicrosecondArray,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                hash_array_primitive!(
                    TimestampNanosecondArray,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Date32 => {
                hash_array_primitive!(
                    Date32Array,
                    col,
                    i32,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Date64 => {
                hash_array_primitive!(
                    Date64Array,
                    col,
                    i64,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Boolean => {
                hash_array!(
                    BooleanArray,
                    col,
                    u8,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Utf8 => {
                hash_array!(
                    StringArray,
                    col,
                    str,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::LargeUtf8 => {
                hash_array!(
                    LargeStringArray,
                    col,
                    str,
                    hashes_buffer,
                    random_state,
                    multi_col
                );
            }
            DataType::Dictionary(index_type, _) => match **index_type {
                DataType::Int8 => {
                    create_hashes_dictionary::<Int8Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::Int16 => {
                    create_hashes_dictionary::<Int16Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::Int32 => {
                    create_hashes_dictionary::<Int32Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::Int64 => {
                    create_hashes_dictionary::<Int64Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::UInt8 => {
                    create_hashes_dictionary::<UInt8Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::UInt16 => {
                    create_hashes_dictionary::<UInt16Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::UInt32 => {
                    create_hashes_dictionary::<UInt32Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                DataType::UInt64 => {
                    create_hashes_dictionary::<UInt64Type>(
                        col,
                        random_state,
                        hashes_buffer,
                        multi_col,
                    )?;
                }
                _ => {
                    return Err(DataFusionError::Internal(format!(
                        "Unsupported dictionary type in hasher hashing: {}",
                        col.data_type(),
                    )))
                }
            },
            _ => {
                // This is internal because we should have caught this before.
                return Err(DataFusionError::Internal(format!(
                    "Unsupported data type in hasher: {}",
                    col.data_type()
                )));
            }
        }
    }
    Ok(hashes_buffer)
}

#[cfg(test)]
mod tests {
    use crate::from_slice::FromSlice;
    use arrow::{array::DictionaryArray, datatypes::Int8Type};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn create_hashes_for_decimal_array() -> Result<()> {
        let array = vec![1, 2, 3, 4]
            .into_iter()
            .map(Some)
            .collect::<DecimalArray>()
            .with_precision_and_scale(20, 3)
            .unwrap();
        let array_ref = Arc::new(array);
        let random_state = RandomState::with_seeds(0, 0, 0, 0);
        let hashes_buff = &mut vec![0; array_ref.len()];
        let hashes = create_hashes(&[array_ref], &random_state, hashes_buff)?;
        assert_eq!(hashes.len(), 4);
        Ok(())
    }

    #[test]
    fn create_hashes_for_float_arrays() -> Result<()> {
        let f32_arr = Arc::new(Float32Array::from_slice(&[0.12, 0.5, 1f32, 444.7]));
        let f64_arr = Arc::new(Float64Array::from_slice(&[0.12, 0.5, 1f64, 444.7]));

        let random_state = RandomState::with_seeds(0, 0, 0, 0);
        let hashes_buff = &mut vec![0; f32_arr.len()];
        let hashes = create_hashes(&[f32_arr], &random_state, hashes_buff)?;
        assert_eq!(hashes.len(), 4,);

        let hashes = create_hashes(&[f64_arr], &random_state, hashes_buff)?;
        assert_eq!(hashes.len(), 4,);

        Ok(())
    }

    #[test]
    // Tests actual values of hashes, which are different if forcing collisions
    #[cfg(not(feature = "force_hash_collisions"))]
    fn create_hashes_for_dict_arrays() {
        let strings = vec![Some("foo"), None, Some("bar"), Some("foo"), None];

        let string_array = Arc::new(strings.iter().cloned().collect::<StringArray>());
        let dict_array = Arc::new(
            strings
                .iter()
                .cloned()
                .collect::<DictionaryArray<Int8Type>>(),
        );

        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        let mut string_hashes = vec![0; strings.len()];
        create_hashes(&[string_array], &random_state, &mut string_hashes).unwrap();

        let mut dict_hashes = vec![0; strings.len()];
        create_hashes(&[dict_array], &random_state, &mut dict_hashes).unwrap();

        // Null values result in a zero hash,
        for (val, hash) in strings.iter().zip(string_hashes.iter()) {
            match val {
                Some(_) => assert_ne!(*hash, 0),
                None => assert_eq!(*hash, 0),
            }
        }

        // same logical values should hash to the same hash value
        assert_eq!(string_hashes, dict_hashes);

        // Same values should map to same hash values
        assert_eq!(strings[1], strings[4]);
        assert_eq!(dict_hashes[1], dict_hashes[4]);
        assert_eq!(strings[0], strings[3]);
        assert_eq!(dict_hashes[0], dict_hashes[3]);

        // different strings should map to different hash values
        assert_ne!(strings[0], strings[2]);
        assert_ne!(dict_hashes[0], dict_hashes[2]);
    }

    #[test]
    // Tests actual values of hashes, which are different if forcing collisions
    #[cfg(not(feature = "force_hash_collisions"))]
    fn create_multi_column_hash_for_dict_arrays() {
        let strings1 = vec![Some("foo"), None, Some("bar")];
        let strings2 = vec![Some("blarg"), Some("blah"), None];

        let string_array = Arc::new(strings1.iter().cloned().collect::<StringArray>());
        let dict_array = Arc::new(
            strings2
                .iter()
                .cloned()
                .collect::<DictionaryArray<Int32Type>>(),
        );

        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        let mut one_col_hashes = vec![0; strings1.len()];
        create_hashes(&[dict_array.clone()], &random_state, &mut one_col_hashes).unwrap();

        let mut two_col_hashes = vec![0; strings1.len()];
        create_hashes(
            &[dict_array, string_array],
            &random_state,
            &mut two_col_hashes,
        )
        .unwrap();

        assert_eq!(one_col_hashes.len(), 3);
        assert_eq!(two_col_hashes.len(), 3);

        assert_ne!(one_col_hashes, two_col_hashes);
    }
}
