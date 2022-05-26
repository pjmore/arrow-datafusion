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
    UInt16Array, UInt32Array, UInt64Array, UInt8Array
};
use arrow::datatypes::{
    ArrowDictionaryKeyType, ArrowNativeType, DataType, TimeUnit, Int8Type, Int32Type, Int16Type, Int64Type, UInt8Type, UInt16Type, UInt64Type, UInt32Type
};
use std::sync::Arc;

// Combines two hashes into one hash
#[inline]
fn combine_hashes(l: u64, r: u64) -> u64 {
    let hash = (17 * 37u64).wrapping_add(l);
    hash.wrapping_mul(37).wrapping_add(r)
}

fn null_hash_value(random_state: &RandomState)->u64{
    usize::get_hash(&0xf0f0f0f, random_state)
}

fn hash_null<const N: usize, const FIRST_COL: bool, const MULTI_COL: bool>(hashes: &mut [u64; N], random_state: &RandomState, num_to_process: u32) {
    let null_hash_value = null_hash_value(random_state);
    for hash in hashes.iter_mut().take(num_to_process as usize){
        let new_hash = null_hash_value;
        *hash =  match (FIRST_COL, MULTI_COL){
            (true, true) => combine_hashes(new_hash, 0),
            (false, true)=> combine_hashes(new_hash, *hash),
            (_, false) => new_hash,
        };
    }
}
///This macro defines how the declare_hash_array macro gets the hash value for the array
/// Allows for specialization for types like StringArray and Decimal array which may require special handling
/// SAFETY: The caller must ensure that the ident $idx is within bounds for the ident $array.
macro_rules! get_hash_unsafe{
    (f32, $array: ident, $idx:ident, $hash_builder:ident)=>{{
        let value: f32 = $array.value_unchecked($idx);
        let value_u32_bytes = u32::from_ne_bytes(value.to_ne_bytes());
        u32::get_hash(&value_u32_bytes, $hash_builder)
    }};
    (f64, $array: ident, $idx:ident, $hash_builder:ident)=>{{
        let value: f64 = $array.value_unchecked($idx);
        let value_u64_bytes = u64::from_ne_bytes(value.to_ne_bytes());
        u64::get_hash(&value_u64_bytes, $hash_builder)
    }};
    (str, $array: ident, $idx:ident, $hash_builder:ident)=>{{
        let value:&str = $array.value_unchecked($idx);
        str::get_hash(value, $hash_builder)
    }};
    (i128, $array: ident, $idx:ident, $hash_builder:ident)=>{{
        if $idx >= $array.len(){
            ::std::hint::unreachable_unchecked();
        }
        let value = $array.value($idx);
        i128::get_hash(&value, $hash_builder)
    }};
    ($ty:ident, $array: ident, $idx:ident, $hash_builder:ident)=>{{
        let value = $array.value_unchecked($idx);
        <$ty>::get_hash(&value, $hash_builder)
    }};
}



macro_rules! declare_hash_array{
    (DECLARE: $func_name:ident, $array_type:ty, $ty:ident, $buf_size:literal)=>{
        fn $func_name<const N: usize, const FIRST_COL: bool, const MULTI_COL: bool, const FULL_RUN: bool> (column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: u32, num_to_process: u32){
            let array = column.as_any().downcast_ref::<$array_type>().unwrap();
            const INTERNAL_HASH_BUFFER_SIZE: usize = $buf_size;
            const INTERNAL_HASH_BUFFER_SIZE_U32: u32 = INTERNAL_HASH_BUFFER_SIZE as u32;
            let mut hash_buffer = [0u64; INTERNAL_HASH_BUFFER_SIZE];
            let mut offset = start_idx;
            let num_to_process = if FULL_RUN{
                N as u32
            }else{
                num_to_process
            };
            let mut mut_chunks = (&mut hashes[..(num_to_process as usize)]).chunks_exact_mut(INTERNAL_HASH_BUFFER_SIZE);
            if array.null_count() == 0{
                for chunk in &mut mut_chunks{
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        let idx = offset as usize + i;
                        //SAFETY: i is always less than INTERNAL_HASH_BUFFER_SIZE. Since the chunks and the offset are incremented by INTERNAL_HASH_BUFFER_SIZE_U32,
                        // which must ALWAYS be equal to INTERNAL_HASH_BUFFER_SIZE and the chunks are always guaranteed to be within bounds for the slice, this means that so long as 
                        // the offset is incremented by INTERNAL_HASH_BUFFER_SIZE_U32 only when a chunk is changed that offset will always be at least INTERNAL_HASH_BUFFER_SIZE elements away
                        // the end of the loop. Since i < INTERNAL_HASH_BUFFER_SIZE_U32 the access will always be within bounds.
                        hash_buffer[i as usize] = unsafe{get_hash_unsafe!($ty, array,idx, random_state)};
                    }
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        chunk[i] = match (FIRST_COL, MULTI_COL){
                            (true, true) => combine_hashes(hash_buffer[i], 0),
                            (false, true)=> combine_hashes(hash_buffer[i], chunk[i]),
                            (_, false) => hash_buffer[i],
                        };
                    }
                    offset+=INTERNAL_HASH_BUFFER_SIZE_U32;
                    
                }
                let remainder = mut_chunks.into_remainder();
                //Ensures that all remaining unchecked memory accesses are within bounds for the array.
                assert!(offset as usize + remainder.len() <= array.len());
                for hash in remainder.iter_mut(){
                    let idx = offset as usize;
                    //SAFETY: At the end of the last loop the offset will be contain the index of the next element to process.
                    // Since offset is incremented after the memory access and because of the assert above the memory access will always be within bounds
                    let new_hash = unsafe{get_hash_unsafe!($ty, array,idx, random_state)};
                    *hash = match (FIRST_COL, MULTI_COL){
                        (true, true) => combine_hashes(new_hash, 0),
                        (false, true)=> combine_hashes(new_hash, *hash),
                        (_, false)=> new_hash,
                    };
                    offset +=1;
                }
            }else{
                let null_hash_val = null_hash_value(random_state);
                for chunk in &mut mut_chunks{
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        let arr_idx = offset as usize +i;
                        //SAFETY: i is always less than INTERNAL_HASH_BUFFER_SIZE. Since the chunks and the offset are incremented by INTERNAL_HASH_BUFFER_SIZE_U32,
                        // which must ALWAYS be equal to INTERNAL_HASH_BUFFER_SIZE and the chunks are always guaranteed to be within bounds for the slice, this means that so long as 
                        // the offset is incremented by INTERNAL_HASH_BUFFER_SIZE_U32 only when a chunk is changed that offset will always be at least INTERNAL_HASH_BUFFER_SIZE elements away
                        // the end of the loop. Since i < INTERNAL_HASH_BUFFER_SIZE_U32 the access will always be within bounds.             
                        let val_hash = unsafe{get_hash_unsafe!($ty, array,arr_idx, random_state)};
                        hash_buffer[i] = if array.is_null(arr_idx){
                            null_hash_val                           
                        }else{
                            val_hash
                        };
                    }
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        chunk[i] = match (FIRST_COL, MULTI_COL){
                            (true, true) => combine_hashes(hash_buffer[i], 0),
                            (false, true)=> combine_hashes(hash_buffer[i], chunk[i]),
                            _=> hash_buffer[i],
                        };
                    }
                    offset+=INTERNAL_HASH_BUFFER_SIZE_U32;
                }
                let remainder = mut_chunks.into_remainder();
                //Ensures that all remaining unchecked accesses are within bounds for the array.
                assert!(offset as usize + remainder.len() <= array.len());
                for hash in remainder.iter_mut(){
                    
                    let arr_idx = offset as usize;
                    //SAFETY: At the end of the last loop the offset will be contain the index of the next element to process.
                    // Since offset is incremented after the memory access and because of the assert above the memory access will always be within bounds
                    let val_hash = unsafe{get_hash_unsafe!($ty, array,arr_idx, random_state)};
                    let new_hash = if array.is_null(arr_idx){
                        null_hash_val
                    }else{
                        val_hash
                    };
                    *hash = match (FIRST_COL, MULTI_COL){
                        (true, true) => combine_hashes(new_hash, 0),
                        (false, true)=> combine_hashes(new_hash, *hash),
                        _=> new_hash,
                    };
                    offset +=1;
                }
            }
        }
    };
}


//Primitive types we know how to hash since ownership and mapping from arrow is straight forward.
declare_hash_array!(DECLARE: hash_uint8_array_chunk, UInt8Array, u8, 32);
declare_hash_array!(DECLARE: hash_uint16_array_chunk, UInt16Array, u16, 32);
declare_hash_array!(DECLARE: hash_uint32_array_chunk, UInt32Array, u32, 32);
declare_hash_array!(DECLARE: hash_uint64_array_chunk, UInt64Array, u64, 32);
declare_hash_array!(DECLARE: hash_int8_array_chunk, Int8Array, i8, 32);
declare_hash_array!(DECLARE: hash_int16_array_chunk, Int16Array, i16, 32);
declare_hash_array!(DECLARE: hash_int32_array_chunk, Int32Array, i32, 32);
declare_hash_array!(DECLARE: hash_int64_array_chunk, Int64Array, i64, 32);
declare_hash_array!(DECLARE: hash_date32_array_chunk, Date32Array, i32, 16);
declare_hash_array!(DECLARE: hash_date64_array_chunk, Date64Array, i64, 16);
declare_hash_array!(DECLARE: hash_time_s_array_chunk, TimestampSecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_ms_array_chunk, TimestampMillisecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_us_array_chunk, TimestampMicrosecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_ns_array_chunk, TimestampNanosecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_boolean_array_chunk, BooleanArray, bool, 32);

//The below 5 items require special handling that is done in get_hash_unsafe. 
//Required due to floats not being default hashable. Use to_ne_bytes() ::from_ne_bytes() to transumte to 
//u32 or u64 without requiring any actual work be done.
declare_hash_array!(DECLARE: hash_float32_array_chunk, Float32Array, f32, 16);
declare_hash_array!(DECLARE: hash_float64_array_chunk, Float64Array, f64, 16);
//Requires special handling due to ownership stuff. Array returns &str while primtive arrays usually return T
declare_hash_array!(DECLARE: hash_utf8_array_chunk, StringArray, str, 16);
declare_hash_array!(DECLARE: hash_largeutf8_array_chunk, LargeStringArray, str, 16);
//Does not have value_unchecked
declare_hash_array!(DECLARE: hash_decimal_array_chunk, DecimalArray, i128, 16);




fn create_hashes_dictionary_chunk<K: ArrowDictionaryKeyType, const N: usize, const FIRST_COL: bool, const MULTI_COL: bool, const FULL_RUN: bool>(
    array: &dyn Array,
    random_state: &RandomState,
    hashes: &mut [u64; N],
    hash_cache: &mut Vec<u64>,
    start_idx: u32,
    num_to_process: u32,
)-> Result<()>{
    let num_to_process = if FULL_RUN{
        N as u32
    }else {
        num_to_process
    };
    let dict_array = array.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();
    if hash_cache.len() == 0{
        // Hash each dictionary value once, and then use that computed
        // hash for each key value to avoid a potentially expensive
        // redundant hashing for large dictionary elements (e.g. strings)
        let dict_values = Arc::clone(dict_array.values());
        hash_cache.reserve(dict_values.len());
        for _ in 0..dict_values.len(){
            hash_cache.push(0);
        }
        create_hashes(&[dict_values], random_state, hash_cache)?;
    }
    let keys = dict_array.keys().values();
    let null_hash_value = null_hash_value(random_state);
    for offset in 0.. num_to_process{
        let idx = (start_idx + offset) as usize;
        let key = keys[idx].to_usize()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Can not convert key value {:?} to usize in dictionary of type {:?}",
                    keys[idx], dict_array.data_type()
                ))
            })?;
        
        let hash = if dict_array.keys().is_null(idx){
            null_hash_value
        } else{
            hash_cache[key]
        };
        
        hashes[offset as usize]=match (FIRST_COL, MULTI_COL){
            (true, true) => combine_hashes(hash, 0),
            (false, true)=> combine_hashes(hash, hashes[offset as usize]),
            (_, false)=> hash,
        };
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
    Ok(hashes_buffer)
}

/// Test version of `create_row_hashes` that produces the same value for
/// all hashes (to test collisions)
///
/// See comments on `hashes_buffer` for more details
#[cfg(feature = "force_hash_collisions")]
pub fn create_row_hashes<'a>(
    _rows: &[Vec<u8>],
    _random_state: &RandomState,
    hashes_buffer: &'a mut Vec<u64>,
) -> Result<&'a mut Vec<u64>> {
    for hash in hashes_buffer.iter_mut() {
        *hash = 0
    }
    Ok(hashes_buffer)
}

/// Creates hash values for every row, based on their raw bytes.
#[cfg(not(feature = "force_hash_collisions"))]
pub fn create_row_hashes<'a>(
    rows: &[Vec<u8>],
    random_state: &RandomState,
    hashes_buffer: &'a mut Vec<u64>,
) -> Result<&'a mut Vec<u64>> {
    for hash in hashes_buffer.iter_mut() {
        *hash = 0
    }
    for (i, hash) in hashes_buffer.iter_mut().enumerate() {
        *hash = <Vec<u8>>::get_hash(&rows[i], random_state);
    }
    Ok(hashes_buffer)
}

#[cfg(feature="force_hash_collisions")]
pub (crate) fn create_hashes_chunked<const N: usize>(arrays: &[ArrayRef], random_state: &RandomState, hashes_buffer: &mut[u64; N], dictionary_hash_cache:&mut[Vec<u64>], start_idx: u32, num_to_process: u32)->Result<()>{
    for hash in hashes_buffer.iter_mut(){
        *hash = 0;
    }
    Ok(())
}

#[cfg(not(feature="force_hash_collisions"))]
pub (crate) fn create_hashes_chunked<const N: usize, const FULL_RUN: bool>(arrays: &[ArrayRef], random_state: &RandomState, hashes_buffer: &mut[u64; N], dictionary_hash_cache:&mut[Vec<u64>], start_idx: u32, num_to_process: u32)->Result<()>{
    assert!(dictionary_hash_cache.len() == arrays.len());
    if arrays.len() == 1{
        hash_column_chunk::<N, true, false, FULL_RUN>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0], start_idx, num_to_process)?;
    }else{
        hash_column_chunk::<N, true, true, FULL_RUN>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0], start_idx, num_to_process)?;
        for (idx, col) in arrays.iter().enumerate().skip(1){
            hash_column_chunk::<N, false, true, FULL_RUN>(col, random_state, hashes_buffer, &mut dictionary_hash_cache[idx],start_idx, num_to_process)?;
        }
    }
    Ok(())
}

fn hash_column_chunk<const N: usize, const FIRST_COL: bool, const MULTI_COL: bool, const FULL_RUN: bool>(col: &dyn Array, random_state: &RandomState, hashes_buffer: &mut[u64; N], dictionary_hash_cache: &mut Vec<u64>, start_idx: u32, num_to_process: u32)->Result<()>{
    // combine hashes with `combine_hashes` if we have more than 1 column
    let num_to_process = if FULL_RUN{
        N as u32
    }else{
        num_to_process
    };
    if num_to_process as usize > N{
        return Err(DataFusionError::Internal(format!("Attempted to process more than {} elements at once, exceeded buffer length {}.", num_to_process, N)));
    }
    match col.data_type() {
        DataType::Null => hash_null::<N, FIRST_COL, MULTI_COL>(hashes_buffer, random_state, num_to_process),
        DataType::Decimal(_, _) => hash_decimal_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt8 => hash_uint8_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt16 =>  hash_uint16_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt32 =>  hash_uint32_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::UInt64 =>  hash_uint64_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int8 =>  hash_int8_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int16 =>  hash_int16_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int32 =>  hash_int32_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Int64 =>  hash_int64_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Float32 =>  hash_float32_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Float64 =>  hash_float64_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Second, None) => hash_time_s_array_chunk::< N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Millisecond, None) =>  hash_time_ms_array_chunk::< N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Microsecond, None) =>  hash_time_us_array_chunk::< N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Timestamp(TimeUnit::Nanosecond, None) =>  hash_time_ns_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Date32 =>  hash_date32_array_chunk::< N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Date64 =>  hash_date64_array_chunk::< N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Boolean => hash_boolean_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Utf8 => hash_utf8_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::LargeUtf8 => hash_largeutf8_array_chunk::<N, FIRST_COL, MULTI_COL, FULL_RUN>(col, hashes_buffer, random_state, start_idx, num_to_process),
        DataType::Dictionary(index_type, _) => match **index_type {
            DataType::Int8 => {
                create_hashes_dictionary_chunk::<Int8Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::Int16 => {
                create_hashes_dictionary_chunk::<Int16Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::Int32 => {
                create_hashes_dictionary_chunk::<Int32Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::Int64 => {
                create_hashes_dictionary_chunk::<Int64Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::UInt8 => {
                create_hashes_dictionary_chunk::<UInt8Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::UInt16 => {
                create_hashes_dictionary_chunk::<UInt16Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::UInt32 => {
                create_hashes_dictionary_chunk::<UInt32Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
                )?;
            }
            DataType::UInt64 => {
                create_hashes_dictionary_chunk::<UInt64Type, N, FIRST_COL, MULTI_COL, FULL_RUN>(
                    col,
                    random_state,
                    hashes_buffer,
                    dictionary_hash_cache,
                    start_idx,
                    num_to_process
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
    const BUF_SIZE: usize = 1024;
    let mut offset = 0;
    let mut dictionary_hash_cache = vec![Vec::default(); arrays.len()];
    let mut chunks = hashes_buffer.chunks_exact_mut(BUF_SIZE);
    for chunk in &mut chunks{
        let chunk_array =  <&mut [u64; BUF_SIZE]>::try_from(chunk).unwrap();
        create_hashes_chunked::<BUF_SIZE, true>(arrays, random_state, chunk_array, &mut dictionary_hash_cache,offset as u32, BUF_SIZE as u32)?;
        offset+=BUF_SIZE;
    }
    let remainder = chunks.into_remainder();
    if remainder.len() != 0{
        let mut buffer = [0u64;BUF_SIZE];
        create_hashes_chunked::<BUF_SIZE, false>(arrays, random_state, &mut buffer,&mut dictionary_hash_cache,offset as u32, remainder.len() as u32)?;
        for i in 0..remainder.len(){
            let idx = offset + i;
            hashes_buffer[idx] = buffer[i];
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
        let null_hash_value = null_hash_value(&random_state);
        for (val, hash) in strings.iter().zip(string_hashes.iter()) {
            match val {
                Some(_) => assert_ne!(*hash, 0),
                None => assert_eq!(*hash, null_hash_value),
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
        use arrow::datatypes::Int32Type;

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
