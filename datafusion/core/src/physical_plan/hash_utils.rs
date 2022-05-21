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





const NULL_HASH_VALUE: u64 = 0;

macro_rules! declare_hash_array{
    //This pattern forwards to the next pattern. Filling in the get_hash block. This only works for primitive types with a .to_ne_bytes() method.
    (DECLARE: $func_name:ident, $array_type:ty, $ty:ty, $buf_size:literal)=>{
        declare_hash_array!(DECLARE: $func_name, $array_type, $buf_size, {|val: $ty, random_state| <$ty>::get_hash(&val, random_state)});
    };

    (DECLARE: $func_name:ident, $array_type:ty, $buf_size:literal, $get_hash:block)=>{
        #[allow(dead_code)]
        fn $func_name<const N: usize, const FIRST_COL: bool, const MULTI_COL: bool, const FULL_RUN: bool> (column: &dyn Array, hashes: &mut [u64; N], random_state: &RandomState, start_idx: u32, num_to_process: u32){
            let array = column.as_any().downcast_ref::<$array_type>().unwrap();
            let get_hash = $get_hash;
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
                    //println!("Processing from {} to {} of size {} from len:{}", offset, offset+INTERNAL_HASH_BUFFER_SIZE_U32, chunk.len(), array.len());
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        hash_buffer[i as usize] = get_hash(array.value(offset as usize + i), random_state);
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
                for hash in mut_chunks.into_remainder().iter_mut(){
                    let new_hash = get_hash(array.value(offset as usize), random_state);
                    *hash = match (FIRST_COL, MULTI_COL){
                        (true, true) => combine_hashes(new_hash, 0),
                        (false, true)=> combine_hashes(new_hash, *hash),
                        (_, false)=> new_hash,
                    };
                    offset +=1;
                }
            }else{
                for chunk in &mut mut_chunks{
                    for i in 0..INTERNAL_HASH_BUFFER_SIZE{
                        let arr_idx = offset as usize +i;
                        hash_buffer[i] = if array.is_null(arr_idx){
                            NULL_HASH_VALUE                           
                        }else{
                            get_hash(array.value(arr_idx), random_state)
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
                for hash in mut_chunks.into_remainder().iter_mut(){
                    
                    let arr_idx = offset as usize;
                    let new_hash = if array.is_null(arr_idx){
                        NULL_HASH_VALUE
                    }else{
                        get_hash(array.value(arr_idx), random_state)
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
declare_hash_array!(DECLARE: hash_uint8_array_chunk, UInt8Array, u8, 16);
declare_hash_array!(DECLARE: hash_uint16_array_chunk, UInt16Array, u16, 16);
declare_hash_array!(DECLARE: hash_uint32_array_chunk, UInt32Array, u32, 16);
declare_hash_array!(DECLARE: hash_uint64_array_chunk, UInt64Array, u64, 16);
declare_hash_array!(DECLARE: hash_int8_array_chunk, Int8Array, i8, 16);
declare_hash_array!(DECLARE: hash_int16_array_chunk, Int16Array, i16, 16);
declare_hash_array!(DECLARE: hash_int32_array_chunk, Int32Array, i32, 16);
declare_hash_array!(DECLARE: hash_int64_array_chunk, Int64Array, i64, 16);
declare_hash_array!(DECLARE: hash_date32_array_chunk, Date32Array, i32, 16);
declare_hash_array!(DECLARE: hash_date64_array_chunk, Date64Array, i64, 16);
declare_hash_array!(DECLARE: hash_time_s_array_chunk, TimestampSecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_ms_array_chunk, TimestampMillisecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_us_array_chunk, TimestampMicrosecondArray, i64, 16);
declare_hash_array!(DECLARE: hash_time_ns_array_chunk, TimestampNanosecondArray, i64, 16);

//Last item specifies how the hash is generated
declare_hash_array!(DECLARE: hash_float32_array_chunk, Float32Array, 16, {|val: f32, random_state| u32::get_hash(&u32::from_ne_bytes(val.to_ne_bytes()), random_state)});
declare_hash_array!(DECLARE: hash_float64_array_chunk, Float64Array, 16, {|val: f64, random_state| u64::get_hash(&u64::from_ne_bytes(val.to_ne_bytes()), random_state)});
declare_hash_array!(DECLARE: hash_utf8_array_chunk, StringArray, 16, {|val: &str, random_state| str::get_hash(val, random_state)});
declare_hash_array!(DECLARE: hash_largeutf8_array_chunk, LargeStringArray, 16, {|val: &str, random_state| str::get_hash(val, random_state)});
declare_hash_array!(DECLARE: hash_decimal_array_chunk, DecimalArray, 16, {|val: i128, random_state| i128::get_hash(&val, random_state)});
declare_hash_array!(DECLARE: hash_boolean_array_chunk, BooleanArray, 32, {|val: bool, random_state| bool::get_hash(&val, random_state)});




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
    println!("Dict arr: {:?}", dict_array);
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
            NULL_HASH_VALUE
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
    return Ok(hashes_buffer);
}

#[cfg(feature="force_hash_collisions")]
pub (crate) fn create_hashes_chunked<const N: usize>(arrays: &[ArrayRef], random_state: &RandomState, hashes_buffer: &mut[u64; N], dictionary_hash_cache:&mut[Vec<u64>], start_idx: u32, num_to_process: u32)->Result<()>{
    for hash in hashes_buffer.iter_mut(){
        *hash = 0;
    }
    Ok(())
}

#[cfg(not(feature="force_hash_collisions"))]
pub (crate) fn create_hashes_chunked<const N: usize>(arrays: &[ArrayRef], random_state: &RandomState, hashes_buffer: &mut[u64; N], dictionary_hash_cache:&mut[Vec<u64>], start_idx: u32, num_to_process: u32)->Result<()>{
    assert!(dictionary_hash_cache.len() == arrays.len());
    if num_to_process == (N as u32){
        if arrays.len() == 1{
            hash_column_chunk::<N, true, false, true>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0], start_idx, num_to_process)?;
        }else{
            hash_column_chunk::<N, true, true, true>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0], start_idx, num_to_process)?;
            for (idx, col) in arrays.iter().enumerate().skip(1){
                hash_column_chunk::<N, false, true, true>(col, random_state, hashes_buffer, &mut dictionary_hash_cache[idx],start_idx, num_to_process)?;
            }
        }
    }else{
        if arrays.len() == 1{
            hash_column_chunk::<N, true, false, false>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0], start_idx, num_to_process)?;
        }else{
            hash_column_chunk::<N, true, true, false>(&arrays[0], random_state, hashes_buffer, &mut dictionary_hash_cache[0],start_idx, num_to_process)?;
            for (idx, col) in arrays.iter().enumerate().skip(1){
                hash_column_chunk::<N, false, true, false>(col, random_state, hashes_buffer, &mut dictionary_hash_cache[idx], start_idx, num_to_process)?;
            }
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
    if num_to_process as usize > N{
        //SAFETY: This is safe since we check the condition above as well. This is to encourage removing bounds checks on array accesses.
        unsafe{std::hint::unreachable_unchecked()};
    }
    match col.data_type() {
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
    let mut buffer = [0u64;BUF_SIZE];
    let mut offset = 0;
    let len = arrays[0].len();
    let mut dictionary_hash_cache = vec![Vec::default(); arrays.len()];
    while offset + BUF_SIZE < len{
        create_hashes_chunked(arrays, random_state, &mut buffer,&mut dictionary_hash_cache,offset as u32, BUF_SIZE as u32)?;
        for i in 0..BUF_SIZE{
            hashes_buffer[offset + i] = buffer[i];
        }
        offset += BUF_SIZE;
    }
    let num_to_process = len - offset;
    if num_to_process != 0{
        create_hashes_chunked(arrays, random_state, &mut buffer,&mut dictionary_hash_cache,offset as u32, num_to_process as u32)?;
        for i in 0..num_to_process{
            hashes_buffer[i+offset] = buffer[i];
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
