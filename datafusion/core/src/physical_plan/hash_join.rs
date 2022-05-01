#![allow(unused_variables,dead_code,unused_imports)]
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

// Defines the join plan for executing partitions in parallel and then joining the results
// into a set of partitions.

use ahash::RandomState;

use arrow::{
    array::{
        ArrayData, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampSecondArray,
        UInt32BufferBuilder, UInt32Builder, UInt64BufferBuilder, UInt64Builder, StringOffsetSizeTrait, GenericStringArray, DictionaryArray, BufferBuilder,
    },
    compute,
    datatypes::{UInt32Type, UInt64Type, ArrowNativeType, ArrowPrimitiveType, ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, Float64Type, Float32Type, TimestampMillisecondType, TimestampSecondType, TimestampMicrosecondType, TimestampNanosecondType},
};
use smallvec::{smallvec, SmallVec};
use std::{sync::Arc};
use std::{any::Any, usize};
use std::{time::Instant, vec};

use async_trait::async_trait;
use futures::{Stream, StreamExt, TryStreamExt};
use tokio::sync::Mutex;

use arrow::array::{new_null_array, Array};
use arrow::datatypes::DataType;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use arrow::array::{
    Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    StringArray, TimestampNanosecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};

use hashbrown::raw::RawTable;

use super::{
    coalesce_partitions::CoalescePartitionsExec,
    expressions::PhysicalSortExpr,
    join_utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinOn, JoinSide},  hash_utils::create_hashes_new,
};
use super::{
    expressions::Column,
    metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
};
use super::{hash_utils::create_hashes, Statistics};
use crate::error::{DataFusionError, Result};
use crate::logical_plan::JoinType;

use super::{
    DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream,
    SendableRecordBatchStream,
};
use crate::arrow::array::BooleanBufferBuilder;
use crate::arrow::datatypes::TimeUnit;
use crate::execution::context::TaskContext;
use crate::physical_plan::coalesce_batches::concat_batches;
use crate::physical_plan::PhysicalExpr;
use log::debug;
use std::fmt;

// Maps a `u64` hash value based on the left ["on" values] to a list of indices with this key's value.
//
// Note that the `u64` keys are not stored in the hashmap (hence the `()` as key), but are only used
// to put the indices in a certain bucket.
// By allocating a `HashMap` with capacity for *at least* the number of rows for entries at the left side,
// we make sure that we don't have to re-hash the hashmap, which needs access to the key (the hash in this case) value.
// E.g. 1 -> [3, 6, 8] indicates that the column values map to rows 3, 6 and 8 for hash value 1
// As the key is a hash value, we need to check possible hash collisions in the probe stage
// During this stage it might be the case that a row is contained the same hashmap value,
// but the values don't match. Those are checked in the [equal_rows] macro
// TODO: speed up collission check and move away from using a hashbrown HashMap
// https://github.com/apache/arrow-datafusion/issues/50
struct JoinHashMap(RawTable<(u64, SmallVec<[u64; 1]>)>);

impl fmt::Debug for JoinHashMap {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}



type JoinLeftData = Arc<(JoinHashMap, RecordBatch)>;

/// join execution plan executes partitions in parallel and combines them into a set of
/// partitions.
#[derive(Debug)]
pub struct HashJoinExec {
    /// left (build) side which gets hashed
    left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    right: Arc<dyn ExecutionPlan>,
    /// Set of common columns used to join on
    on: Vec<(Column, Column)>,
    /// How the join is performed
    join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Build-side
    build_side: Arc<Mutex<Option<JoinLeftData>>>,
    /// Shares the `RandomState` for the hashing algorithm
    random_state: RandomState,
    /// Partitioning mode to use
    mode: PartitionMode,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// If null_equals_null is true, null == null else null != null
    null_equals_null: bool,
}

/// Metrics for HashJoinExec
#[derive(Debug)]
struct HashJoinMetrics {
    /// Total time for joining probe-side batches to the build-side batches
    join_time: metrics::Time,
    /// Number of batches consumed by this operator
    input_batches: metrics::Count,
    /// Number of rows consumed by this operator
    input_rows: metrics::Count,
    /// Number of batches produced by this operator
    output_batches: metrics::Count,
    /// Number of rows produced by this operator
    output_rows: metrics::Count,
}

impl HashJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);

        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);

        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);

        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);

        let output_rows = MetricBuilder::new(metrics).output_rows(partition);

        Self {
            join_time,
            input_batches,
            input_rows,
            output_batches,
            output_rows,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// Partitioning mode to use for hash join
pub enum PartitionMode {
    /// Left/right children are partitioned using the left and right keys
    Partitioned,
    /// Left side will collected into one partition
    CollectLeft,
}

impl HashJoinExec {
    /// Tries to create a new [HashJoinExec].
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        partition_mode: PartitionMode,
        null_equals_null: &bool,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        if on.is_empty() {
            return Err(DataFusionError::Plan(
                "On constraints in HashJoinExec should be non-empty".to_string(),
            ));
        }

        check_join_is_valid(&left_schema, &right_schema, &on)?;

        let (schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);

        let random_state = RandomState::with_seeds(0, 0, 0, 0);

        Ok(HashJoinExec {
            left,
            right,
            on,
            join_type: *join_type,
            schema: Arc::new(schema),
            build_side: Arc::new(Mutex::new(None)),
            random_state,
            mode: partition_mode,
            metrics: ExecutionPlanMetricsSet::new(),
            column_indices,
            null_equals_null: *null_equals_null,
        })
    }

    /// left (build) side which gets hashed
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(Column, Column)] {
        &self.on
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &PartitionMode {
        &self.mode
    }

    /// Get null_equals_null
    pub fn null_equals_null(&self) -> &bool {
        &self.null_equals_null
    }
}

#[async_trait]
impl ExecutionPlan for HashJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HashJoinExec::try_new(
            children[0].clone(),
            children[1].clone(),
            self.on.clone(),
            &self.join_type,
            self.mode,
            &self.null_equals_null,
        )?))
    }

    fn output_partitioning(&self) -> Partitioning {
        self.right.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[PhysicalSortExpr]> {
        None
    }

    fn relies_on_input_order(&self) -> bool {
        false
    }

    async fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let on_left = self.on.iter().map(|on| on.0.clone()).collect::<Vec<_>>();
        // we only want to compute the build side once for PartitionMode::CollectLeft
        let left_data = {
            match self.mode {
                PartitionMode::CollectLeft => {
                    let mut build_side = self.build_side.lock().await;

                    match build_side.as_ref() {
                        Some(stream) => stream.clone(),
                        None => {
                            let start = Instant::now();

                            // merge all left parts into a single stream
                            let merge = CoalescePartitionsExec::new(self.left.clone());
                            let stream = merge.execute(0, context.clone()).await?;

                            // This operation performs 2 steps at once:
                            // 1. creates a [JoinHashMap] of all batches from the stream
                            // 2. stores the batches in a vector.
                            let initial = (0, Vec::new());
                            let (num_rows, batches) = stream
                                .try_fold(initial, |mut acc, batch| async {
                                    acc.0 += batch.num_rows();
                                    acc.1.push(batch);
                                    Ok(acc)
                                })
                                .await?;
                            let mut hashmap =
                                JoinHashMap(RawTable::with_capacity(num_rows));
                            let mut hashes_buffer = Vec::new();
                            let mut offset = 0;
                            for batch in batches.iter() {
                                hashes_buffer.clear();
                                hashes_buffer.resize(batch.num_rows(), 0);
                                update_hash(
                                    &on_left,
                                    batch,
                                    &mut hashmap,
                                    offset,
                                    &self.random_state,
                                    &mut hashes_buffer,
                                )?;
                                offset += batch.num_rows();
                            }
                            // Merge all batches into a single batch, so we
                            // can directly index into the arrays
                            let single_batch =
                                concat_batches(&self.left.schema(), &batches, num_rows)?;

                            let left_side = Arc::new((hashmap, single_batch));

                            *build_side = Some(left_side.clone());

                            debug!(
                                "Built build-side of hash join containing {} rows in {} ms",
                                num_rows,
                                start.elapsed().as_millis()
                            );

                            left_side
                        }
                    }
                }
                PartitionMode::Partitioned => {
                    let start = Instant::now();

                    // Load 1 partition of left side in memory
                    let stream = self.left.execute(partition, context.clone()).await?;

                    // This operation performs 2 steps at once:
                    // 1. creates a [JoinHashMap] of all batches from the stream
                    // 2. stores the batches in a vector.
                    let initial = (0, Vec::new());
                    let (num_rows, batches) = stream
                        .try_fold(initial, |mut acc, batch| async {
                            acc.0 += batch.num_rows();
                            acc.1.push(batch);
                            Ok(acc)
                        })
                        .await?;
                    let mut hashmap = JoinHashMap(RawTable::with_capacity(num_rows));
                    let mut hashes_buffer = Vec::new();
                    let mut offset = 0;
                    for batch in batches.iter() {
                        hashes_buffer.clear();
                        hashes_buffer.resize(batch.num_rows(), 0);
                        update_hash(
                            &on_left,
                            batch,
                            &mut hashmap,
                            offset,
                            &self.random_state,
                            &mut hashes_buffer,
                        )?;
                        offset += batch.num_rows();
                    }
                    // Merge all batches into a single batch, so we
                    // can directly index into the arrays
                    let single_batch =
                        concat_batches(&self.left.schema(), &batches, num_rows)?;

                    let left_side = Arc::new((hashmap, single_batch));

                    debug!(
                        "Built build-side {} of hash join containing {} rows in {} ms",
                        partition,
                        num_rows,
                        start.elapsed().as_millis()
                    );

                    left_side
                }
            }
        };

        // we have the batches and the hash map with their keys. We can how create a stream
        // over the right that uses this information to issue new batches.

        let right_stream = self.right.execute(partition, context.clone()).await?;
        let on_right = self.on.iter().map(|on| on.1.clone()).collect::<Vec<_>>();

        let num_rows = left_data.1.num_rows();
        let visited_left_side = match self.join_type {
            JoinType::Left | JoinType::Full | JoinType::Semi | JoinType::Anti => {
                let mut buffer = BooleanBufferBuilder::new(num_rows);

                buffer.append_n(num_rows, false);

                buffer
            }
            JoinType::Inner | JoinType::Right => BooleanBufferBuilder::new(0),
        };
        Ok(Box::pin(HashJoinStream::try_new(
            self.schema.clone(),
            on_left,
            on_right,
            self.join_type,
            left_data,
            right_stream,
            self.column_indices.clone(),
            self.random_state.clone(),
            visited_left_side,
            HashJoinMetrics::new(partition, &self.metrics),
            self.null_equals_null,
        )?))
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(
                    f,
                    "HashJoinExec: mode={:?}, join_type={:?}, on={:?}",
                    self.mode, self.join_type, self.on
                )
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Statistics {
        // TODO stats: it is not possible in general to know the output size of joins
        // There are some special cases though, for example:
        // - `A LEFT JOIN B ON A.col=B.col` with `COUNT_DISTINCT(B.col)=COUNT(B.col)`
        Statistics::default()
    }
}

/// Updates `hash` with new entries from [RecordBatch] evaluated against the expressions `on`,
/// assuming that the [RecordBatch] corresponds to the `index`th
fn update_hash(
    on: &[Column],
    batch: &RecordBatch,
    hash_map: &mut JoinHashMap,
    offset: usize,
    random_state: &RandomState,
    hashes_buffer: &mut Vec<u64>,
) -> Result<()> {
    // evaluate the keys
    let keys_values = on
        .iter()
        .map(|c| Ok(c.evaluate(batch)?.into_array(batch.num_rows())))
        .collect::<Result<Vec<_>>>()?;

    // calculate the hash values
    let hash_values = create_hashes(&keys_values, random_state, hashes_buffer)?;

    // insert hashes to key of the hashmap
    for (row, hash_value) in hash_values.iter().enumerate() {
        let item = hash_map
            .0
            .get_mut(*hash_value, |(hash, _)| *hash_value == *hash);
        if let Some((_, indices)) = item {
            indices.push((row + offset) as u64);
        } else {
            hash_map.0.insert(
                *hash_value,
                (*hash_value, smallvec![(row + offset) as u64]),
                |(hash, _)| *hash,
            );
        }
    }
    Ok(())
}
//During matching 13*N bytes will be taken up by various buffers, 8 bytes for N u64, 4 bytes from N u32, and 1 byte from N bools. 
//Unlikely to be evicted from cache because every op will use tcolumn comparison will use them. Typical L1 cache size is 64KB
//Want to leave significant portion of L1 cache open for column data to avoid evicting match and index buffers.
// N | Est L1 % utilization 
// 64 | 1.2 %
// 256 | 5%
// 1024 | 20%
const HASH_BUFFER_SIZE: usize = 256;
/// A stream that issues [RecordBatch]es as they arrive from the right  of the join.
struct HashJoinStream {
    //Describes what kind of join is being done on what columns
    join_data: JoinData,
    /// right side of the join
    right: SendableRecordBatchStream,
    /// Keeps track of the left side rows whether they are visited
    visited_left_side: BooleanBufferBuilder,
    /// There is nothing to process anymore and left side is processed in case of left join
    is_exhausted: bool,
    /// Metrics
    join_metrics: HashJoinMetrics,
    join_buffers: Box<HashJoinBuffers<HASH_BUFFER_SIZE>>,
}

#[allow(clippy::too_many_arguments)]
impl HashJoinStream {
    fn try_new(
        schema: Arc<Schema>,
        on_left: Vec<Column>,
        on_right: Vec<Column>,
        join_type: JoinType,
        left_data: JoinLeftData,
        right: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        random_state: RandomState,
        visited_left_side: BooleanBufferBuilder,
        join_metrics: HashJoinMetrics,
        null_equals_null: bool,
    ) -> Result<Self> {
        Ok(HashJoinStream {
            join_data: JoinData::try_new(
                schema,
                on_left,
                on_right,
                join_type,
                left_data,
                random_state,
                column_indices,
                null_equals_null,
            )?,
            join_buffers: HashJoinBuffers{
                hash_buffer: [0; HASH_BUFFER_SIZE],
                left_index_buffer: [0; HASH_BUFFER_SIZE],
                right_index_buffer: [0; HASH_BUFFER_SIZE], 
                match_buffer: [false; HASH_BUFFER_SIZE],
            }.into(),
            right,
            visited_left_side,
            is_exhausted: false,
            join_metrics,
        })
    }
}

impl RecordBatchStream for HashJoinStream {
    fn schema(&self) -> SchemaRef {
        self.join_data.schema.clone()
    }
}

/// Returns a new [RecordBatch] by combining the `left` and `right` according to `indices`.
/// The resulting batch has [Schema] `schema`.
/// # Error
/// This function errors when:
/// *
fn build_batch_from_indices(
    join_data: &JoinData,
    right: &RecordBatch,
    left_indices: UInt64Array,
    right_indices: UInt32Array,
) -> ArrowResult<(RecordBatch, UInt64Array)> {
    // build the columns of the new [RecordBatch]:
    // 1. pick whether the column is from the left or right
    // 2. based on the pick, `take` items from the different RecordBatches
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(join_data.schema.fields().len());

    for column_index in &join_data.column_indices {
        let array = match column_index.side {
            JoinSide::Left => {
                let array = join_data.left_data.1.column(column_index.index);
                if array.is_empty() || left_indices.null_count() == left_indices.len() {
                    // Outer join would generate a null index when finding no match at our side.
                    // Therefore, it's possible we are empty but need to populate an n-length null array,
                    // where n is the length of the index array.
                    assert_eq!(left_indices.null_count(), left_indices.len());
                    new_null_array(array.data_type(), left_indices.len())
                } else {
                    compute::take(array.as_ref(), &left_indices, None)?
                }
            }
            JoinSide::Right => {
                let array = right.column(column_index.index);
                if array.is_empty() || right_indices.null_count() == right_indices.len() {
                    assert_eq!(right_indices.null_count(), right_indices.len());
                    new_null_array(array.data_type(), right_indices.len())
                } else {
                    compute::take(array.as_ref(), &right_indices, None)?
                }
            }
        };
        columns.push(array);
    }
    RecordBatch::try_new(Arc::new(join_data.schema.as_ref().clone()), columns).map(|x| (x, left_indices))
}





 struct HashJoinBuffers<const N: usize>{
    //Buffer that stores the hashes of the rows being considered
    hash_buffer: [u64; N],
    //Buffer that stores the index mapping between the hash side and probe side of the join
    //64 bit 12*N bytes taken up
    left_index_buffer:[u64; N],
    right_index_buffer: [u32; N],
    //N bytes taken up
    //Buffer that stores the row equalities of the mappings stored in the index_buffer. match_buffer[0]=true means that the rows are equal for the mapping at index_buffer[0] 
    match_buffer: [bool; N],   
}

impl<const N: usize> HashJoinBuffers<N>{
    #[inline]
    unsafe fn get_matching_data_unchecked(&self, idx: usize)->(bool, u64, u32){
        let is_match = *self.match_buffer.get_unchecked(idx);
        let lidx = *self.left_index_buffer.get_unchecked(idx);
        let ridx = *self.right_index_buffer.get_unchecked(idx);
        (is_match, lidx, ridx)
    }
    #[inline]
    fn get_matching_data(&self, idx: usize)->(bool, u64, u32){
        let is_match = self.match_buffer[idx];
        let lidx = self.left_index_buffer[idx];
        let ridx = self.right_index_buffer[idx];
        (is_match, lidx, ridx)
    }
    #[inline]
    fn set_mapping_data(&mut self, idx: usize, lidx: u64, ridx: u32){
        self.left_index_buffer[idx] = lidx;
        self.right_index_buffer[idx]=ridx;
    }
}










/// returns a vector with (index from left, index from right).
/// The size of this vector corresponds to the total size of a joined batch
// For a join on column A:
// left       right
//     batch 1
// A B         A D
// ---------------
// 1 a         3 6
// 2 b         1 2
// 3 c         2 4
//     batch 2
// A B         A D
// ---------------
// 1 a         5 10
// 2 b         2 2
// 4 d         1 1
// indices (batch, batch_row)
// left       right
// (0, 2)     (0, 0)
// (0, 0)     (0, 1)
// (0, 1)     (0, 2)
// (1, 0)     (0, 1)
// (1, 1)     (0, 2)
// (0, 1)     (1, 1)
// (0, 0)     (1, 2)
// (1, 1)     (1, 1)
// (1, 0)     (1, 2)

fn build_join_indexes_left<const N: usize>(
    buffers: &mut HashJoinBuffers<N>, 
    left_data:&JoinLeftData, 
    right: &RecordBatch,
    left_on: &[Column],
    right_on: &[Column],
    random_state: &RandomState,
    null_equals_null: bool
)->Result<(UInt64Array, UInt32Array)>{
    let mut left_indices = UInt64Builder::new(0);
    let mut right_indices = UInt32Builder::new(0);
    /* 
    // First visit all of the rows
    for (row, hash_value) in hash_values.iter().enumerate() {
        if let Some((_, indices)) =
            left.0.get(*hash_value, |(hash, _)| *hash_value == *hash)
        {
            for &i in indices {
                // Collision check
                if equal_rows(
                    i as usize,
                    row,
                    &left_join_values,
                    &keys_values,
                    *null_equals_null,
                )? {
                    left_indices.append_value(i)?;
                    right_indices.append_value(row as u32)?;
                }
            }
        };
    }
    */
    Ok((left_indices.finish(), right_indices.finish()))
}

impl<const N:usize> HashJoinBuffers<N>{
    fn build_mappings_full_or_right(&self,left_indices: &mut UInt64Builder, right_indices: &mut UInt32Builder, num_to_process: usize)->ArrowResult<()>{
        let mut curr_ridx = u32::MAX;
        let mut matched = true;
        for i in 0..num_to_process{
            let (is_match, lidx, ridx) = unsafe{self.get_matching_data_unchecked(i)};
            if curr_ridx != ridx{
                if !matched{
                    left_indices.append_null()?;
                    right_indices.append_value(ridx)?;
                }
                curr_ridx = ridx;
                matched = false;
            }
            if is_match{
                left_indices.append_value(lidx)?;
                right_indices.append_value(ridx)?;
                matched = true;
            }
        }
        Ok(())
    }

    fn build_mappings_inner_or_left(&self, left_indices: &mut BufferBuilder<u64>, right_indices: &mut BufferBuilder<u32>, num_to_process: usize){
        assert!(num_to_process <= N);
        for i in 0..num_to_process{
            let (is_match, lidx, ridx) = unsafe{self.get_matching_data_unchecked(i)};
            if is_match{
                left_indices.append(lidx);
                right_indices.append(ridx as u32);
            }
        }
    }

    fn build_mappings_semi_or_anti(&self, left_indices: &mut BufferBuilder<u64>, num_to_process: usize){
        assert!(num_to_process <= N);
        for i in 0..num_to_process{
            let (is_match, lidx, ridx) = unsafe{self.get_matching_data_unchecked(i)};
            if is_match{
                left_indices.append(lidx);
            }
        }
    }

}



fn build_join_indexes_full_or_right<const N: usize>(
    left_join_values: &[Arc<dyn Array>],
    left_hashmap: &JoinHashMap,
    keys_values: &[Arc<dyn Array>],
    random_state: &RandomState,
    buffers: &mut HashJoinBuffers<N>,
    null_equals_null: bool
)->ArrowResult<(UInt64Array, UInt32Array)>{
    
    //This is annoying can't use u32 as array size so have to dynamically assert N is less than u32::MAX. Probably fine since using u32::MAX for the static buffer sizes 
    //Would thrash the cpu cache like nothing else as it would take 10.5 MiB in memory
    assert!(N <= u32::MAX as usize);
    let mut left_indices = UInt64Array::builder(0);
    let mut right_indices = UInt32Array::builder(0);
    let mut start_idx:u32 = 0;
    let mut buffer_idx = 0;
    let right_row_count = keys_values[0].len().try_into().expect("Number of rows for record batch should be less than u32::MAX, which is 4294967295, rows");
    

    while start_idx < right_row_count{
        let num_to_process = (right_row_count - start_idx).min(N);
        //Fill the buffer with num_to_process hashes
        create_hashes_new(&keys_values, random_state, &mut buffers.hash_buffer, start_idx, num_to_process)?;
    
        for hash_buf_idx in 0..(N as u32){
            let row = start_idx + hash_buf_idx;
            let hash_value = buffers.hash_buffer[hash_buf_idx as usize];
            if let Some((_, indices)) = left_hashmap.0.get(hash_value, |(hash, _)| hash_value == *hash){
                for i in indices{
                    buffers.set_mapping_data(buffer_idx, *i, row as u32)
                    buffers.index_buffer[buffer_idx]=(*i, row);
                    buffer_idx +=1;
                    if buffer_idx == N{
                        check_equal_rows_vectorized_new(&mut buffers.match_buffer, &left_join_values, keys_values, &buffers.index_buffer, N, null_equals_null)?;
                        buffers.build_mappings_full_or_right(&mut left_indices, &mut right_indices, num_to_process)?;
                        buffer_idx = 0;
                    }
                }
            }else{
                left_indices.append_null()?;
                right_indices.append_value(row as u32)?;
            }
        }
        start_idx += num_to_process;
    }
    if buffer_idx != 0{
        check_equal_rows_vectorized_new(&mut buffers.match_buffer, &left_join_values, keys_values, &buffers.index_buffer, buffer_idx, null_equals_null)?;
        buffers.build_mappings_full_or_right(&mut left_indices, &mut right_indices, buffer_idx)?;
    }
    Ok((left_indices.finish(), right_indices.finish())) 
}

fn build_join_indexes_inner_or_left<const N: usize>(
    left_join_values: &[Arc<dyn Array>],
    left_hashmap: &JoinHashMap,
    keys_values: &[Arc<dyn Array>],
    random_state: &RandomState,
    buffers: &mut HashJoinBuffers<N>,
    null_equals_null: bool
)->Result<(UInt64Array, UInt32Array)>{

    // Using a buffer builder to avoid slower normal builder
    let mut left_indices = UInt64BufferBuilder::new(0);
    let mut right_indices = UInt32BufferBuilder::new(0);
    let mut right_idx = 0;
    let mut buffer_idx = 0;
    let right_row_count = keys_values[0].len();
    while right_idx < right_row_count{
        let num_to_process = (right_row_count - right_idx).min(N);
        //Fill the buffer with num_to_process hashes
        create_hashes_new(&keys_values, random_state, &mut buffers.hash_buffer, right_idx, num_to_process)?;

        
        //For each hash get matching row mappings. When buffer is full then 
        for hash_idx in 0..num_to_process{
            let row = right_idx + hash_idx;
            let hash_value = buffers.hash_buffer[hash_idx];
            if let Some((_, indices)) = left_hashmap.0.get(hash_value, |(hash, _)| hash_value == *hash){
                for i in indices{
                    buffers.index_buffer[buffer_idx]=(*i, row);
                    buffer_idx +=1;
                    if buffer_idx == N{
                        check_equal_rows_vectorized_new(&mut buffers.match_buffer, left_join_values, keys_values, &buffers.index_buffer, N, null_equals_null)?;
                        buffers.build_mappings_inner_or_left(&mut left_indices, &mut right_indices, N);
                        buffer_idx = 0;
                    }
                }
            }
        }
        right_idx += num_to_process;
    }
    if buffer_idx != 0{
        check_equal_rows_vectorized_new(&mut buffers.match_buffer, left_join_values, keys_values, &buffers.index_buffer, buffer_idx, null_equals_null)?;
        buffers.build_mappings_inner_or_left(&mut left_indices, &mut right_indices, buffer_idx);
    }

    let left = ArrayData::builder(DataType::UInt64)
        .len(left_indices.len())
        .add_buffer(left_indices.finish())
        .build()
        .unwrap();
    let right = ArrayData::builder(DataType::UInt32)
        .len(right_indices.len())
        .add_buffer(right_indices.finish())
        .build()
        .unwrap();

    Ok((
        PrimitiveArray::<UInt64Type>::from(left),
        PrimitiveArray::<UInt32Type>::from(right),
    ))
}


fn build_join_indexes_semi_or_anti<const N: usize>(
    left_join_values: &[Arc<dyn Array>],
    left_hashmap: &JoinHashMap,
    keys_values: &[Arc<dyn Array>],
    random_state: &RandomState,
    buffers: &mut HashJoinBuffers<N>,
    null_equals_null: bool
)->Result<UInt64Array>{

    // Using a buffer builder to avoid slower normal builder
    let mut left_indices = UInt64BufferBuilder::new(0);
    let mut right_idx = 0;
    let mut idx_buffer_idx = 0;
    let right_row_count = keys_values[0].len();
    while right_idx < right_row_count{
        let num_to_process = (right_row_count - right_idx).min(N);
        //Fill the buffer with num_to_process hashes
        create_hashes_new(&keys_values, random_state, &mut buffers.hash_buffer, right_idx, num_to_process)?;

        
        //For each hash get matching row mappings. When buffer is full then 
        for hash_idx in 0..num_to_process{
            let row = right_idx + hash_idx;
            let hash_value = buffers.hash_buffer[hash_idx];
            if let Some((_, indices)) = left_hashmap.0.get(hash_value, |(hash, _)| hash_value == *hash){
                for i in indices{
                    buffers.index_buffer[idx_buffer_idx]=(*i, row);
                    idx_buffer_idx +=1;
                    if idx_buffer_idx == N{
                        check_equal_rows_vectorized_new(&mut buffers.match_buffer, left_join_values, keys_values, &buffers.index_buffer, N, null_equals_null)?;
                        buffers.build_mappings_semi_or_anti(&mut left_indices, N);
                        idx_buffer_idx = 0;
                    }
                }
            }
        }
        right_idx += num_to_process;
    }
    if idx_buffer_idx != 0{
        check_equal_rows_vectorized_new(&mut buffers.match_buffer, left_join_values, keys_values, &buffers.index_buffer, idx_buffer_idx, null_equals_null)?;
        buffers.build_mappings_semi_or_anti(&mut left_indices, idx_buffer_idx);
    }

    let left = ArrayData::builder(DataType::UInt64)
        .len(left_indices.len())
        .add_buffer(left_indices.finish())
        .build()
        .unwrap();


    Ok(
        PrimitiveArray::<UInt64Type>::from(left),
    )
}





fn build_join_indexes(
    match_buffer: &mut Vec<bool>,
    join_index_buffer: &mut Vec<(u64, usize)>,
    left_data: &JoinLeftData,
    right: &RecordBatch,
    join_type: JoinType,
    left_on: &[Column],
    right_on: &[Column],
    random_state: &RandomState,
    null_equals_null: &bool,
) -> Result<(UInt64Array, UInt32Array)> {
    let keys_values = right_on
        .iter()
        .map(|c| Ok(c.evaluate(right)?.into_array(right.num_rows())))
        .collect::<Result<Vec<_>>>()?;
    let left_join_values = left_on
        .iter()
        .map(|c| Ok(c.evaluate(&left_data.1)?.into_array(left_data.1.num_rows())))
        .collect::<Result<Vec<_>>>()?;
    let hashes_buffer = &mut vec![0; keys_values[0].len()];
    let hash_values = create_hashes(&keys_values, random_state, hashes_buffer)?;
    let left = &left_data.0;
    join_index_buffer.clear();
    match_buffer.clear();
    match join_type {
        JoinType::Inner | JoinType::Semi | JoinType::Anti => {
            // Using a buffer builder to avoid slower normal builder
            let mut left_indices = UInt64BufferBuilder::new(0);
            let mut right_indices = UInt32BufferBuilder::new(0);
            // Visit all of the right rows
            for (row, hash_value) in hash_values.iter().enumerate() {
                // Get the hash and find it in the build index

                // For every item on the left and right we check if it matches
                // This possibly contains rows with hash collisions,
                // So we have to check here whether rows are equal or not
                if let Some((_, indices)) =
                    left.0.get(*hash_value, |(hash, _)| *hash_value == *hash)
                {
                    for &i in indices {
                        // Check hash collisions
                        if equal_rows(
                            i as usize,
                            row,
                            &left_join_values,
                            &keys_values,
                            *null_equals_null,
                        )? {
                            left_indices.append(i);
                            right_indices.append(row as u32);
                        }
                    }
                }
            }
            let left = ArrayData::builder(DataType::UInt64)
                .len(left_indices.len())
                .add_buffer(left_indices.finish())
                .build()
                .unwrap();
            let right = ArrayData::builder(DataType::UInt32)
                .len(right_indices.len())
                .add_buffer(right_indices.finish())
                .build()
                .unwrap();
            
            Ok((
                PrimitiveArray::<UInt64Type>::from(left),
                PrimitiveArray::<UInt32Type>::from(right),
            ))
        }
        JoinType::Left => {
            let mut left_indices = UInt64Builder::new(0);
            let mut right_indices = UInt32Builder::new(0);

            // First visit all of the rows
            for (row, hash_value) in hash_values.iter().enumerate() {
                if let Some((_, indices)) =
                    left.0.get(*hash_value, |(hash, _)| *hash_value == *hash)
                {
                    for &i in indices {
                        // Collision check
                        if equal_rows(
                            i as usize,
                            row,
                            &left_join_values,
                            &keys_values,
                            *null_equals_null,
                        )? {
                            left_indices.append_value(i)?;
                            right_indices.append_value(row as u32)?;
                        }
                    }
                };
            }
            Ok((left_indices.finish(), right_indices.finish()))
        }
        JoinType::Right | JoinType::Full => {
            let mut left_indices = UInt64Builder::new(0);
            let mut right_indices = UInt32Builder::new(0);

            for (row, hash_value) in hash_values.iter().enumerate() {
                match left.0.get(*hash_value, |(hash, _)| *hash_value == *hash) {
                    Some((_, indices)) => {
                        let mut no_match = true;
                        for &i in indices {
                            if equal_rows(
                                i as usize,
                                row,
                                &left_join_values,
                                &keys_values,
                                *null_equals_null,
                            )? {
                                left_indices.append_value(i)?;
                                right_indices.append_value(row as u32)?;
                                no_match = false;
                            }
                        }
                        // If no rows matched left, still must keep the right
                        // with all nulls for left
                        if no_match {
                            left_indices.append_null()?;
                            right_indices.append_value(row as u32)?;
                        }
                    }
                    None => {
                        // when no match, add the row with None for the left side
                        left_indices.append_null()?;
                        right_indices.append_value(row as u32)?;
                    }
                }
            }
            Ok((left_indices.finish(), right_indices.finish()))
        }
    }
}



fn equal_rows_check<LARR: 'static, RARR: 'static, LeftNull: Fn(&LARR, usize)->bool + 'static, RightNull: Fn(&RARR, usize)->bool + 'static, IsEq: Fn(&LARR, usize, &RARR, usize)->bool>(matches: &mut [bool], first_column: bool, left_indexes: &[u64;N], right_indexes: &[u32; N], indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool, left_is_null: LeftNull, right_is_null: RightNull, is_equal: IsEq)->bool{
    assert!(matches.len() == indexes.len());
    let left_array = l.as_any().downcast_ref::<LARR>().unwrap();
    let right_array = r.as_any().downcast_ref::<RARR>().unwrap();
    let mut at_least_one_match = false;
    

    if null_equals_null{
        for (is_match, (l, r)) in matches.iter_mut().zip(indexes){
            let lidx = *l as usize;
            let ridx = *r;
            let left_is_null = left_is_null(left_array, lidx) as u8;
            let right_is_null = right_is_null(right_array, lidx) as u8;
            let values_eq = is_equal(left_array, lidx, right_array, ridx) as u8;
            let rows_eq = (left_is_null & right_is_null) | (((left_is_null | right_is_null)^0x1) & values_eq) != 0;
            *is_match = if first_column{
                rows_eq
            }else{
                rows_eq && *is_match
            };
            at_least_one_match = at_least_one_match || *is_match;

        }
    }else{
        for (is_match, (l, r)) in matches.iter_mut().zip(indexes){
            let lidx = *l as usize;
            let ridx = *r;
            let left_is_null = left_is_null(left_array, lidx) as u8;
            let right_is_null = right_is_null(right_array, lidx) as u8;
            let values_eq = is_equal(left_array, lidx, right_array, ridx) as u8;
            let rows_eq = (((left_is_null | right_is_null)^0x1) & values_eq) != 0;
            *is_match = if first_column{
                rows_eq
            }else{
                rows_eq && *is_match
            };
            at_least_one_match = at_least_one_match || *is_match; 
        }
    }
    at_least_one_match

}



fn check_equal_rows_vectorized_new<const N: usize>(
    matches: &mut [bool; N],
    left_arrays: &[ArrayRef],
    right_arrays: &[ArrayRef],
    left_indexes: &[u64; N],
    right_index: &[u32; N],
    num_to_process: usize,
    null_equals_null: bool,
) -> Result<()> {
    assert!(matches.len() == indexes.len());
    if num_to_process > N{
        return Err(DataFusionError::Internal(format!("Found process count {} out of range of {}", num_to_process, N)));
    }
    let mut  first_column = true;
    let mut at_least_one_match;
    for (l,r) in left_arrays.iter().zip(right_arrays.iter()){
        at_least_one_match = match l.data_type(){
            DataType::Null => {
                matches.iter_mut().for_each(|matches| *matches = true);
                true
            },
            DataType::Boolean => equal_rows_check_same_type::<BooleanArray, _, _>(matches, first_column, indexes, l,r, null_equals_null, |arr, idx| arr.is_null(idx), |larr, lidx, rarr, ridx| {
                larr.value(lidx) == rarr.value(ridx)
            }),
            DataType::Int8 => equal_rows_check_primitive::<Int8Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Int16 => equal_rows_check_primitive::<Int16Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Int32 => equal_rows_check_primitive::<Int32Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Int64 => equal_rows_check_primitive::<Int64Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::UInt8 => equal_rows_check_primitive::<UInt8Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::UInt16 => equal_rows_check_primitive::<UInt16Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::UInt32 => equal_rows_check_primitive::<UInt32Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::UInt64 => equal_rows_check_primitive::<UInt64Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Float32 => equal_rows_check_primitive::<Float32Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Float64 => equal_rows_check_primitive::<Float64Type>(matches, first_column, indexes, l,r, null_equals_null),
            DataType::Timestamp(time_unit, None) => match time_unit {
                TimeUnit::Second =>equal_rows_check_primitive::<TimestampSecondType>(matches, first_column, indexes, l,r,null_equals_null), 
                TimeUnit::Millisecond => equal_rows_check_primitive::<TimestampMillisecondType>(matches, first_column, indexes, l,r,null_equals_null), 
                TimeUnit::Microsecond => equal_rows_check_primitive::<TimestampMicrosecondType>(matches, first_column, indexes, l,r,null_equals_null),
                TimeUnit::Nanosecond =>equal_rows_check_primitive::<TimestampNanosecondType>(matches, first_column, indexes, l,r,null_equals_null),
            },
            DataType::Utf8 => equal_rows_check_utf8::<i32>(matches, first_column, indexes, l,r,null_equals_null),
            DataType::LargeUtf8 => equal_rows_check_utf8::<i64>(matches, first_column, indexes, l,r,null_equals_null),
            DataType::Dictionary(key_ty, value_ty) if value_ty.as_ref() == &DataType::Utf8=>{
                let right_key_ty = if let DataType::Dictionary(rkey_ty, lval_ty) = r.data_type(){
                    if lval_ty.as_ref() != &DataType::Utf8{
                        panic!("Found nont utf8 values");
                    }
                    rkey_ty.as_ref()
                }else{
                    panic!("Right was not dictionary type");
                };
                match (key_ty.as_ref(), right_key_ty){
                    (DataType::Int8, DataType::Int8) => equal_rows_utf8_dictionaries::<Int8Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null), 
                    (DataType::Int8, DataType::Int16) => equal_rows_utf8_dictionaries::<Int8Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int8, DataType::Int32) => equal_rows_utf8_dictionaries::<Int8Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int8, DataType::Int64) => equal_rows_utf8_dictionaries::<Int8Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int8, DataType::UInt8) => equal_rows_utf8_dictionaries::<Int8Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null), 
                    (DataType::Int8, DataType::UInt16) => equal_rows_utf8_dictionaries::<Int8Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int8, DataType::UInt32) => equal_rows_utf8_dictionaries::<Int8Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int8, DataType::UInt64) => equal_rows_utf8_dictionaries::<Int8Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),


                    (DataType::Int16, DataType::Int8) => equal_rows_utf8_dictionaries::<Int16Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null), 
                    (DataType::Int16, DataType::Int16) => equal_rows_utf8_dictionaries::<Int16Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int16, DataType::Int32) => equal_rows_utf8_dictionaries::<Int16Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int16, DataType::Int64) => equal_rows_utf8_dictionaries::<Int16Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int16, DataType::UInt8) => equal_rows_utf8_dictionaries::<Int16Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null), 
                    (DataType::Int16, DataType::UInt16) => equal_rows_utf8_dictionaries::<Int16Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int16, DataType::UInt32) => equal_rows_utf8_dictionaries::<Int16Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int16, DataType::UInt64) => equal_rows_utf8_dictionaries::<Int16Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    
                    (DataType::Int32, DataType::Int8)  => equal_rows_utf8_dictionaries::<Int32Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::Int16) => equal_rows_utf8_dictionaries::<Int32Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::Int32) => equal_rows_utf8_dictionaries::<Int32Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::Int64) => equal_rows_utf8_dictionaries::<Int32Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::UInt8)  => equal_rows_utf8_dictionaries::<Int32Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::UInt16) => equal_rows_utf8_dictionaries::<Int32Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::UInt32) => equal_rows_utf8_dictionaries::<Int32Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int32, DataType::UInt64) => equal_rows_utf8_dictionaries::<Int32Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),

                    (DataType::Int64, DataType::Int8) => equal_rows_utf8_dictionaries::<Int64Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::Int16)=> equal_rows_utf8_dictionaries::<Int64Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::Int32)=> equal_rows_utf8_dictionaries::<Int64Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::Int64)=> equal_rows_utf8_dictionaries::<Int64Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::UInt8) => equal_rows_utf8_dictionaries::<Int64Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::UInt16)=> equal_rows_utf8_dictionaries::<Int64Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::UInt32)=> equal_rows_utf8_dictionaries::<Int64Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::Int64, DataType::UInt64)=> equal_rows_utf8_dictionaries::<Int64Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),

                    (DataType::UInt8, DataType::Int8) => equal_rows_utf8_dictionaries::<UInt8Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::Int16)=> equal_rows_utf8_dictionaries::<UInt8Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::Int32)=> equal_rows_utf8_dictionaries::<UInt8Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::Int64)=> equal_rows_utf8_dictionaries::<UInt8Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::UInt8) => equal_rows_utf8_dictionaries::<UInt8Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::UInt16)=> equal_rows_utf8_dictionaries::<UInt8Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::UInt32)=> equal_rows_utf8_dictionaries::<UInt8Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt8, DataType::UInt64)=> equal_rows_utf8_dictionaries::<UInt8Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),

                    (DataType::UInt16, DataType::Int8) => equal_rows_utf8_dictionaries::<UInt16Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::Int16)=> equal_rows_utf8_dictionaries::<UInt16Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::Int32)=> equal_rows_utf8_dictionaries::<UInt16Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::Int64)=> equal_rows_utf8_dictionaries::<UInt16Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::UInt8) => equal_rows_utf8_dictionaries::<UInt16Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::UInt16)=> equal_rows_utf8_dictionaries::<UInt16Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::UInt32)=> equal_rows_utf8_dictionaries::<UInt16Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt16, DataType::UInt64)=> equal_rows_utf8_dictionaries::<UInt16Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),

                    (DataType::UInt32, DataType::Int8) => equal_rows_utf8_dictionaries::<UInt32Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::Int16)=> equal_rows_utf8_dictionaries::<UInt32Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::Int32)=> equal_rows_utf8_dictionaries::<UInt32Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::Int64)=> equal_rows_utf8_dictionaries::<UInt32Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::UInt8) => equal_rows_utf8_dictionaries::<UInt32Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::UInt16)=> equal_rows_utf8_dictionaries::<UInt32Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::UInt32)=> equal_rows_utf8_dictionaries::<UInt32Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt32, DataType::UInt64)=> equal_rows_utf8_dictionaries::<UInt32Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),

                    (DataType::UInt64, DataType::Int8) => equal_rows_utf8_dictionaries::<UInt64Type,Int8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::Int16)=> equal_rows_utf8_dictionaries::<UInt64Type,Int16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::Int32)=> equal_rows_utf8_dictionaries::<UInt64Type,Int32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::Int64)=> equal_rows_utf8_dictionaries::<UInt64Type,Int64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::UInt8) => equal_rows_utf8_dictionaries::<UInt64Type,UInt8Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::UInt16)=> equal_rows_utf8_dictionaries::<UInt64Type,UInt16Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::UInt32)=> equal_rows_utf8_dictionaries::<UInt64Type,UInt32Type>(matches, first_column, indexes, l,r,null_equals_null),
                    (DataType::UInt64, DataType::UInt64)=> equal_rows_utf8_dictionaries::<UInt64Type,UInt64Type>(matches, first_column, indexes, l,r,null_equals_null),
                    _=> panic!(),
                }
            }
            _ => {
                // This is internal because we should have caught this before.
                return Err(DataFusionError::Internal(
                    "Unsupported data type in hasher".to_string(),
                ));
            }
        };
        if !at_least_one_match{
            break;
        }
        first_column = false;
    }
    Ok(())
}



fn equal_rows_check_same_type<ARR: 'static, IsNull: Fn(&ARR, usize)->bool + Copy + 'static, IsEq: Fn(&ARR, usize, &ARR, usize)->bool>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool, is_null: IsNull, is_equal: IsEq)->bool{
    equal_rows_check::<ARR, ARR,_, _, _>(matches, first_column, indexes, l,r, null_equals_null, is_null, is_null, is_equal)
}


fn equal_rows_check_primitive<T: ArrowPrimitiveType>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check_same_type::<PrimitiveArray<T>, _,_>(matches, first_column, indexes, l, r, null_equals_null, |arr, idx| arr.is_null(idx), |larr, lidx, rarr, ridx| larr.value(lidx) == rarr.value(ridx))
}

fn equal_rows_check_utf8<O: StringOffsetSizeTrait>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check_same_type::<GenericStringArray<O>, _, _>(matches, first_column, indexes, l, r, null_equals_null, |arr, idx| arr.is_null(idx), |larr,lidx, rarr, ridx| larr.value(lidx) == rarr.value(ridx))
}



fn equal_rows_check_dict_prim<K: ArrowDictionaryKeyType, T: ArrowPrimitiveType>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check::<DictionaryArray<K>, PrimitiveArray<T>, _,_,_>(matches, first_column, indexes, l, r, null_equals_null, 
        |larr, idx| larr.is_null(idx), |larr, idx| larr.is_null(idx),
        |larr: &DictionaryArray<K>, lidx, rarr: &PrimitiveArray<T>, ridx|->bool{
            let key = larr.keys().value(lidx).to_usize().unwrap();
            let values = larr.values().as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
            let lval = values.value(key);
            let rval = rarr.value(ridx);
            lval == rval 
        })
}

fn equal_rows_check_prim_dict<T: ArrowPrimitiveType, K: ArrowDictionaryKeyType>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check::< PrimitiveArray<T>,DictionaryArray<K>, _,_,_>(matches, first_column, indexes, l, r, null_equals_null, 
        |larr, idx| larr.is_null(idx), |larr, idx| larr.is_null(idx),
        |larr, lidx, rarr, ridx|->bool{
            let lval = larr.value(ridx);
            let key = rarr.keys().value(ridx).to_usize().unwrap();
            let values = rarr.values().as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
            let rval = values.value(key);
            lval == rval 
        })
}

fn equal_rows_utf8_dictionaries<LK: ArrowDictionaryKeyType, RK: ArrowDictionaryKeyType>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check::<DictionaryArray<LK>, DictionaryArray<RK>, _, _, _>(
        matches, first_column,
        indexes, l,r,null_equals_null,
        |larr, idx| larr.is_null(idx),
        |rarr, idx| rarr.is_null(idx),
        |larr, lidx, rarr, ridx| {
            let lkey = larr.keys().value(lidx).to_usize().unwrap();
            let lvalues = larr.values().as_any().downcast_ref::<GenericStringArray<i32>>().unwrap();
            let lval = lvalues.value(lkey);

            let rkey = rarr.keys().value(ridx).to_usize().unwrap();
            let rvalues = rarr.values().as_any().downcast_ref::<GenericStringArray<i32>>().unwrap();
            let rval = rvalues.value(rkey);
            lval == rval 
        }
    )
}

fn equal_rows_check_dict_utf8<K: ArrowDictionaryKeyType, O: StringOffsetSizeTrait>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check::<DictionaryArray<K>, GenericStringArray<O>, _, _, _>(
        matches, first_column,
        indexes, l,r,null_equals_null,
        |larr, idx| larr.is_null(idx),
        |rarr, idx| rarr.is_null(idx),
        |larr, lidx, rarr, ridx| {
            let key = larr.keys().value(lidx).to_usize().unwrap();
            let values = larr.values().as_any().downcast_ref::<GenericStringArray<O>>().unwrap();
            let lval = values.value(key);
            let rval = rarr.value(ridx);
            lval == rval 
        }
    )
}


fn equal_rows_check_utf8_dict<K: ArrowDictionaryKeyType, O: StringOffsetSizeTrait>(matches: &mut [bool], first_column: bool, indexes: &[(u64, usize)], l: &dyn Array, r: &dyn Array, null_equals_null: bool)->bool{
    equal_rows_check::<GenericStringArray<O>, DictionaryArray<K>, _, _,_>(
        matches, first_column,
        indexes, l,r,null_equals_null,
        |larr, idx| larr.is_null(idx),
        |rarr, idx| rarr.is_null(idx),
        |larr, lidx, rarr, ridx| {
            let lval = larr.value(lidx);

            let key = rarr.keys().value(ridx).to_usize().unwrap();
            let values = rarr.values().as_any().downcast_ref::<GenericStringArray<O>>().unwrap();
            let rval = values.value(key);
            lval == rval 
        }
    )
}


macro_rules! equal_rows_elem {
    ($array_type:ident, $l: ident, $r: ident, $left: ident, $right: ident, $null_equals_null: ident) => {{
        let left_array = $l.as_any().downcast_ref::<$array_type>().unwrap();
        let right_array = $r.as_any().downcast_ref::<$array_type>().unwrap();

        match (left_array.is_null($left), right_array.is_null($right)) {
            (false, false) => left_array.value($left) == right_array.value($right),
            (true, true) => $null_equals_null,
            _ => false,
        }
    }};
}

/// Left and right row have equal values
fn equal_rows(
    left: usize,
    right: usize,
    left_arrays: &[ArrayRef],
    right_arrays: &[ArrayRef],
    null_equals_null: bool,
) -> Result<bool> {
    let mut err = None;
    let res = left_arrays
        .iter()
        .zip(right_arrays)
        .all(|(l, r)| match l.data_type() {
            DataType::Null => true,
            DataType::Boolean => {
                equal_rows_elem!(BooleanArray, l, r, left, right, null_equals_null)
            }
            DataType::Int8 => {
                equal_rows_elem!(Int8Array, l, r, left, right, null_equals_null)
            }
            DataType::Int16 => {
                equal_rows_elem!(Int16Array, l, r, left, right, null_equals_null)
            }
            DataType::Int32 => {
                equal_rows_elem!(Int32Array, l, r, left, right, null_equals_null)
            }
            DataType::Int64 => {
                equal_rows_elem!(Int64Array, l, r, left, right, null_equals_null)
            }
            DataType::UInt8 => {
                equal_rows_elem!(UInt8Array, l, r, left, right, null_equals_null)
            }
            DataType::UInt16 => {
                equal_rows_elem!(UInt16Array, l, r, left, right, null_equals_null)
            }
            DataType::UInt32 => {
                equal_rows_elem!(UInt32Array, l, r, left, right, null_equals_null)
            }
            DataType::UInt64 => {
                equal_rows_elem!(UInt64Array, l, r, left, right, null_equals_null)
            }
            DataType::Float32 => {
                equal_rows_elem!(Float32Array, l, r, left, right, null_equals_null)
            }
            DataType::Float64 => {
                equal_rows_elem!(Float64Array, l, r, left, right, null_equals_null)
            }
            DataType::Timestamp(time_unit, None) => match time_unit {
                TimeUnit::Second => {
                    equal_rows_elem!(
                        TimestampSecondArray,
                        l,
                        r,
                        left,
                        right,
                        null_equals_null
                    )
                }
                TimeUnit::Millisecond => {
                    equal_rows_elem!(
                        TimestampMillisecondArray,
                        l,
                        r,
                        left,
                        right,
                        null_equals_null
                    )
                }
                TimeUnit::Microsecond => {
                    equal_rows_elem!(
                        TimestampMicrosecondArray,
                        l,
                        r,
                        left,
                        right,
                        null_equals_null
                    )
                }
                TimeUnit::Nanosecond => {
                    equal_rows_elem!(
                        TimestampNanosecondArray,
                        l,
                        r,
                        left,
                        right,
                        null_equals_null
                    )
                }
            },
            DataType::Utf8 => {
                equal_rows_elem!(StringArray, l, r, left, right, null_equals_null)
            }
            DataType::LargeUtf8 => {
                equal_rows_elem!(LargeStringArray, l, r, left, right, null_equals_null)
            }
            _ => {
                // This is internal because we should have caught this before.
                err = Some(Err(DataFusionError::Internal(
                    "Unsupported data type in hasher".to_string(),
                )));
                false
            }
        });

    err.unwrap_or(Ok(res))
}

// Produces a batch for left-side rows that have/have not been matched during the whole join
fn produce_from_matched(
    visited_left_side: &BooleanBufferBuilder,
    schema: &SchemaRef,
    column_indices: &[ColumnIndex],
    left_data: &JoinLeftData,
    unmatched: bool,
) -> ArrowResult<RecordBatch> {
    let indices = if unmatched {
        UInt64Array::from_iter_values(
            (0..visited_left_side.len())
                .filter_map(|v| (!visited_left_side.get_bit(v)).then(|| v as u64)),
        )
    } else {
        UInt64Array::from_iter_values(
            (0..visited_left_side.len())
                .filter_map(|v| (visited_left_side.get_bit(v)).then(|| v as u64)),
        )
    };

    // generate batches by taking values from the left side and generating columns filled with null on the right side
    let num_rows = indices.len();
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());
    for (idx, column_index) in column_indices.iter().enumerate() {
        let array = match column_index.side {
            JoinSide::Left => {
                let array = left_data.1.column(column_index.index);
                compute::take(array.as_ref(), &indices, None).unwrap()
            }
            JoinSide::Right => {
                let datatype = schema.field(idx).data_type();
                arrow::array::new_null_array(datatype, num_rows)
            }
        };

        columns.push(array);
    }
    RecordBatch::try_new(schema.clone(), columns)
}
//Contains all of the required data to execute the hash join
struct JoinData{
    /// Input schema
    schema: Arc<Schema>,
    /// columns from the left
    on_left: Vec<Column>,
    /// columns from the right used to compute the hash
    on_right: Vec<Column>,
    /// type of the join
    join_type: JoinType,
    /// information from the left
    left_data: JoinLeftData,
    left_join_values: Vec<Arc<dyn Array>>,
    /// Random state used for hashing initialization
    random_state: RandomState,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// If null_equals_null is true, null == null else null != null
    null_equals_null: bool,
}


fn build_semi_or_anti_join_indices<const N: usize>(left_join_values: &[Arc<dyn Array>], keys_values:&[Arc<dyn Array>], left_hashmap: &JoinHashMap, buffers: &mut HashJoinBuffers<N>, random_state: &RandomState, null_equals_null: bool)->Result<UInt64Array>{
        // Using a buffer builder to avoid slower normal builder
        let mut left_indices = UInt64BufferBuilder::new(0);
        let mut right_idx = 0;
        let mut buffer_idx = 0;
        let right_row_count = keys_values[0].len();
        while right_idx < right_row_count{
            let num_to_process = (right_row_count - right_idx).min(N);
            //Fill the buffer with num_to_process hashes
            create_hashes_new(&keys_values, &random_state, &mut buffers.hash_buffer, right_idx, num_to_process)?;
    
            
            //For each hash get matching row mappings. When buffer is full then 
            for hash_idx in 0..num_to_process{
                let row = right_idx + hash_idx;
                let hash_value = buffers.hash_buffer[hash_idx];
                if let Some((_, indices)) = left_hashmap.0.get(hash_value, |(hash, _)| hash_value == *hash){
                    for i in indices{
                        buffers.index_buffer[buffer_idx]=(*i, row);
                        buffer_idx +=1;
                        if buffer_idx == N{
                            check_equal_rows_vectorized_new(&mut buffers.match_buffer, &left_join_values, keys_values, &buffers.index_buffer, N, null_equals_null)?;
                            buffers.build_mappings_semi_or_anti(&mut left_indices, N);
                            buffer_idx = 0;
                        }
                    }
                }
            }
            right_idx += num_to_process;
        }
        if buffer_idx != 0{
            check_equal_rows_vectorized_new(&mut buffers.match_buffer, &left_join_values, keys_values, &buffers.index_buffer, buffer_idx, null_equals_null)?;
            buffers.build_mappings_semi_or_anti(&mut left_indices, buffer_idx);
        }
    
        let left = ArrayData::builder(DataType::UInt64)
            .len(left_indices.len())
            .add_buffer(left_indices.finish())
            .build()
            .unwrap();
        
        Ok(PrimitiveArray::<UInt64Type>::from(left))
}


impl JoinData{
    fn try_new(schema: Arc<Schema>, on_left: Vec<Column>, on_right: Vec<Column>, join_type: JoinType, left_data: JoinLeftData,random_state: RandomState,column_indices: Vec<ColumnIndex>,null_equals_null: bool)->Result<Self>{
        let left_join_values = Self::left_join_values(&on_left, &left_data)?;
        Ok(
            Self{
                schema,
                on_left,
                on_right,
                join_type,
                left_data,
                left_join_values,
                random_state,
                column_indices,
                null_equals_null,
            }
        )
    }

    fn left_join_values(on_left: &[Column], left_data: &JoinLeftData)->Result<Vec<Arc<dyn Array>>>{
        on_left
            .iter()
            .map(|c| Ok(c.evaluate(&left_data.1)?.into_array(left_data.1.num_rows())))
            .collect::<Result<Vec<_>>>()
    }

    fn keys_values(&self, right: &RecordBatch)->Result<Vec<Arc<dyn Array>>>{
        self.on_right
            .iter()
            .map(|c| Ok(c.evaluate(right)?.into_array(right.num_rows())))
            .collect::<Result<Vec<_>>>()
    }
}

fn build_batch_new<const N: usize>(right: &RecordBatch, join_data: &JoinData, buffers: &mut HashJoinBuffers<N>)->ArrowResult<(RecordBatch, UInt64Array)>{
    let keys_values = join_data.keys_values(right)?;
    let (left_indices, right_indices) = match join_data.join_type{
        JoinType::Left |
        JoinType::Inner => build_join_indexes_inner_or_left(&join_data.left_join_values, &join_data.left_data.0, &keys_values, &join_data.random_state, buffers, join_data.null_equals_null)?,
        JoinType::Right |
        JoinType::Full => build_join_indexes_full_or_right(&join_data.left_join_values, &join_data.left_data.0, &keys_values, &join_data.random_state, buffers, join_data.null_equals_null)?,
        JoinType::Semi| 
        JoinType::Anti =>{
            let left_indices = build_join_indexes_semi_or_anti(&join_data.left_join_values, &join_data.left_data.0, &keys_values, &join_data.random_state, buffers, join_data.null_equals_null)?;
            return Ok((RecordBatch::new_empty(join_data.schema.clone()), left_indices));
        } 
    };
    build_batch_from_indices(join_data, right, left_indices, right_indices)
}


impl HashJoinStream{
    fn build_batch(&mut self, right: &RecordBatch)->ArrowResult<(RecordBatch, UInt64Array)>{
        let join_data = &self.join_data;
        let buffers = &mut self.join_buffers;
        build_batch_new(right, join_data, buffers)
    }
}

impl Stream for HashJoinStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.right
            .poll_next_unpin(cx)
            .map(|maybe_batch| match maybe_batch {
                Some(Ok(batch)) => {
                    let start = self.join_metrics.join_time.clone();
                    let timer = start.timer();
                    let result = self.build_batch(&batch);
                    
                    self.join_metrics.input_batches.add(1);
                    self.join_metrics.input_rows.add(batch.num_rows());
                    if let Ok((ref batch, ref left_side)) = result {
                        timer.done();
                        self.join_metrics.output_batches.add(1);
                        self.join_metrics.output_rows.add(batch.num_rows());

                        match self.join_data.join_type {
                            JoinType::Left
                            | JoinType::Full
                            | JoinType::Semi
                            | JoinType::Anti => {
                                left_side.iter().flatten().for_each(|x| {
                                    self.visited_left_side.set_bit(x as usize, true);
                                });
                            }
                            JoinType::Inner | JoinType::Right => {}
                        }
                    }
                    Some(result.map(|x| x.0))
                }
                other => {
                    let timer = self.join_metrics.join_time.timer();
                    // For the left join, produce rows for unmatched rows
                    match self.join_data.join_type {
                        JoinType::Left
                        | JoinType::Full
                        | JoinType::Semi
                        | JoinType::Anti
                            if !self.is_exhausted =>
                        {
                            let result = produce_from_matched(
                                &self.visited_left_side,
                                &self.join_data.schema,
                                &self.join_data.column_indices,
                                &self.join_data.left_data,
                                self.join_data.join_type != JoinType::Semi,
                            );
                            if let Ok(ref batch) = result {
                                self.join_metrics.input_batches.add(1);
                                self.join_metrics.input_rows.add(batch.num_rows());
                                if let Ok(ref batch) = result {
                                    self.join_metrics.output_batches.add(1);
                                    self.join_metrics.output_rows.add(batch.num_rows());
                                }
                            }
                            timer.done();
                            self.is_exhausted = true;
                            return Some(result);
                        }
                        JoinType::Left
                        | JoinType::Full
                        | JoinType::Semi
                        | JoinType::Anti
                        | JoinType::Inner
                        | JoinType::Right => {}
                    }

                    other
                }
            })
            
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_batches_sorted_eq,
        physical_plan::{
            common, expressions::Column, memory::MemoryExec, repartition::RepartitionExec,
        },
        test::{build_table_i32, columns},
    };

    use super::*;
    use crate::prelude::SessionContext;
    use std::sync::Arc;

    fn build_table(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();
        Arc::new(MemoryExec::try_new(&[vec![batch]], schema, None).unwrap())
    }

    fn join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equals_null: bool,
    ) -> Result<HashJoinExec> {
        HashJoinExec::try_new(
            left,
            right,
            on,
            join_type,
            PartitionMode::CollectLeft,
            &null_equals_null,
        )
    }

    async fn join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>)> {
        let join = join(left, right, on, join_type, null_equals_null)?;
        let columns = columns(&join.schema());

        let stream = join.execute(0, context).await?;
        let batches = common::collect(stream).await?;

        Ok((columns, batches))
    }

    async fn partitioned_join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equals_null: bool,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>)> {
        let partition_count = 4;

        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| {
                (
                    Arc::new(l.clone()) as Arc<dyn PhysicalExpr>,
                    Arc::new(r.clone()) as Arc<dyn PhysicalExpr>,
                )
            })
            .unzip();

        let join = HashJoinExec::try_new(
            Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::Hash(left_expr, partition_count),
            )?),
            Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::Hash(right_expr, partition_count),
            )?),
            on,
            join_type,
            PartitionMode::Partitioned,
            &null_equals_null,
        )?;

        let columns = columns(&join.schema());

        let mut batches = vec![];
        for i in 0..partition_count {
            let stream = join.execute(i, context.clone()).await?;
            let more_batches = common::collect(stream).await?;
            batches.extend(
                more_batches
                    .into_iter()
                    .filter(|b| b.num_rows() > 0)
                    .collect::<Vec<_>>(),
            );
        }

        Ok((columns, batches))
    }

    #[tokio::test]
    async fn join_inner_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );

        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Inner,
            false,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_inner_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = partitioned_join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Inner,
            false,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_one_no_shared_column_names() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b2", &right.schema())?,
        )];

        let (columns, batches) =
            join_collect(left, right, on, &JoinType::Inner, false, task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 5  | 9  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_two() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 2]),
            ("b2", &vec![1, 2, 2]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Column::new_with_schema("a1", &left.schema())?,
                Column::new_with_schema("a1", &right.schema())?,
            ),
            (
                Column::new_with_schema("b2", &left.schema())?,
                Column::new_with_schema("b2", &right.schema())?,
            ),
        ];

        let (columns, batches) =
            join_collect(left, right, on, &JoinType::Inner, false, task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b2 | c1 | a1 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 1  | 7  | 1  | 1  | 70 |",
            "| 2  | 2  | 8  | 2  | 2  | 80 |",
            "| 2  | 2  | 9  | 2  | 2  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    /// Test where the left has 2 parts, the right with 1 part => 1 part
    #[tokio::test]
    async fn join_inner_one_two_parts_left() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let batch1 = build_table_i32(
            ("a1", &vec![1, 2]),
            ("b2", &vec![1, 2]),
            ("c1", &vec![7, 8]),
        );
        let batch2 =
            build_table_i32(("a1", &vec![2]), ("b2", &vec![2]), ("c1", &vec![9]));
        let schema = batch1.schema();
        let left = Arc::new(
            MemoryExec::try_new(&[vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Column::new_with_schema("a1", &left.schema())?,
                Column::new_with_schema("a1", &right.schema())?,
            ),
            (
                Column::new_with_schema("b2", &left.schema())?,
                Column::new_with_schema("b2", &right.schema())?,
            ),
        ];

        let (columns, batches) =
            join_collect(left, right, on, &JoinType::Inner, false, task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b2 | c1 | a1 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 1  | 7  | 1  | 1  | 70 |",
            "| 2  | 2  | 8  | 2  | 2  | 80 |",
            "| 2  | 2  | 9  | 2  | 2  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    /// Test where the left has 1 part, the right has 2 parts => 2 parts
    #[tokio::test]
    async fn join_inner_one_two_parts_right() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );

        let batch1 = build_table_i32(
            ("a2", &vec![10, 20]),
            ("b1", &vec![4, 6]),
            ("c2", &vec![70, 80]),
        );
        let batch2 =
            build_table_i32(("a2", &vec![30]), ("b1", &vec![5]), ("c2", &vec![90]));
        let schema = batch1.schema();
        let right = Arc::new(
            MemoryExec::try_new(&[vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Inner, false)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        // first part
        let stream = join.execute(0, task_ctx.clone()).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 1);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        // second part
        let stream = join.execute(1, task_ctx.clone()).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 1);
        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 2  | 5  | 8  | 30 | 5  | 90 |",
            "| 3  | 5  | 9  | 30 | 5  | 90 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    fn build_table_two_batches(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();
        Arc::new(
            MemoryExec::try_new(&[vec![batch.clone(), batch]], schema, None).unwrap(),
        )
    }

    #[tokio::test]
    async fn join_left_multi_batch() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b1", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Left, false).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let stream = join.execute(0, task_ctx).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_full_multi_batch() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        // create two identical batches for the right side
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Full, false).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_left_empty_right() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b1", &right.schema()).unwrap(),
        )];
        let schema = right.schema();
        let right = Arc::new(MemoryExec::try_new(&[vec![right]], schema, None).unwrap());
        let join = join(left, right, on, &JoinType::Left, false).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let stream = join.execute(0, task_ctx).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  |    |    |    |",
            "| 2  | 5  | 8  |    |    |    |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_full_empty_right() {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b2", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", &right.schema()).unwrap(),
        )];
        let schema = right.schema();
        let right = Arc::new(MemoryExec::try_new(&[vec![right]], schema, None).unwrap());
        let join = join(left, right, on, &JoinType::Full, false).unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx).await.unwrap();
        let batches = common::collect(stream).await.unwrap();

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  |    |    |    |",
            "| 2  | 5  | 8  |    |    |    |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);
    }

    #[tokio::test]
    async fn join_left_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Left,
            false,
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_left_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) = partitioned_join_collect(
            left.clone(),
            right.clone(),
            on.clone(),
            &JoinType::Left,
            false,
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_semi() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 2, 3]),
            ("b1", &vec![4, 5, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 5, 6, 5]), // 5 is double on the right
            ("c2", &vec![70, 80, 90, 100]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Semi, false)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 1  | 4  | 7  |",
            "| 2  | 5  | 8  |",
            "| 2  | 5  | 8  |",
            "+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_anti() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 2, 3, 5]),
            ("b1", &vec![4, 5, 5, 7, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 8, 9, 11]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 5, 6, 5]), // 5 is double on the right
            ("c2", &vec![70, 80, 90, 100]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let join = join(left, right, on, &JoinType::Anti, false)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 3  | 7  | 9  |",
            "| 5  | 7  | 11 |",
            "+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    async fn join_right_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) =
            join_collect(left, right, on, &JoinType::Right, false, task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_right_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema())?,
            Column::new_with_schema("b1", &right.schema())?,
        )];

        let (columns, batches) =
            partitioned_join_collect(left, right, on, &JoinType::Right, false, task_ctx)
                .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b1 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];

        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[tokio::test]
    async fn join_full_one() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Column::new_with_schema("b1", &left.schema()).unwrap(),
            Column::new_with_schema("b2", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Full, false)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }

    #[test]
    fn join_with_hash_collision() -> Result<()> {
        let mut hashmap_left = RawTable::with_capacity(2);
        let left = build_table_i32(
            ("a", &vec![10, 20]),
            ("x", &vec![100, 200]),
            ("y", &vec![200, 300]),
        );

        let random_state = RandomState::with_seeds(0, 0, 0, 0);
        let hashes_buff = &mut vec![0; left.num_rows()];
        let hashes =
            create_hashes(&[left.columns()[0].clone()], &random_state, hashes_buff)?;

        // Create hash collisions (same hashes)
        hashmap_left.insert(hashes[0], (hashes[0], smallvec![0, 1]), |(h, _)| *h);
        hashmap_left.insert(hashes[1], (hashes[1], smallvec![0, 1]), |(h, _)| *h);

        let right = build_table_i32(
            ("a", &vec![10, 20]),
            ("b", &vec![0, 0]),
            ("c", &vec![30, 40]),
        );

        let left_data = JoinLeftData::new((JoinHashMap(hashmap_left), left));
        let mut match_buffer = Vec::new();
        let mut index_buffer = Vec::new();
        let (l, r) = build_join_indexes(
            &mut match_buffer,
            &mut index_buffer,
            &left_data,
            &right,
            JoinType::Inner,
            &[Column::new("a", 0)],
            &[Column::new("a", 0)],
            &random_state,
            &false,
        )?;

        let mut left_ids = UInt64Builder::new(0);
        left_ids.append_value(0)?;
        left_ids.append_value(1)?;

        let mut right_ids = UInt32Builder::new(0);
        right_ids.append_value(0)?;
        right_ids.append_value(1)?;

        assert_eq!(left_ids.finish(), l);

        assert_eq!(right_ids.finish(), r);

        Ok(())
    }

    #[tokio::test]
    async fn join_with_duplicated_column_names() -> Result<()> {
        let session_ctx = SessionContext::new();
        let task_ctx = session_ctx.task_ctx();
        let left = build_table(
            ("a", &vec![1, 2, 3]),
            ("b", &vec![4, 5, 7]),
            ("c", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30]),
            ("b", &vec![1, 2, 7]),
            ("c", &vec![70, 80, 90]),
        );
        let on = vec![(
            // join on a=b so there are duplicate column names on unjoined columns
            Column::new_with_schema("a", &left.schema()).unwrap(),
            Column::new_with_schema("b", &right.schema()).unwrap(),
        )];

        let join = join(left, right, on, &JoinType::Inner, false)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx).await?;
        let batches = common::collect(stream).await?;

        let expected = vec![
            "+---+---+---+----+---+----+",
            "| a | b | c | a  | b | c  |",
            "+---+---+---+----+---+----+",
            "| 1 | 4 | 7 | 10 | 1 | 70 |",
            "| 2 | 5 | 8 | 20 | 2 | 80 |",
            "+---+---+---+----+---+----+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        Ok(())
    }
}
