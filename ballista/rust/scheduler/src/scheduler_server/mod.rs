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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tonic::transport::Channel;

use ballista_core::config::{BallistaConfig, TaskSchedulingPolicy};
use ballista_core::error::Result;
use ballista_core::event_loop::EventLoop;
use ballista_core::serde::protobuf::executor_grpc_client::ExecutorGrpcClient;

use ballista_core::serde::{AsExecutionPlan, AsLogicalPlan, BallistaCodec};
use datafusion::prelude::{ExecutionConfig, ExecutionContext};

use crate::scheduler_server::event_loop::{
    SchedulerServerEvent, SchedulerServerEventAction,
};
use crate::state::backend::StateBackendClient;
use crate::state::SchedulerState;

// include the generated protobuf source as a submodule
#[allow(clippy::all)]
pub mod externalscaler {
    include!(concat!(env!("OUT_DIR"), "/externalscaler.rs"));
}

mod event_loop;
mod external_scaler;
mod grpc;

type ExecutorsClient = Arc<RwLock<HashMap<String, ExecutorGrpcClient<Channel>>>>;

#[derive(Clone)]
pub struct SchedulerServer<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> {
    pub(crate) state: Arc<SchedulerState<T, U>>,
    pub start_time: u128,
    policy: TaskSchedulingPolicy,
    executors_client: Option<ExecutorsClient>,
    event_loop: Option<EventLoop<SchedulerServerEvent>>,
    ctx: Arc<RwLock<ExecutionContext>>,
    codec: BallistaCodec<T, U>,
}

impl<T: 'static + AsLogicalPlan, U: 'static + AsExecutionPlan> SchedulerServer<T, U> {
    pub fn new(
        config: Arc<dyn StateBackendClient>,
        namespace: String,
        ctx: Arc<RwLock<ExecutionContext>>,
        codec: BallistaCodec<T, U>,
    ) -> Self {
        SchedulerServer::new_with_policy(
            config,
            namespace,
            TaskSchedulingPolicy::PullStaged,
            ctx,
            codec,
        )
    }

    pub fn new_with_policy(
        config: Arc<dyn StateBackendClient>,
        namespace: String,
        policy: TaskSchedulingPolicy,
        ctx: Arc<RwLock<ExecutionContext>>,
        codec: BallistaCodec<T, U>,
    ) -> Self {
        let state = Arc::new(SchedulerState::new(config, namespace, codec.clone()));

        let (executors_client, event_loop) =
            if matches!(policy, TaskSchedulingPolicy::PushStaged) {
                let executors_client = Arc::new(RwLock::new(HashMap::new()));
                let event_action: Arc<SchedulerServerEventAction<T, U>> =
                    Arc::new(SchedulerServerEventAction::new(
                        state.clone(),
                        executors_client.clone(),
                    ));
                let event_loop =
                    EventLoop::new("scheduler".to_owned(), 10000, event_action);
                (Some(executors_client), Some(event_loop))
            } else {
                (None, None)
            };
        Self {
            state,
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            policy,
            executors_client,
            event_loop,
            ctx,
            codec,
        }
    }

    pub async fn init(&mut self) -> Result<()> {
        {
            // initialize state
            let ctx = self.ctx.read().await;
            self.state.init(&ctx).await?;
        }

        {
            if let Some(event_loop) = self.event_loop.as_mut() {
                event_loop.start()?;
            }
        }

        Ok(())
    }
}

/// Create a DataFusion context that is compatible with Ballista
pub fn create_datafusion_context(config: &BallistaConfig) -> ExecutionContext {
    let config = ExecutionConfig::new()
        .with_target_partitions(config.default_shuffle_partitions());
    ExecutionContext::with_config(config)
}
