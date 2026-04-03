use crate::api::models::{GraphQueryResponse, HealthResponse, QueryRequest};
use crate::config::Settings;
use crate::error::{Result, StudyBuddyError};
use reqwest::Client;
use std::time::Duration;

pub struct ApiClient {
    client: Client,
    base_url: String,
}

impl ApiClient {
    pub fn new(settings: &Settings) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(settings.api.timeout_seconds))
            .build()
            .map_err(StudyBuddyError::from)?;

        Ok(Self {
            client,
            base_url: settings.api.base_url.trim_end_matches('/').to_string(),
        })
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Check API health
    pub async fn health(&self) -> Result<String> {
        let response = self
            .client
            .get(format!("{}/", self.base_url))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: format!("Health check failed with status {}", status),
            });
        }

        let health: HealthResponse = response.json().await?;
        Ok(health.message)
    }

    /// Query the knowledge graph
    pub async fn query(&self, request: QueryRequest) -> Result<GraphQueryResponse> {
        let url = format!("{}/query", self.base_url);

        let response = self.client.post(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            // Try to parse error message from response
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: GraphQueryResponse = response.json().await?;
        Ok(result)
    }

    /// Check if the API is reachable
    pub async fn is_connected(&self) -> bool {
        self.health().await.is_ok()
    }
}