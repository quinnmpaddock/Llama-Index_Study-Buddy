use crate::api::models::{
    CommunityDetail, CommunityEntitiesResponse, CommunityListResponse, EntityDetail,
    EntitySearchResponse, GraphQueryResponse, HealthResponse, IngestPreviewResponse,
    IngestRequest, IngestResponse, IngestStatus, QueryRequest, SummaryCleanupResponse,
    SummaryListResponse, SummaryVersion,
};
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

    // =========================================================================
    // Entity Endpoints
    // =========================================================================

    /// Search for entities by name
    pub async fn search_entities(
        &self,
        query: Option<&str>,
        limit: i32,
    ) -> Result<EntitySearchResponse> {
        let mut url = format!("{}/entities?limit={}", self.base_url, limit);
        if let Some(q) = query {
            url.push_str(&format!("&q={}", urlencoding::encode(q)));
        }

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: EntitySearchResponse = response.json().await?;
        Ok(result)
    }

    /// Get details for a specific entity
    pub async fn get_entity(&self, name: &str) -> Result<EntityDetail> {
        let url = format!("{}/entities/{}", self.base_url, urlencoding::encode(name));

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: EntityDetail = response.json().await?;
        Ok(result)
    }

    // =========================================================================
    // Community Endpoints
    // =========================================================================

    /// List all communities
    pub async fn list_communities(&self) -> Result<CommunityListResponse> {
        let url = format!("{}/communities", self.base_url);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: CommunityListResponse = response.json().await?;
        Ok(result)
    }

    /// Get details for a specific community
    pub async fn get_community(&self, id: i32) -> Result<CommunityDetail> {
        let url = format!("{}/communities/{}", self.base_url, id);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: CommunityDetail = response.json().await?;
        Ok(result)
    }

    /// Get entities in a specific community
    pub async fn get_community_entities(&self, id: i32) -> Result<CommunityEntitiesResponse> {
        let url = format!("{}/communities/{}/entities", self.base_url, id);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: CommunityEntitiesResponse = response.json().await?;
        Ok(result)
    }

    // =========================================================================
    // Ingestion Endpoints
    // =========================================================================

    /// Preview files that would be ingested from a directory
    pub async fn preview_ingest(&self, directory: &str) -> Result<IngestPreviewResponse> {
        let url = format!(
            "{}/ingest/preview?directory={}",
            self.base_url,
            urlencoding::encode(directory)
        );

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: IngestPreviewResponse = response.json().await?;
        Ok(result)
    }

    /// Ingest documents from a directory
    pub async fn ingest(&self, request: IngestRequest) -> Result<IngestResponse> {
        let url = format!("{}/ingest", self.base_url);

        let response = self.client.post(&url).json(&request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: IngestResponse = response.json().await?;
        Ok(result)
    }

    /// Get the status of an ingestion task
    pub async fn get_ingest_status(&self, task_id: &str) -> Result<IngestStatus> {
        let url = format!("{}/ingest/status/{}", self.base_url, task_id);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: IngestStatus = response.json().await?;
        Ok(result)
    }

    // =========================================================================
    // Summaries Endpoints
    // =========================================================================

    /// List all summary versions
    pub async fn list_summaries(&self) -> Result<SummaryListResponse> {
        let url = format!("{}/summaries", self.base_url);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: SummaryListResponse = response.json().await?;
        Ok(result)
    }

    /// Get the current summary version info
    pub async fn get_current_summary(&self) -> Result<SummaryVersion> {
        let url = format!("{}/summaries/current", self.base_url);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: SummaryVersion = response.json().await?;
        Ok(result)
    }

    /// Get a specific summary version's content
    pub async fn get_summary_version(&self, version: &str) -> Result<serde_json::Value> {
        let url = format!("{}/summaries/{}", self.base_url, version);

        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: serde_json::Value = response.json().await?;
        Ok(result)
    }

    /// Delete old summary versions, keeping N most recent
    pub async fn cleanup_summaries(&self, keep: i32) -> Result<SummaryCleanupResponse> {
        let url = format!("{}/summaries?keep={}", self.base_url, keep);

        let response = self.client.delete(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(StudyBuddyError::ApiError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let result: SummaryCleanupResponse = response.json().await?;
        Ok(result)
    }
}