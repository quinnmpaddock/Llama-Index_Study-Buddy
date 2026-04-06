use crate::api::models::{
    CommunityDetail, CommunityEntitiesResponse, CommunityListResponse, EntityDetail,
    EntitySearchResponse, GraphQueryResponse, HealthResponse, QueryRequest,
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
}