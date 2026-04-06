use serde::{Deserialize, Serialize};

/// Request body for the /query endpoint
#[derive(Debug, Clone, Serialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_top_k: Option<i32>,
}

impl QueryRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            similarity_top_k: None,
        }
    }

    pub fn with_top_k(mut self, k: i32) -> Self {
        self.similarity_top_k = Some(k);
        self
    }
}

/// Response from the /query endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GraphQueryResponse {
    pub answer: String,
    pub communities_consulted: Vec<CommunityId>,
    pub entities_found: Vec<String>,
}

/// Community ID can be either string or int in the API response
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CommunityId {
    Int(i32),
    String(String),
}

impl std::fmt::Display for CommunityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunityId::Int(i) => write!(f, "{}", i),
            CommunityId::String(s) => write!(f, "{}", s),
        }
    }
}

/// Health check response
#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    pub message: String,
}

// =============================================================================
// Entity Models
// =============================================================================

/// Single entity from search results
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntityInfo {
    pub name: String,
    pub communities: Vec<i32>,
}

/// Response from /entities endpoint (search)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntitySearchResponse {
    pub entities: Vec<EntityInfo>,
    pub total: i32,
}

/// Response from /entities/{name} endpoint (single entity details)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntityDetail {
    pub name: String,
    pub communities: Vec<i32>,
}

// =============================================================================
// Community Models
// =============================================================================

/// Single community from list
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CommunityInfo {
    pub id: i32,
    pub entity_count: i32,
    pub summary_preview: String,
}

/// Response from /communities endpoint (list all)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CommunityListResponse {
    pub communities: Vec<CommunityInfo>,
    pub total: i32,
}

/// Response from /communities/{id} endpoint (single community details)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CommunityDetail {
    pub id: i32,
    pub summary: String,
    pub entity_count: i32,
}

/// Response from /communities/{id}/entities endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CommunityEntitiesResponse {
    pub community_id: i32,
    pub entities: Vec<String>,
    pub total: i32,
}

// =============================================================================
// Ingestion Models
// =============================================================================

/// Request for the /ingest endpoint
#[derive(Debug, Clone, Serialize)]
pub struct IngestRequest {
    pub directory: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
}

/// Response from /ingest endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IngestResponse {
    pub status: String,
    pub directory: String,
    pub files_processed: Vec<String>,
    pub total_nodes: i32,
    #[serde(default)]
    pub total_entities: i32,
    #[serde(default)]
    pub total_relationships: i32,
    #[serde(default)]
    pub communities_built: i32,
    pub message: String,
}

/// Ingestion task status
#[derive(Debug, Clone, Deserialize)]
pub struct IngestStatus {
    pub status: String,
    pub progress: i32,
    #[serde(default)]
    pub total_nodes: i32,
    #[serde(default)]
    pub total_entities: i32,
    #[serde(default)]
    pub total_communities: i32,
    #[serde(default)]
    pub files_processed: Vec<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// File info from /ingest/preview endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct IngestFileInfo {
    pub name: String,
    pub extension: String,
    pub size_bytes: u64,
}

/// Response from /ingest/preview endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct IngestPreviewResponse {
    pub directory: String,
    pub supported_extensions: Vec<String>,
    pub files: Vec<IngestFileInfo>,
    pub total_files: i32,
}

// =========================================================================
// Summaries Endpoints
// =========================================================================

/// Summary version info
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SummaryVersion {
    pub version: String,
    pub created_at: String,
    pub files: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub stats: std::collections::HashMap<String, i32>,
}

/// Response from /summaries endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SummaryListResponse {
    pub current: Option<SummaryVersion>,
    pub versions: Vec<SummaryVersionInfo>,
}

/// Version info in listing
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SummaryVersionInfo {
    pub version: String,
    pub filename: String,
    pub modified: String,
    pub size_bytes: u64,
}

/// Response from DELETE /summaries endpoint
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SummaryCleanupResponse {
    pub deleted: Vec<String>,
    pub kept: Vec<String>,
    pub message: String,
}