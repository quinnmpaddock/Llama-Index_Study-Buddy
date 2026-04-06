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