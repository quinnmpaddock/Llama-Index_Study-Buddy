"""
Study Buddy Configuration Module

Loads configuration from YAML file with environment variable overrides.
Priority: Environment Variables > Config File > Defaults
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

# Optional YAML support - fall back to basic parsing if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = "study_buddy.yaml"


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class LLMConfig:
    """LLM (Large Language Model) settings."""
    def __init__(self, config_dict: dict):
        self.model = config_dict.get("model", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.api_base = config_dict.get("api_base", "https://api.groq.com/openai/v1")
        # API key must come from environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY")


class EmbeddingConfig:
    """Embedding model settings."""
    def __init__(self, config_dict: dict):
        self.model = config_dict.get("model", "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5")


class Neo4jConfig:
    """Neo4j database settings."""
    def __init__(self, config_dict: dict):
        self.url = config_dict.get("url", "bolt://localhost:7687")
        self.username = config_dict.get("username", "neo4j")
        # Password: environment variable overrides config file
        self.password = os.environ.get("NEO4J_PASSWORD", config_dict.get("password", "neo4j2026"))
        self.timeout = config_dict.get("timeout", 30.0)


class ServerConfig:
    """Backend server settings."""
    def __init__(self, config_dict: dict):
        self.port = int(os.environ.get("SERVER_PORT", config_dict.get("port", 8000)))
        self.host = config_dict.get("host", "127.0.0.1")
        self.log_level = config_dict.get("log_level", "INFO")


class GraphRAGConfig:
    """Knowledge graph settings."""
    def __init__(self, config_dict: dict):
        self.max_paths_per_chunk = config_dict.get("max_paths_per_chunk", 2)
        self.extraction_prompt = config_dict.get("extraction_prompt", "kg_extract_template.txt")


class DockerConfig:
    """Docker container settings."""
    def __init__(self, config_dict: dict):
        self.container_name = config_dict.get("container_name", "neo4j-apoc-gds")
        self.image = config_dict.get("image", "neo4j:latest")
        self.http_port = config_dict.get("http_port", 7474)
        self.bolt_port = config_dict.get("bolt_port", 7687)


class Config:
    """
    Main configuration class.
    
    Usage:
        from config import config
        print(config.llm.model)
        print(config.neo4j.url)
    """
    
    _instance: Optional['Config'] = None
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.environ.get("STUDY_BUDDY_CONFIG", DEFAULT_CONFIG_PATH)
        self._raw_config = self._load_config()
        
        # Initialize all config sections
        self.llm = LLMConfig(self._raw_config.get("llm", {}))
        self.embedding = EmbeddingConfig(self._raw_config.get("embedding", {}))
        self.neo4j = Neo4jConfig(self._raw_config.get("neo4j", {}))
        self.server = ServerConfig(self._raw_config.get("server", {}))
        self.graphrag = GraphRAGConfig(self._raw_config.get("graphrag", {}))
        self.docker = DockerConfig(self._raw_config.get("docker", {}))
        
        # Validate required settings
        self._validate()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            logger.warning("Using default configuration. Create study_buddy.yaml to customize.")
            return {}
        
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed, parsing config file with basic parser")
            return self._parse_simple_yaml(config_file)
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}
    
    def _parse_simple_yaml(self, filepath: Path) -> dict:
        """
        Basic YAML parser for when PyYAML is not installed.
        Handles simple key: value pairs (no nested structures beyond one level).
        """
        config = {}
        current_section = None
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    
                    # Skip empty lines and comments
                    if not line or line.strip().startswith('#'):
                        continue
                    
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].rstrip()
                    
                    # Section header (e.g., "llm:")
                    if line.endswith(':') and not line.startswith(' '):
                        current_section = line[:-1]
                        config[current_section] = {}
                        continue
                    
                    # Key-value pair
                    if ':' in line:
                        # Handle indentation
                        stripped = line.strip()
                        if current_section and line.startswith(' ') and stripped:
                            key, value = stripped.split(':', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            config[current_section][key] = value
        except Exception as e:
            logger.error(f"Failed to parse config file: {e}")
        
        return config
    
    def _validate(self):
        """Validate required configuration settings."""
        if not self.llm.api_key:
            raise ConfigError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in your shell or .env file:\n"
                "  export OPENAI_API_KEY='your-api-key-here'"
            )
    
    def reload(self, config_path: Optional[str] = None):
        """Reload configuration from file."""
        if config_path:
            self.config_path = config_path
        self._raw_config = self._load_config()
        self.llm = LLMConfig(self._raw_config.get("llm", {}))
        self.embedding = EmbeddingConfig(self._raw_config.get("embedding", {}))
        self.neo4j = Neo4jConfig(self._raw_config.get("neo4j", {}))
        self.server = ServerConfig(self._raw_config.get("server", {}))
        self.graphrag = GraphRAGConfig(self._raw_config.get("graphrag", {}))
        self.docker = DockerConfig(self._raw_config.get("docker", {}))
        self._validate()
    
    @classmethod
    def get(cls) -> 'Config':
        """Get singleton config instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


# Global config instance (lazy initialization)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns a singleton Config object that loads from study_buddy.yaml
    on first access.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset the global config (useful for testing or reloading)."""
    global _config
    _config = None