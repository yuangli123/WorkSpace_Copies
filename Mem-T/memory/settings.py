"""Settings used by the memory tools; model and experiment settings live in configs/."""
from dataclasses import dataclass, field


@dataclass
class VectorDBConfig:
    embedding_model: str = "BAAI/bge-m3"
    backend: str = "chroma"
    db_type: str = "persistent"
    path: str = "artifacts/database"
    host: str = "localhost"
    port: int = 8070
    index_type: str = "HNSW"
    metric_type: str = "cosine"
    from_scratch: bool = False


@dataclass
class MemoryConfig:
    retrieval_topk: int = 5
    max_tool_steps: int = 6


@dataclass
class SystemConfig:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data_name: str = "locomo"
    traj_dir: str = "artifacts/tool_logs"
