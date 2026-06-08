from pathlib import Path
from typing import Literal

from joblib import Memory, Parallel

STRUCT_ATTRS_TYPE = list[str] | Literal["all"]

AUX_COLS_INFERENCE_TYPE = list[str] | Literal["feat", "struct", "all"]

AUX_COLS_LINKABILITY_TYPE = (
    tuple[list[str], list[str]]
    | Literal["feat-feat", "struct-struct", "feat-struct", "random"]
)

MEMORY = Memory(Path(__file__).parent / "memory", verbose=0)

PARALLEL = Parallel(n_jobs=-2)
