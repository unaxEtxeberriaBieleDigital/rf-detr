Arquitectura propuesta

El problema tiene 3 ejes ortogonales que hay que desacoplar: modelo (embeddings+predicciones), dataset (imágenes+ground truth), y evaluación/transporte (comparar contra GT y servir por HTTP). Si mezclas esos ejes, cada modelo/dataset nuevo obliga a tocar la API. La clave es definir un esquema canónico intermedio que todos produzcan, y que la API solo hable ese esquema.

1.  BaseModel  (ya tienes la base) — contrato único para detectores y clasificadores

@dataclass
class Prediction:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # None en clasificación

class BaseModel(ABC):
    @abstractmethod
    def get_batch_embeddings(self, batch) -> tuple[torch.Tensor, list[list[Prediction]]]: ...

Un clasificador siempre devuelve listas internas de longitud 1 (sin bbox); un detector, N por imagen. Así el resto del pipeline no necesita saber si es detector o clasificador.

2.  BaseDataset  — añade ground truth, no solo iteración de imágenes

class BaseDataset(ABC):
    task_type: Literal["detection", "classification"]  # decide qué evaluador usar
    @abstractmethod
    def iter_batches(self, split, batch_size) -> Iterator[list[Path]]: ...
    @abstractmethod
    def get_ground_truth(self, image: Path) -> list[Prediction]: ...  # mismo tipo que las predicciones
    categories: dict[int, str]

 COCODetectionDataset  y  ClassificationDataset  implementan esto. Un dataset nuevo (p.ej. YOLO-txt) solo implementa estos 2 métodos.

3. Registry — para no tocar la API al añadir tipos

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}

def register_model(name):
    def deco(cls): MODEL_REGISTRY[name] = cls; return cls
    return deco

La API expone  GET /model-types  y  GET /dataset-types  leyendo el registry dinámicamente.

4. Evaluator — matching genérico por  task_type , no por modelo concreto

• Detección: IoU-matching (estilo COCO) entre  predictions  y  get_ground_truth()  → TP / FP / FN / misclassification.
• Clasificación: comparación directa de  class_id  → correct / incorrect.

Esto vive en un módulo separado ( evaluator/detection.py ,  evaluator/classification.py ) seleccionado por  dataset.task_type . Añadir un modelo nunca toca esto; solo añadir un task_type nuevo lo haría.

5. Esquema canónico que viaja por HTTP ( EmbeddingRecord )

@dataclass
class EmbeddingRecord:
    id: str
    image_path: str
    split: str
    embedding: list[float]          # o coords PCA ya reducidas
    prediction: Prediction | None
    ground_truth: Prediction | None
    status: Literal["tp","fp","fn","misclassified","correct","incorrect"]

La API nunca serializa objetos de modelo/dataset directamente: siempre pasa por  EmbeddingRecord  (Pydantic). Esto es lo que estandariza la comunicación, independientemente de qué modelo/dataset lo generó.

6. API (FastAPI recomendado sobre Flask por Pydantic + async + docs automáticas)

•  /api/v1/model-types ,  /api/v1/dataset-types  (discovery)
•  POST /api/v1/jobs   {dataset_path, dataset_type, model_path, model_type}  → job asíncrono (inferencia puede tardar)
•  GET /api/v1/jobs/{id}  → status
•  GET /api/v1/jobs/{id}/records?split=&status=&class_id=&pca=2  → filtrado + PCA on-demand (sklearn  PCA , cacheado por job)
•  GET /api/v1/images/{record_id}  → sirve el bytes de imagen (para overlay de bbox en frontend)
• CORS abierto para Tauri/web; versionar desde ya ( /v1/ ).

Por qué esto resuelve tu objetivo

• Modelo nuevo → solo  class Foo(BaseModel)  +  get_batch_embeddings  +  @register_model . Cero cambios en API/evaluador.
• Dataset nuevo → solo  iter_batches  +  get_ground_truth  + declarar  task_type . Si el  task_type  ya existe, el evaluador se reutiliza tal cual.
• Frontend solo conoce  EmbeddingRecord  — nunca depende de si detrás hay RF-DETR o un clasificador ResNet.