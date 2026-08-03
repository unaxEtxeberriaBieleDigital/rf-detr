"""Interactive FiftyOne app for inspecting RF-DETR predictions and decoder-query embeddings.

Builds a FiftyOne dataset from the COCO ``train``/``valid``/``test`` splits of a Roboflow-style
detection dataset, runs RF-DETR inference (with per-detection decoder embeddings), evaluates
predictions against ground truth, and launches the FiftyOne App with saved views for:

    - each split (``train`` / ``valid`` / ``test``)
    - false positives / false negatives
    - detections that localized correctly but predicted the wrong class ("misclassified")

The per-detection embeddings (concatenated across decoder layers) are attached to each
predicted object so they can be visualized in 2D (UMAP/t-SNE) directly in the App, and
compared against ground truth labels.

Usage - ALWAYS via the project venv so CUDA torch is used, never ``uv run``/``uv sync``
(which resync the environment against the lockfile and revert torch to the CPU-only
pinned build, potentially clobbering fiftyone/fiftyone-brain/pycocotools too):

    .\\tests\\run_fiftyone_app.ps1

or, equivalently:

    .venv\\Scripts\\python.exe tests\\fiftyone_embeddings_app.py
"""

from __future__ import annotations

import os

import fiftyone as fo
import fiftyone.brain as fob
from PIL import Image
from tqdm.auto import tqdm

from rfdetr.utilities.logger import get_logger
from rfdetr.variants import RFDETRLarge

logger = get_logger()

# --- Configuration -----------------------------------------------------------------
DATASET_ROOT = r"C:\Users\u.etxeberria\Downloads\lat_dataset\training_dataset"
CHECKPOINT = r"C:\Users\u.etxeberria\Downloads\lat1_large.pth"
FIFTYONE_DATASET_NAME = "rfdetr-lat-embeddings"
SPLITS = ("train", "valid", "test")
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 4
# Limit the number of images loaded per split while iterating on the workflow. Set to
# `None` to process the full split once you're happy with the results.
MAX_SAMPLES_PER_SPLIT: int | None = None
IOU_THRESHOLD = 0.5
# If True, deletes any existing FiftyOne dataset with `FIFTYONE_DATASET_NAME` and rebuilds
# it from scratch: reloads the COCO splits, reruns RF-DETR inference (including decoder
# embeddings), re-evaluates against ground truth, and recomputes the embedding
# visualization. Set back to False for subsequent runs to reuse the cached dataset.
OVERRIDE: bool = False
# -----------------------------------------------------------------------------------


def build_dataset() -> fo.Dataset:
    """Builds (or loads a cached) FiftyOne dataset from the COCO splits, tagged by split.

    If `OVERRIDE` is True, any existing dataset named `FIFTYONE_DATASET_NAME` is deleted
    first so the splits, predictions, and embeddings are all regenerated from scratch.

    Returns:
        A persistent `fo.Dataset` containing samples from every split in `SPLITS`, each
        tagged with its split name and carrying ground truth detections in the
        `ground_truth` field.
    """
    try:
        if OVERRIDE and fo.dataset_exists(FIFTYONE_DATASET_NAME):
            logger.info(
                "OVERRIDE=True: deleting existing dataset '%s'...", FIFTYONE_DATASET_NAME
            )
            fo.delete_dataset(FIFTYONE_DATASET_NAME)

        if fo.dataset_exists(FIFTYONE_DATASET_NAME):
            existing = fo.load_dataset(FIFTYONE_DATASET_NAME)
            if len(existing) > 0:
                return existing
            # A previous run failed partway through building the dataset; start over.
            fo.delete_dataset(FIFTYONE_DATASET_NAME)
    except OSError as e:
        if "downgrading" not in str(e).lower() and "migrate" not in str(e).lower():
            raise
        raise RuntimeError(
            "FiftyOne's local database is stuck at a newer schema version than the "
            f"currently installed fiftyone=={fo.__version__} can read (original error: "
            f"{e}). This happens if the database was migrated up by a newer fiftyone "
            "version and then the package was downgraded without following the "
            "official downgrade procedure. Fix it with these 3 steps:\n\n"
            "  1. uv pip install fiftyone==1.20.0 fiftyone-brain>=0.23.0\n"
            "  2. .venv\\Scripts\\python.exe -c \"import fiftyone.migrations as fom; "
            "fom.migrate_database_if_necessary(destination='1.19.0')\"\n"
            "  3. uv pip install fiftyone==1.19.0 \"fiftyone-brain>=0.22.0,<0.23\" pycocotools\n\n"
            "IMPORTANT: run step 3 immediately after step 2, without running any other "
            "fiftyone command in between - fiftyone auto-upgrades the database schema "
            "back up to whatever version is currently installed as soon as it connects, "
            "so any fiftyone 1.20.0 command run between steps 2 and 3 undoes step 2."
        ) from e

    dataset = fo.Dataset(FIFTYONE_DATASET_NAME, persistent=True)
    for split in SPLITS:
        split_dir = os.path.join(DATASET_ROOT, split)
        full_split_dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=split_dir,
            labels_path=os.path.join(split_dir, "_annotations.coco.json"),
            label_field="ground_truth",
            label_types="detections",
        )
        split_view = (
            full_split_dataset.take(MAX_SAMPLES_PER_SPLIT)
            if MAX_SAMPLES_PER_SPLIT is not None
            else full_split_dataset
        )
        split_view.tag_samples(split)
        dataset.add_collection(split_view)
        full_split_dataset.delete()

    return dataset


def add_predictions(dataset: fo.Dataset, model: RFDETRLarge) -> None:
    """Runs RF-DETR over every sample, storing predictions and per-detection embeddings.

    Args:
        dataset: The FiftyOne dataset to populate with a `predictions` label field.
        model: A loaded RF-DETR model used to run inference.
    """
    samples = list(dataset)
    for start in tqdm(range(0, len(samples), BATCH_SIZE), desc="Running inference"):
        batch = samples[start : start + BATCH_SIZE]
        paths = [sample.filepath for sample in batch]
        predictions = model.predict(
            paths,
            threshold=CONFIDENCE_THRESHOLD,
            include_source_image=False,
            return_query_embeddings=True,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]

        for sample, detections in zip(batch, predictions):
            with Image.open(sample.filepath) as img:
                width, height = img.size

            fo_detections = []
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                label = str(detections.data["class_name"][i])
                confidence = float(detections.confidence[i])
                embedding = detections.query_embeddings[i]

                fo_detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=[
                            x1 / width,
                            y1 / height,
                            (x2 - x1) / width,
                            (y2 - y1) / height,
                        ],
                        confidence=confidence,
                        embedding=embedding.tolist(),
                    )
                )

            sample["predictions"] = fo.Detections(detections=fo_detections)
            sample.save()


def evaluate_and_flag_misclassifications(dataset: fo.Dataset) -> None:
    """Evaluates predictions against ground truth and flags class-only mistakes.

    Runs two COCO-style evaluations: one class-aware (standard TP/FP/FN under `eval`) and
    one class-agnostic (matches on IoU alone, under `eval_loc`). A prediction that is a
    false positive in the class-aware pass but a true positive in the class-agnostic pass
    localized an object correctly but predicted the wrong class; these are flagged via the
    boolean `misclassified` attribute on each predicted `Detection`.

    Args:
        dataset: The dataset with `predictions` and `ground_truth` label fields populated.
    """
    dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        classwise=True,
        iou=IOU_THRESHOLD,
    )
    dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval_loc",
        classwise=False,
        iou=IOU_THRESHOLD,
    )

    for sample in dataset.iter_samples(progress=True, autosave=True):
        for detection in sample.predictions.detections:
            detection["misclassified"] = detection.eval == "fp" and detection.eval_loc == "tp"


def compute_embedding_visualization(dataset: fo.Dataset) -> None:
    """Projects the per-detection decoder embeddings to 2D for inspection in the App.

    Args:
        dataset: The dataset with `predictions` detections carrying an `embedding`
            attribute, as populated by `add_predictions`.
    """
    fob.compute_visualization(
        dataset,
        patches_field="predictions",
        embeddings="embedding",
        brain_key="query_embeddings_2d",
        method="tsne",
    )


def save_views(dataset: fo.Dataset) -> None:
    """Saves convenience views for split filtering and error inspection.

    Args:
        dataset: The fully processed dataset (predictions + evaluation already computed).
    """
    for split in SPLITS:
        if not dataset.has_saved_view(split):
            dataset.save_view(split, dataset.match_tags(split))

    if not dataset.has_saved_view("false_positives"):
        dataset.save_view(
            "false_positives",
            dataset.filter_labels("predictions", fo.ViewField("eval") == "fp"),
        )
    if not dataset.has_saved_view("false_negatives"):
        dataset.save_view(
            "false_negatives",
            dataset.filter_labels("ground_truth", fo.ViewField("eval") == "fn"),
        )
    if not dataset.has_saved_view("misclassified"):
        dataset.save_view(
            "misclassified",
            dataset.filter_labels("predictions", fo.ViewField("misclassified") == True),  # noqa: E712
        )


def _check_cuda() -> None:
    """Verifies CUDA torch is active, aborting immediately if it is not.

    ``uv run``/``uv sync`` re-resolve this project's lockfile-pinned CPU-only
    torch build, silently reverting any manually installed CUDA build. This
    check aborts *before* touching the dataset (e.g. before an `OVERRIDE`
    delete) so an accidental ``uv run`` invocation can't waste time or wipe
    out the cached dataset while running on the CPU-only build.

    Raises:
        RuntimeError: if `torch.cuda.is_available()` is False.
    """
    import torch

    if torch.cuda.is_available():
        logger.info(
            "CUDA is available - using GPU '%s' for inference.",
            torch.cuda.get_device_name(0),
        )
        return

    raise RuntimeError(
        f"CUDA is NOT available (torch {torch.__version__}) - aborting before touching "
        "the dataset. This almost always means the project's CPU-only torch build got "
        "silently re-installed by 'uv run'/'uv sync' (which resync against the "
        "lockfile-pinned CPU-only torch). Fix it with:\n\n"
        "  uv pip install torch torchvision --extra-index-url "
        "https://download.pytorch.org/whl/cu126 --reinstall\n\n"
        "Then always launch this script via tests\\run_fiftyone_app.ps1 (or "
        ".venv\\Scripts\\python.exe directly) - never 'uv run', since that resyncs the "
        "environment and reverts torch to CPU-only again."
    )


def main() -> None:
    """Builds the dataset, runs inference/evaluation if needed, and launches the App."""
    _check_cuda()
    dataset = build_dataset()

    if "predictions" not in dataset.get_field_schema():
        logger.info("Running RF-DETR inference over %d samples...", len(dataset))
        model = RFDETRLarge(pretrain_weights=CHECKPOINT)
        add_predictions(dataset, model)
        evaluate_and_flag_misclassifications(dataset)

    if "query_embeddings_2d" not in dataset.list_brain_runs():
        logger.info("Computing 2D embedding visualization...")
        compute_embedding_visualization(dataset)

    save_views(dataset)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
