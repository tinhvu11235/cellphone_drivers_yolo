# Driver Cellphone Use YOLO Detector

This project fine-tunes a YOLO detector for driver cellphone-use detection. The source dataset is a Roboflow COCO export and is converted to Ultralytics YOLO format before training.

Classes:

- `Cellphone-in-drivers`
- `driver`
- `phone`
- `wheel`

The Gradio demo displays detections for all four classes and raises a practical alert when a detected `phone` is spatially associated with a detected `driver`, optionally reinforced by proximity to `wheel`.

## Workflow

```text
Roboflow COCO dataset
-> COCO to YOLO conversion
-> Dataset validation
-> YOLO fine-tuning on Kaggle GPU
-> Evaluation and benchmark
-> models/best.pt
-> Hugging Face Spaces Gradio demo
```

## Convert Dataset Locally

The source dataset is expected at:

```text
D:\cellphone detector drivers.v1i.coco
```

Convert it to YOLO format on D: because the workspace C: drive may not have enough free space:

```bash
python scripts/convert_coco_to_yolo.py \
  --source "D:\cellphone detector drivers.v1i.coco" \
  --output "D:\cellphone_drivers_yolo"
```

The converter writes:

```text
D:\cellphone_drivers_yolo
  data.yaml
  conversion_report.json
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

## Validate Dataset

```bash
python src/dataset_check.py \
  --data "D:\cellphone_drivers_yolo\data.yaml" \
  --output outputs/dataset_report_cellphone_drivers.json
```

## Train On Kaggle GPU

Use `scripts/kaggle_train.ipynb`. Attach the converted YOLO dataset to the notebook. The notebook finds `data.yaml` under `/kaggle/input`, validates it, trains, evaluates, benchmarks, and zips artifacts.

Inside Kaggle, the cloned repo path is:

```text
/kaggle/working/homeobjects-yolo-detector
```

The notebook calls scripts through absolute paths built from that clone path, for example:

```text
/kaggle/working/homeobjects-yolo-detector/src/train.py
/kaggle/working/homeobjects-yolo-detector/src/evaluate.py
/kaggle/working/homeobjects-yolo-detector/src/benchmark.py
```

The attached YOLO dataset path is auto-detected. It will look like:

```text
/kaggle/input/<converted-yolo-dataset>/data.yaml
```

If the uploaded `data.yaml` still contains a stale local or `/kaggle/working/...` `path:` value, the notebook writes a patched training YAML under:

```text
/kaggle/working/homeobjects-yolo-detector/outputs/cellphone_drivers_kaggle_data.yaml
```

Training and evaluation use that patched YAML so image paths resolve under the attached `/kaggle/input/...` dataset.

Core training command:

```bash
python /kaggle/working/homeobjects-yolo-detector/src/train.py \
  --data /kaggle/input/<converted-yolo-dataset>/data.yaml \
  --model yolo11s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project /kaggle/working/homeobjects-yolo-detector/runs \
  --name cellphone_drivers_yolo
```

`yolo11s.pt` is the default because the `phone` class can be small. Use `yolo11n.pt` if you need a lighter CPU demo.

## Evaluate

```bash
python /kaggle/working/homeobjects-yolo-detector/src/evaluate.py \
  --weights /kaggle/working/homeobjects-yolo-detector/runs/cellphone_drivers_yolo/weights/best.pt \
  --data /kaggle/input/<converted-yolo-dataset>/data.yaml \
  --imgsz 640 \
  --device 0 \
  --split val \
  --output outputs/eval_summary_val.json
```

The Kaggle notebook also evaluates the held-out test split:

```bash
python /kaggle/working/homeobjects-yolo-detector/src/evaluate.py \
  --weights /kaggle/working/homeobjects-yolo-detector/runs/cellphone_drivers_yolo/weights/best.pt \
  --data /kaggle/input/<converted-yolo-dataset>/data.yaml \
  --imgsz 640 \
  --device 0 \
  --split test \
  --output outputs/eval_summary_test.json
```

## Evaluate Rule-Based Behavior Heuristic

The behavior decision is rule-based, not a separately trained behavior model.
YOLO detects objects, then `src/driver_phone_heuristic.py` infers
`driver_using_phone` from the spatial relationship between driver/person,
phone/cellphone, and wheel/steering wheel boxes.

Create a manual image-level label CSV with:

```csv
image,label
img_001.jpg,1
img_002.jpg,0
img_003.jpg,1
img_004.jpg,0
```

Use `configs/driver_phone_usage_labels_template.csv` as a starting point, then
run:

```bash
python src/heuristic_evaluate.py \
  --weights models/best.pt \
  --labels-csv configs/driver_phone_usage_labels_template.csv \
  --images-dir path/to/test/images \
  --imgsz 640 \
  --device cpu \
  --output-json outputs/heuristic_eval_summary.json \
  --output-csv outputs/heuristic_eval_predictions.csv
```

The report includes Accuracy, Precision, Recall, F1-score, and confusion matrix
for the final image-level behavior decision.

## Benchmark

```bash
python /kaggle/working/homeobjects-yolo-detector/src/benchmark.py \
  --weights /kaggle/working/homeobjects-yolo-detector/runs/cellphone_drivers_yolo/weights/best.pt \
  --source /kaggle/input/<converted-yolo-dataset>/images/val \
  --imgsz 640 \
  --device 0 \
  --warmup 20 \
  --runs 500 \
  --save-json outputs/benchmark_summary.json \
  --save-csv outputs/benchmark_results.csv
```

## Deploy Demo

1. Download `runs/cellphone_drivers_yolo/weights/best.pt` from Kaggle.
2. Place it at `models/best.pt`.
3. Push the repo to Hugging Face Spaces.
4. The Space runs `app.py`.

The app runs inference only. It does not train the model.

## Metrics Template

| Model              | Dataset Split | mAP50 | mAP50-95 | Precision | Recall | F1 |
| ------------------ | ------------- | ----: | -------: | --------: | -----: | -: |
| YOLO11s fine-tuned | Validation    |   TBD |      TBD |       TBD |    TBD | TBD |

| Environment        | Device       | Image Size | Batch | Images Tested | Avg Latency | p50 Latency | p95 Latency | FPS |
| ------------------ | ------------ | ---------: | ----: | ------------: | ----------: | ----------: | ----------: | --: |
| Kaggle             | Tesla T4 GPU |        640 |     1 |           TBD |         TBD |         TBD |         TBD | TBD |
| Local              | CPU          |        640 |     1 |           TBD |         TBD |         TBD |         TBD | TBD |
| Hugging Face Space | CPU          |        640 |     1 |   per request |         TBD |           - |           - |   - |

Do not fake metrics. Keep TBD values until measured from a real run.

## Model Card Notes

Intended use:

- Detect driver cellphone-use scenes and related objects in road-safety imagery.
- Demonstrate a reproducible object detection training and deployment pipeline.

Limitations:

- May fail under poor lighting, glare, motion blur, occlusion, or unusual camera angles.
- May miss small phones or confuse phone-like objects with phones.
- Alert quality depends on reliable `driver`, `phone`, and `wheel` detections and the spatial heuristic used to associate a phone with the driver.

Ethical considerations:

- Use as an assistive demo, not as the sole basis for enforcement or safety-critical decisions.
- Avoid collecting or sharing personally identifying driver imagery without permission.

CV bullet:

- Fine-tuned YOLO on a driver cellphone-use dataset, converting Roboflow COCO annotations to YOLO format and deploying a Gradio demo that shows all detections while alerting only for driver cellphone-use.
