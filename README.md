# Driver Cellphone Use YOLO Detector

This project fine-tunes a YOLO detector for driver cellphone-use detection. The source dataset is a Roboflow COCO export and is converted to Ultralytics YOLO format before training.

Classes:

- `Cellphone-in-drivers`
- `driver`
- `phone`
- `wheel`

The Gradio demo displays detections for all four classes, but raises an alert only when `Cellphone-in-drivers` is detected.

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

Use `scripts/kaggle_train.ipynb`. Attach the COCO dataset to the notebook. The notebook will convert the dataset to YOLO inside `/kaggle/working/cellphone_drivers_yolo`, validate it, train, evaluate, benchmark, and zip artifacts.

Inside Kaggle, the cloned repo path is:

```text
/kaggle/working/homeobjects-yolo-detector
```

The notebook calls scripts through absolute paths built from that clone path, for example:

```text
/kaggle/working/homeobjects-yolo-detector/src/train.py
/kaggle/working/homeobjects-yolo-detector/src/evaluate.py
/kaggle/working/homeobjects-yolo-detector/scripts/convert_coco_to_yolo.py
```

Core training command:

```bash
python /kaggle/working/homeobjects-yolo-detector/src/train.py \
  --data /kaggle/working/cellphone_drivers_yolo/data.yaml \
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
  --data /kaggle/working/cellphone_drivers_yolo/data.yaml \
  --imgsz 640 \
  --device 0 \
  --split val \
  --output outputs/eval_summary_val.json
```

The Kaggle notebook also evaluates the held-out test split:

```bash
python /kaggle/working/homeobjects-yolo-detector/src/evaluate.py \
  --weights /kaggle/working/homeobjects-yolo-detector/runs/cellphone_drivers_yolo/weights/best.pt \
  --data /kaggle/working/cellphone_drivers_yolo/data.yaml \
  --imgsz 640 \
  --device 0 \
  --split test \
  --output outputs/eval_summary_test.json
```

## Benchmark

```bash
python /kaggle/working/homeobjects-yolo-detector/src/benchmark.py \
  --weights /kaggle/working/homeobjects-yolo-detector/runs/cellphone_drivers_yolo/weights/best.pt \
  --source /kaggle/working/cellphone_drivers_yolo/images/val \
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
- Alert quality depends on the `Cellphone-in-drivers` class quality in the dataset.

Ethical considerations:

- Use as an assistive demo, not as the sole basis for enforcement or safety-critical decisions.
- Avoid collecting or sharing personally identifying driver imagery without permission.

CV bullet:

- Fine-tuned YOLO on a driver cellphone-use dataset, converting Roboflow COCO annotations to YOLO format and deploying a Gradio demo that shows all detections while alerting only for driver cellphone-use.
