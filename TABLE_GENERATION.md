# Performance Test Results — Table Generation Guide

All results are output in **tab-separated format** for easy copy-paste into spreadsheets or markdown tables.

---

## 1. Model Comparison Table (yolo11n vs yolo26n)
**What:** Inference latency and resource consumption across nano model variants

```bash
echo -e "Model\tMean Latency (ms)\tP95 Latency (ms)\tRSS Mean (MB)" && pytest tests/unit/detection/test_cv_detection.py::TestResourceByModel::test_resource_and_latency_by_model -v -s 2>&1 | grep -E "^(yolo)"
```

**Output Format:**
```
Model                      Mean Latency (ms)    P95 Latency (ms)    RSS Mean (MB)
yolo11n.pt                 44.7                 51.2                534.0
yolo26n.pt                 42.3                 45.6                649.1
```

---

## 2. Scene Entropy Impact Table (yolo26n only)
**What:** How detection performance varies between sparse (low) and dense (high) human scenes

```bash
"pytest tests/unit/detection/test_cv_detection.py::TestResourceByEntropy::test_resource_by_entropy -q -s"
```

**Output Format:**
```
Entropy         Mean Latency (ms)    RSS Mean (MB)
low             44.3                 588.0
high            39.7                 742.9
```

---

## 3. Resolution Scaling Table
**What:** Latency across input resolutions (480p, 720p, 1080p)

```bash
echo -e "Resolution\tEntropy\tMean Latency (ms)\tP95 Latency (ms)" && pytest tests/unit/detection/test_cv_detection.py::TestDetectionLatencyScreening::test_latency_by_resolution_and_entropy -v -s 2>&1 | grep -E "^(480p|720p|1080p)"
```

**Output Format:**
```
Resolution    Entropy      Mean Latency (ms)    P95 Latency (ms)
480p          simple       45.3                 51.7
480p          complex      43.6                 50.1
720p          simple       45.5                 51.2
720p          complex      44.9                 53.8
1080p         simple       44.4                 48.9
1080p         complex      49.0                 57.6
```

---

## 4. Pipeline Stage Breakdown Table
**What:** Time spent in inference, box extraction, and drawing per resolution

```bash
echo -e "Resolution\tInference (ms)\tBox Extract (ms)\tDrawing (ms)\tTotal (ms)" && pytest tests/unit/detection/test_cv_detection.py::TestDetectionStageBreakdown::test_stage_breakdown_on_real_frames -v -s 2>&1 | grep -E "^(480p|720p|1080p)"
```

**Output Format:**
```
Resolution    Inference (ms)    Box Extract (ms)    Drawing (ms)    Total (ms)
480p          43.78             0.08                0.02            43.88
720p          40.53             0.03                0.03            40.59
1080p         42.30             0.03                0.03            42.36
```

---

## Running All Detection Tests

```bash
pytest tests/unit/detection/test_cv_detection.py -v -s
```

This will run:
- ✓ Model comparison (2 models = 2 rows)
- ✓ Scene entropy (2 entropy levels = 2 rows)
- ✓ Resolution scaling (3 resolutions × 2 entropy levels = 6 rows)
- ✓ Stage breakdown (3 resolutions = 3 rows)

**Total: 13 test cases** across ~2–3 minutes runtime
