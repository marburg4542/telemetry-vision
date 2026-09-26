# Codebase Walkthrough — telemetry-vision

**เอกสารสรุปความเข้าใจซอร์สโค้ด** สำหรับผู้ที่กำลังจะเข้ามาพัฒนาต่อ
อ้างอิงจากโค้ดหลังเปลี่ยนหน้าเว็บเป็นวิดีโอที่ประมวลผลด้วย AI · เวอร์ชัน `0.5.0` · Python 3.10–3.12

> เลขบรรทัดในเอกสารนี้เป็นค่าโดยประมาณและจะเลื่อนเมื่อแก้โค้ด
> ใช้ชื่อฟังก์ชันและคลาสค้นหาแทน ส่วนตารางจำนวนบรรทัดในหัวข้อ 2.2
> สร้างจากไฟล์บนดิสก์จริง ไม่ได้พิมพ์มือ

---

## สารบัญ

1. [ภาพรวมและไลบรารีที่ใช้](#1-ภาพรวมและไลบรารีที่ใช้)
2. [โครงสร้างไฟล์](#2-โครงสร้างไฟล์)
3. [เจาะลึกรายฟังก์ชัน](#3-เจาะลึกรายฟังก์ชัน)
4. [ลำดับการทำงานและการไหลของข้อมูล](#4-ลำดับการทำงานและการไหลของข้อมูล)
5. [จุดสังเกตสำหรับการพัฒนาต่อ](#5-จุดสังเกตสำหรับการพัฒนาต่อ)

---

# 1. ภาพรวมและไลบรารีที่ใช้

## 1.1 โปรเจคนี้ทำอะไร

อ่าน **ข้อความ Telemetry ที่ซ้อนอยู่บนภาพวิดีโอโดรน (OSD)** ให้กลับมาเป็นข้อมูลที่คำนวณต่อได้
พร้อมกับ **ตรวจจับยานพาหนะ** ในฉากเดียวกัน แล้วส่งผลลัพธ์ออกทางไฟล์และทาง LINE

ระบบมีสองเส้นทางการใช้งาน

| เส้นทาง | ใช้เมื่อไร | จุดเข้า |
|---|---|---|
| **เว็บ (โต้ตอบ)** | เปิดหน้าเว็บ ดูภาพสด กดบันทึกภาพเอง แล้วส่งเข้า LINE | `telemetry-ocr web` |
| **CLI (แบตช์)** | ประมวลผลไฟล์วิดีโอทั้งไฟล์ หรืออ่านเซสชันที่บันทึกไว้ | `telemetry-ocr run` / `capture` / `extract` |

## 1.2 ไลบรารีหลักและหน้าที่ของแต่ละตัว

### ที่ติดตั้งเสมอ (core dependencies)

| ไลบรารี | เวอร์ชัน | ใช้ทำอะไรในโปรเจคนี้ |
|---|---|---|
| **OpenCV** (`opencv-contrib-python`) | `4.10.0.84` (ตรึงเวอร์ชัน) | ถอดรหัสวิดีโอและ RTSP · ครอป ROI · ปรับปรุงภาพ (CLAHE, unsharp, threshold, morphology) · เข้ารหัส JPEG · วาดกรอบตรวจจับ · ย่อภาพ thumbnail |
| **NumPy** | `>=1.26,<3` | รูปแบบข้อมูลของภาพทุกภาพในระบบคือ NumPy array (BGR uint8) |
| **PyYAML** | `>=6,<7` | อ่าน/เขียนไฟล์ตั้งค่า ROI (`configs/*.yaml`) |

> **ตรึงเวอร์ชัน OpenCV ไว้ตายตัวโดยตั้งใจ** เพราะ PaddleOCR ต้องการ `opencv-contrib-python`
> ส่วน Ultralytics ประกาศ `opencv-python` ถ้าเวอร์ชันสองตัวนี้ไม่ตรงกัน pip จะสลับไปมาแล้วพัง

### Extras (ติดตั้งตามการใช้งาน)

| Extra | ไลบรารี | ใช้ทำอะไร |
|---|---|---|
| `ocr` | **PaddleOCR** `>=3.7,<3.8` | รู้จำข้อความ ใช้เฉพาะโมดูล **recognition** (`PP-OCRv6_tiny_rec`) ไม่ใช้ส่วน text detection |
| `web` | **aiohttp** `>=3.10,<4` | เว็บเซิร์ฟเวอร์แบบ async สำหรับ REST API และส่งไฟล์วิดีโอแบบ Range |
| `ai` | **Ultralytics** `>=8.3,<9` | YOLOv8 ตรวจจับยานพาหนะ + ติดตามด้วย BoT-SORT |
| `ai` | **dill**, **lap** | dill = serialize ของ Ultralytics · lap = Linear Assignment Problem สำหรับ tracker |
| `report` | **pypdf**, **reportlab** | สร้างรายงาน PDF (อยู่ในแพ็กเกจ `vision_lab`) |
| `test` / `lint` | pytest, ruff, mypy | ชุดตรวจคุณภาพ |

### ข้อจำกัดที่กำหนดสถาปัตยกรรมทั้งระบบ

> **PaddlePaddle กับ PyTorch อยู่ในโปรเซสเดียวกันไม่ได้**
>
> ทั้งสองไลบรารีแนบ **cuDNN 9** มาคนละชุด บน Windows ตัวที่โหลดทีหลังจะล้มด้วย `WinError 127`
>
> ผลคือ **`telemetry-ocr` (ฝั่ง OCR) ห้าม import torch เด็ดขาด** งาน YOLO ทั้งหมดจึงถูกผลักไปอยู่ใน
> **โปรเซสลูกที่ spawn ออกไป** — [render_job.py](src/telemetry_ocr/render_job.py) สำหรับหน้าเว็บ
> และ [detection_worker.py](src/telemetry_ocr/detection_worker.py) สำหรับ CLI `run --detect-model`
> — หรือรันแยกภายใต้คำสั่ง `vision-lab`
>
> โค้ดที่บังคับเรื่องนี้อยู่ที่ [`_hide_torch_from_this_process()`](src/telemetry_ocr/ocr.py) — ocr.py:31

---

# 2. โครงสร้างไฟล์

## 2.1 จุดเข้าโปรแกรม (Entry Points)

ประกาศไว้ใน `pyproject.toml` → `[project.scripts]`

```
telemetry-ocr  →  telemetry_ocr.cli:main      ← ตัวหลัก (ห้ามมี torch)
vision-lab     →  vision_lab.cli:main         ← เครื่องมือฝั่ง dataset/training (มี torch ได้)
```

เรียกผ่านโมดูลก็ได้: `python -m telemetry_ocr` → [`__main__.py`](src/telemetry_ocr/__main__.py) เรียก `cli.main()`

**คำสั่งย่อยของ `telemetry-ocr`** (ประกาศใน [cli.py:17 `build_parser`](src/telemetry_ocr/cli.py))

| คำสั่ง | หน้าที่ |
|---|---|
| `web` | เปิดเว็บ UI (จุดเข้าที่ใช้บ่อยที่สุด) |
| `run` | ประมวลผลไฟล์หรือสตรีมแบบเรียลไทม์ เขียน JSONL/CSV |
| `capture` | บันทึกเฟรม + manifest ลงโฟลเดอร์เซสชัน (ยังไม่ OCR) |
| `extract` | อ่าน Telemetry จากเซสชันที่บันทึกไว้ |
| `session-info` | สรุปข้อมูลเซสชัน |
| `calibrate` | เลือกสูตรปรับปรุงภาพที่ดีที่สุดของแต่ละ ROI |
| `export-crops` | ส่งออกภาพครอป ROI ไปกำกับค่าอ้างอิงด้วยมือ |
| `preview-rois` | วาดกรอบ ROI ลงบนเฟรมตัวอย่างเพื่อตรวจตำแหน่ง |
| `validate-config` | ตรวจไฟล์ตั้งค่าอย่างเดียว |
| `benchmark` | วัดจำนวนเธรด CPU ที่เหมาะกับ PaddleOCR |

## 2.2 แผนผังไฟล์และความรับผิดชอบ

```
src/telemetry_ocr/                       (26 ไฟล์ · 9,793 บรรทัด)
│
├─ __main__.py            5    เรียก cli.main()
├─ cli.py               522    ★ Entry point · argparse · เชื่อมทุกคำสั่งเข้ากับโมดูล
├─ errors.py             19    ชนิด exception ของโปรเจค
│
├── ชั้นตั้งค่า ──────────────────────────────────────────────
├─ config.py            342    dataclass ของ ROI/runtime/temporal · โหลด·ตรวจ·เขียน YAML
│
├── ชั้นประมวลผลภาพและ OCR ────────────────────────────────
├─ preprocess.py        180    ครอป ROI · ใช้สูตรปรับปรุงภาพ (channel/CLAHE/threshold)
├─ ocr.py               167    ห่อ PaddleOCR · ซ่อน torch · recognizer หลายตัวขนานกัน
├─ validation.py        357    ★ ทำให้เป็นมาตรฐาน · regex · พิสัยค่า · ซ่อมข้อความที่อ่านผิด
├─ temporal.py          149    โหวตข้ามเฟรม (fast 2/3 · stable 3/5)
├─ pipeline.py          736    ★ หัวใจ: VideoDecoder + TelemetryPipeline
├─ output.py            107    TelemetryRecord + เขียน JSONL/CSV
│
├── ชั้นตรวจจับวัตถุ (แยกโปรเซส) ────────────────────────────
├─ detection.py         267    ห่อ Ultralytics YOLO · เลือก device · แปลงผลเป็น Detection
├─ render_job.py      1,137    ★ รันวิดีโอทั้งไฟล์ผ่าน YOLO+BoT-SORT · เขียนกรอบลง tracks.jsonl · ทำต่อได้
├─ render_export.py     463    วาดกรอบลงต้นฉบับทุกเฟรมเป็น MP4 ไฟล์เดียว (เต็ม / 720p) ไว้ส่งต่อ
├─ auto_capture.py      372    เลือกวัตถุจาก tracks ระหว่างประมวลผล → Capture/ส่งไลน์อัตโนมัติ
├─ geocode.py          507    พิกัด TAG → ที่อยู่จริง (Google Geocoding) สำหรับข้อความไลน์
├─ training_export.py   385    เลือกเฟรมจาก render ไปทำ label เทรนโมเดล (YOLOv8 dataset)
├─ annotate.py           97    วาดกรอบลงสำเนาภาพ (ใช้ร่วม capture กับ render ไม่ import OCR)
├─ detection_worker.py  527    โปรเซสลูก + คิวรับส่งเฟรม/ผล (ใช้กับ CLI run --detect-model)
├─ detect_sidecar.py    162    ตัวช่วยฝั่ง CLI สำหรับรัน detection คู่กับ run
│
├── ชั้นเซสชัน/บันทึกภาพ ──────────────────────────────────
├─ capture.py           408    รูปแบบเซสชันบนดิสก์ · เขียน/อ่าน manifest
├─ capture_service.py   818    ★ CaptureSession · ensemble · วาดกรอบ · ตรวจสอบซ้ำ
├─ extract.py           168    OCR ย้อนหลังจากเซสชันที่บันทึกไว้
│
├── ชั้นเว็บ ─────────────────────────────────────
├─ video_library.py    155    รายการวิดีโอในโฟลเดอร์ · อ่านเฟรมต้นฉบับทีละเฟรม
├─ webapp.py          1,761    ★ ใหญ่ที่สุด: aiohttp app · 33 routes · middleware · state
├─ file_response.py     47    ส่งไฟล์เกิน 2 GB เป็นช่วง (Windows ส่งทีละไม่เกิน 2 GiB)
├─ public_share.py     207    เผยแพร่ภาพชั่วคราวด้วย token (ให้ LINE ดึงได้)
├─ quick_tunnel.py     262    คุม cloudflared quick tunnel
├─ line_client.py      750    จัดรูปแบบข้อความ · แปลง DMS · push เข้า LINE
│
├── เครื่องมือเสริม ────────────────────────────────────────
├─ calibration.py      387    กวาดสูตรปรับปรุงภาพทุกแบบ เลือกที่ดีที่สุด
├─ preview.py           79    วาดกรอบ ROI ลงภาพเพื่อตรวจตำแหน่ง
│
└── web_static/                 (3,661 บรรทัด)
   ├─ index.html       255    โครงหน้า
   ├─ app.js         1,998    ★ ตรรกะหน้าเว็บทั้งหมด
   ├─ app.css          853    สไตล์
   ├─ track-overlay.js  565   ★ วาดกรอบของ render ทับวิดีโอต้นฉบับทุกเฟรม · รอ tracker
   ├─ format.js          43    จัดรูปแบบพิกัด DMS (แชร์กับเทสต์ Node)
   └─ service-worker.js  85   PWA offline shell

src/vision_lab/                          (10 ไฟล์ · 3,139 บรรทัด) — แยกจากกันโดยสิ้นเชิง
├─ cli.py              185    Entry point ที่สอง (`vision-lab`)
├─ report.py           968    สร้างรายงาน PDF ภาษาไทย
├─ handbook.py         868    สร้างคู่มือ
├─ evaluate.py         273    ประเมินโมเดล YOLO (mAP ฯลฯ)
├─ paths.py            229    ทะเบียนที่อยู่ของ dataset/checkpoint (+ overlay ของหน้าเว็บ)
├─ model_registry.py   251    เพิ่ม/เอาออก/ตั้งค่าเริ่มต้นโมเดลจากหน้าเว็บ (models.local.yaml)
├─ envcheck.py         146    ตรวจสภาพแวดล้อม
├─ frames.py           109    สกัดเฟรมไปทำ dataset
└─ torch_env.py        101    จัดการ torch (โมดูลนี้ import torch ได้)
```

## 2.3 ไฟล์ที่ควรอ่านก่อนถ้ามีเวลาจำกัด

1. **[config.py](src/telemetry_ocr/config.py)** — เข้าใจโครงข้อมูลก่อน ทุกอย่างหมุนรอบ `FieldConfig`
2. **[pipeline.py](src/telemetry_ocr/pipeline.py)** — `_process_frame_locked` คือหัวใจของการอ่านหนึ่งเฟรม
3. **[validation.py](src/telemetry_ocr/validation.py)** — ตรรกะที่ตัดสินว่าค่าไหนเชื่อได้
4. **[webapp.py](src/telemetry_ocr/webapp.py)** — ถ้าจะแก้ฝั่งเว็บ

---

# 3. เจาะลึกรายฟังก์ชัน

## 3.1 `config.py` — โครงสร้างข้อมูลกลาง

### `RecipeConfig` (dataclass, บรรทัด 22)
สูตรปรับปรุงภาพหนึ่งชุด: `name`, `channel`, `scale`, `clahe`, `unsharp`, `morphology`, `threshold`, `invert`

### `FieldConfig` (dataclass, บรรทัด 40)
นิยามของ ROI หนึ่งช่อง — **โครงสร้างที่สำคัญที่สุดในโปรเจค**

| ฟิลด์ | ความหมาย |
|---|---|
| `field_id` | ชื่อฟิลด์ เช่น `tag_lat_dms` |
| `box` | `(x, y, w, h)` แบบนอร์มัลไลซ์ 0–1 |
| `regex` | รูปแบบที่ค่าต้องตรง |
| `data_type` | `string` / `integer` / `float` / `dms` |
| `normalizer` | `none` / `numeric` / `dms` / `optics` |
| `min_value`, `max_value` | พิสัยที่ยอมรับ |
| `decimals`, `dms_decimal_places`, `dms_degrees_digits` | ความละเอียด |
| `profile` | `fast` หรือ `stable` (ใช้เลือกหน้าต่างโหวต) |
| `padding` | ระยะขยาย ROI (None = ใช้ค่า runtime) |
| `ensemble_override` | ยอมให้ ensemble ทับค่าเดิมได้หรือไม่ |
| `derive_from_source_time` | คำนวณจากเวลาวิดีโอแทนการ OCR (ใช้กับนาฬิกา) |
| `primary`, `fallback` | สูตรปรับปรุงภาพหลักและสำรอง |

### `load_config(path) -> AppConfig` (บรรทัด 191)
- **Input:** path ของไฟล์ YAML
- **ทำอะไร:** อ่าน YAML → `parse_config` → `validate_config`
- **Output:** `AppConfig` ที่ผ่านการตรวจแล้ว
- **โยน:** `ConfigError` เมื่อค่าผิด เช่น box เกินขอบภาพ, regex ไม่สอดคล้อง `data_type`

### `validate_config(config)` (บรรทัด 224)
ตรวจกฎที่จับข้อผิดพลาดตอนตั้งค่าได้ก่อนจะไปพังตอนรัน เช่น `derive_from_source_time`
ต้องคู่กับ regex แบบ `HH:MM:SS` เท่านั้น (บรรทัด 304–311)

---

## 3.2 `preprocess.py` — เตรียมภาพให้ OCR

### `extract_roi(frame, box, padding) -> ndarray` (บรรทัด 20)
- **Input:** เฟรม BGR, กล่องนอร์มัลไลซ์, padding (สัดส่วนของขนาด ROI)
- **ทำอะไร:** แปลงพิกัด 0–1 เป็นพิกเซล ขยายด้วย padding แล้วครอป
- **Output:** ภาพย่อย (NumPy array)

### `apply_recipe(crop, recipe) -> ndarray` (บรรทัด 39)
- **Input:** ภาพครอป + `RecipeConfig`
- **ทำอะไร:** เรียงลำดับตายตัว 7 ขั้น
  1. `cv2.resize` ด้วย `INTER_CUBIC` ตาม `scale`
  2. `_select_channel` เลือกช่องสัญญาณ (`color`/`gray`/`b`/`g`/`r`/`lab_*`/`hsv_s`/`hsv_v`)
  3. CLAHE (ถ้าเปิด)
  4. Unsharp mask (Gaussian blur แล้ว `addWeighted`)
  5. Morphology (tophat/blackhat)
  6. Threshold (otsu/adaptive)
  7. Invert
- **Output:** ภาพ **BGR 3 ช่อง เสมอ** (แปลงกลับถ้ากลายเป็น grayscale) เพราะ PaddleOCR ต้องการ 3 ช่อง

---

## 3.3 `ocr.py` — ห่อ PaddleOCR

### `_hide_torch_from_this_process()` (บรรทัด 31)
- **ทำอะไร:** ใส่ตัวขวางลงใน `sys.modules` เพื่อให้การ `import torch` ในโปรเซสนี้ล้มทันที
- **ทำไม:** กัน cuDNN ชนกัน — ถ้าไม่ทำ จะพังตอนรันแบบหาสาเหตุยาก

### `PaddleRecognizer.recognize(images) -> list[OCRResult]` (บรรทัด 85)
- **Input:** ลำดับภาพ (แต่ละภาพคือ ROI ที่ปรับปรุงแล้ว)
- **ทำอะไร:** ส่งเข้า `PP-OCRv6_tiny_rec` เป็นชุดเดียว
- **Output:** `OCRResult(text, confidence)` เรียงตรงกับลำดับ input

### `create_recognizer(runtime) -> Recognizer` (บรรทัด 139)
- คืน `PaddleRecognizer` ตัวเดียว หรือ `ParallelRecognizer` ที่มีหลายตัว
  ตาม `runtime.recognizer_workers`

---

## 3.4 `validation.py` — ตัดสินว่าค่าไหนเชื่อได้ ★

โมดูลนี้คือเกราะกัน **"ค่าที่ผิดแต่ดูสมเหตุสมผล"** ควรอ่านให้ละเอียดที่สุด

### `validate_text(raw_text, config) -> ValidatedValue` (บรรทัด 36)
- **Input:** ข้อความดิบจาก OCR + `FieldConfig`
- **ทำอะไร:** สองรอบ
  1. รอบแรก: `_normalize` (ตัดช่องว่าง) → `_normalize_for_field` (ตามชนิด normalizer) → `_validate_candidate`
  2. ถ้าไม่ผ่าน: ใช้ `CONFUSION_MAP` (`O`→`0`, `I`→`1`, `S`→`5`, `B`→`8` …) แล้วลองใหม่
- **Output:** `ValidatedValue(raw_text, normalized_text, value, valid, corrected, reason)`

### `_normalize_for_field(text, config)` (บรรทัด 54)
แยกตาม `normalizer` สี่แบบ

| normalizer | ทำอะไร |
|---|---|
| `none` | ส่งคืนตามเดิม |
| `numeric` | แปลงตัวคั่นทศนิยม (`,` `:` `。` `．` `：` เต็มความกว้าง) · วางจุดตามจำนวนทศนิยมที่กำหนด · **ปฏิเสธถ้าเห็นจุดแต่แยกตัวเลขไม่ได้** |
| `dms` | นับจำนวนหลักตามผัง องศา-ลิปดา-ฟิลิปดา · ตัดหลักเกินที่เกิดจากอ่านเครื่องหมาย `"` เป็นเลข 2 |
| `optics` | ส่งต่อให้ `_normalize_optics` |

> **จุดที่ต้องเข้าใจ (บรรทัด ~102–110):** ถ้าข้อความมีจุดทศนิยมอยู่แต่ regex ดึงตัวเลขไม่ได้
> ระบบจะ **ปฏิเสธทั้งเฟรม** แทนที่จะเลื่อนจุด เพราะการเลื่อนจุดเคยทำให้ `646.0` กลายเป็น
> `64.6` ซึ่งผ่านการตรวจทุกด่านและผิดสิบเท่า

### `_normalize_optics(text, config)` (บรรทัด 165)
ประกอบข้อความ EO/IR ขึ้นใหม่จากส่วนย่อย แทนการซ่อมในที่
1. ซ่อมคำว่า `FOV` ที่อ่านเพี้ยน (`F0V`, `FQv`, `FDV`, `ROV`)
2. แยกด้วย `FOV` เป็นส่วนซูมกับส่วนมุมมอง
3. เรียก `_optics_number` กับแต่ละส่วน

### `_optics_number(part) -> str | None` (บรรทัด 197)
**หลักการสำคัญ:** ทุกตัวเลขใน OSD ลงท้ายด้วยเครื่องหมาย (`x` หรือ `°`) ซึ่งเป็น glyph เดียวที่ไม่ใช่ตัวเลข
จึงเป็นตัวที่ OCR อ่านผิดบ่อยที่สุด
- ถ้าตัวท้ายไม่ใช่ตัวเลข → ถือเป็นเครื่องหมาย ตัดทิ้ง
- ถ้าตัวท้ายเป็นตัวเลข **และไม่มีจุดทศนิยม** → เครื่องหมายถูกอ่านเป็นตัวเลข ตัดทิ้ง
- ถ้ามีจุดทศนิยมอยู่แล้ว → เก็บทุกหลัก (ไม่งั้น `19.5` จะกลายเป็น `1.9`)

### `_dms_to_decimal(candidate, config) -> float` (บรรทัด 328)
- **Input:** ข้อความ DMS มาตรฐาน เช่น `14°50'57.37"`
- **ทำอะไร:** `degrees + minutes/60 + seconds/3600` — **ความละเอียด float เต็ม ไม่มีการปัด**
- **Output:** องศาทศนิยม (ติดลบถ้า hemisphere เป็น S/W)

---

## 3.5 `temporal.py` — โหวตข้ามเฟรม

### `FieldStabilizer.add(...) -> StabilizedValue` (บรรทัด 40)
- **Input:** ผลการอ่านของเฟรมปัจจุบัน
- **ทำอะไร:** เก็บลง `deque` ขนาดเท่าหน้าต่าง (`fast`=3, `stable`=5)
  แล้วดูว่ามีค่าเดียวกันซ้ำถึงเกณฑ์ (`fast`=2, `stable`=3) หรือยัง
  - ค่าที่ไม่ผ่าน validation **ก็กินที่ในหน้าต่างด้วย** เพื่อให้ค่าเก่าหมดอายุตามเวลาจริง
  - `_passes_delta` กันค่าที่กระโดดเกิน `max_delta_per_second`
- **Output:** `StabilizedValue` พร้อมธง `confirmed`

> **ผลข้างเคียงที่ต้องรู้:** ค่าที่เผยแพร่จะ **ล้าหลังภาพเล็กน้อยโดยตั้งใจ** วัดบนสตรีมจริง
> ข้อความ OCR ตรงกัน 10/11 ฟิลด์ ขณะที่ค่าที่เผยแพร่ต่าง 8/11 — ไม่ใช่บั๊ก

---

## 3.6 `pipeline.py` — หัวใจของระบบ ★

### `VideoDecoder` (คลาส, บรรทัด 106)
เธรดอ่านเฟรมแยกต่างหาก

| เมธอด | หน้าที่ |
|---|---|
| `_open` (235) | เปิด `cv2.VideoCapture` ตั้ง `CAP_PROP_OPEN_TIMEOUT_MSEC` / `READ_TIMEOUT_MSEC` = 10 วิ |
| `_prefer_tcp_for_rtsp` (272) | บังคับ RTSP over TCP ผ่าน env `OPENCV_FFMPEG_CAPTURE_OPTIONS` |
| `_source_time_ms` (297) | คำนวณตำแหน่งเวลาในวิดีโอ |
| `_reconnect` (314) | ต่อใหม่เมื่อสตรีมหลุด |
| `_put_latest` (335) | ใส่คิวขนาด 1 แบบ **ทิ้งของเก่า** (`drop_when_full`) |

> OpenCV รุ่นนี้ **ไม่สนใจ** ตัวเลือก timeout ของ FFmpeg (`stimeout`, `rw_timeout`)
> ทางเดียวที่ได้ผลคือ `CAP_PROP_*_TIMEOUT_MSEC` — ก่อนแก้เคยค้าง 4 นาทีเมื่อกล้องเงียบ

### `TelemetryPipeline.run(...) -> PipelineSummary` (บรรทัด 390)
- **Input:** `video_path`, `jsonl_path`, `csv_path` + `max_seconds`, `on_frame`, `stop_event`
- **ทำอะไร:** ลูปหลักของโหมด `run` — ดึงเฟรมจาก decoder ตามอัตรา `ocr_hz`
  เรียก `process_frame` แล้วเขียนผลผ่าน `RecordWriter`
  - `on_frame` = ช่องให้ฝั่ง detection รับเฟรมเดียวกันไป **โดยที่โมดูลนี้ไม่ต้องรู้จัก torch**
  - `stop_event` = ให้ผู้เรียกที่ไม่ใช่คอนโซลสั่งหยุดได้
- **Output:** `PipelineSummary` (จำนวนเฟรม, latency, ฯลฯ)

### `TelemetryPipeline.process_frame(packet)` (บรรทัด 558)
ห่อ `_process_frame_locked` ด้วย **`_RECOGNIZER_LOCK`** (บรรทัด 27)
เพราะ PaddleOCR ไม่รับประกัน thread-safety และมีอย่างน้อยสองเธรดที่อยากใช้ (การบันทึกภาพ + การตรวจซ้ำจากไฟล์ภาพ)

### `TelemetryPipeline._process_frame_locked(packet)` (บรรทัด 569) — **อ่านฟังก์ชันนี้ให้เข้าใจ**
- **Input:** `FramePacket(frame, frame_index, source_time_ms, received_at)`
- **ทำอะไร:** ตามลำดับ
  1. ฟิลด์ที่ `derive_from_source_time` และมี anchor แล้ว → คำนวณเวลา ไม่ต้อง OCR (บรรทัด 574)
  2. ฟิลด์ที่ `ocr_interval > 1` และยังไม่ถึงคิว → ใช้ค่าเดิม (`held_last_ocr`)
  3. ครอป ROI + `apply_recipe` ด้วย **สูตรหลัก** ทุกฟิลด์ → รวมเป็นชุดเดียว
  4. `recognize_grouped` (บรรทัด 357) จัดกลุ่มตามอัตราส่วนภาพก่อนส่งเข้าโมเดล
  5. `validate_text` ทีละฟิลด์
  6. ฟิลด์ที่ไม่ผ่าน **หรือ** conf < `confidence_threshold` → ทำซ้ำด้วย **สูตรสำรอง** แล้ว `prefer_candidate`
  7. ส่งเข้า `TemporalStabilizer`
- **Output:** `(list[TelemetryRecord], latency_ms)`

---

## 3.7 `render_job.py` — ประมวลผลวิดีโอทั้งไฟล์ ★

### `run_render(video, output_dir, settings, meta, detector, publish, cancelled, fresh)`
**ฟังก์ชันที่รันในโปรเซสลูก** (หรือใน thread ตอนเทสต์) — ที่เดียวของหน้าเว็บที่แตะ torch
- **ทำอะไร:** ต่อเฟรม `grab()` → ข้ามตาม stride → `retrieve()` → `detector.track()` (BoT-SORT, persist)
  → ต่อท้าย `tracks.jsonl` (flush ทุกเฟรม เพราะหน้าเว็บและ Capture อ่านไฟล์นี้ระหว่างประมวลผล)
  ไม่วาดและไม่เข้ารหัสวิดีโอ — หน้าเว็บวาดกรอบเอง ส่วน MP4 วาดตอนบันทึก
- ส่ง `started` (มี `start_frame`) แล้ว `progress` ทุก 0.5 วิ พร้อม `stages_ms` (decode / track)
- **ทำต่อ:** ผล `partial` (หรือ `running` ที่ process ตาย) ที่ fingerprint และค่าตรงกัน ทำต่อจากเฟรมถัดจาก
  บรรทัดสมบูรณ์สุดท้ายใน `tracks.jsonl` (`last_frame()` อ่านจากท้ายไฟล์) ตัดบรรทัดที่เขียนค้างทิ้ง บวก
  offset ให้ Track ID ผลแบบ `hls` รุ่นก่อนทำต่อได้เหมือนกัน (format กลายเป็น `tracks`)
- **Output:** `tracks.jsonl`, `tracker.yaml`, `meta.json` (format `tracks`)
- `settle_render(dir)` เรียกทุกครั้งที่จบแบบไม่สมบูรณ์ — ถ้ามีเฟรมแล้วเป็น `partial` พร้อม `next_frame`
  ถ้ายังไม่มีเลยลบทั้งโฟลเดอร์
- `render_state(dir, meta)` — `processed_ms` = (เฟรมสุดท้าย + stride) ÷ fps หรือความยาวทั้งไฟล์เมื่อเสร็จ

### `RenderManager` (คลาส) — ฝั่ง server ไม่ import torch

| เมธอด | ทำอะไร |
|---|---|
| `start(...)` | spawn ลูกผ่าน `spawn_launcher` · งานซ้อนได้ `RenderBusy` (ไม่ต้องมี ffmpeg) |
| `status()` / `cancel()` | อ่านคิวความคืบหน้า · ตั้ง event ให้ลูกหยุด |
| `renders(video_id)` / `delete(render_id)` | ผลบนดิสก์ทุก format ที่มี `tracks.jsonl` (พร้อม `processed_ms` และ `size_bytes` ของทั้งโฟลเดอร์) · ลบผล (ยกเว้นงานที่กำลังรัน) |
| `tracks(render_id, from_ms, to_ms)` | สถานะ + `rate` + กรอบของช่วงเวลานั้นแบบกะทัดรัดผ่าน `TrackIndex.between()` (bisect บน offset) |
| `detections_at(render_id, frame_index)` | กรอบของเฟรมหนึ่งจาก `tracks.jsonl` ผ่าน `TrackIndex` (byte offset) |

### `render_export.py` — บันทึกผลเป็น MP4 ไฟล์เดียว

ผลคือกรอบ ไม่ใช่วิดีโอ ไฟล์นี้วาดกรอบลงต้นฉบับทุกเฟรมแล้วเข้ารหัสเป็นไฟล์เดียวไว้ส่งต่อ

| ส่วน | ทำอะไร |
|---|---|
| `boxes_at(frame, frames, boxes, stride)` | กรอบของเฟรมใดก็ได้: เฟรมที่ประมวลผล = กรอบของมัน · ระหว่างสองเฟรม = เลื่อนตามเส้นตรง (Track ID เดียวกัน) ฝั่งเดียว = แสดงครึ่งที่ใกล้ · เลยเฟรมสุดท้าย = ค้างไว้ไม่เกิน stride · กติกาเดียวกับ `boxesAt` ใน `track-overlay.js` |
| `encoder_args(quality, encoder)` | `full` = ค่าของ encoder · `small` = ย่อ 720p (ไม่ขยาย) + ค่าคุณภาพสำหรับแชท |
| `Mp4Writer` | raw BGR ทาง stdin → MP4 `+faststart` |
| `export_filename(meta, quality, percent)` | `<วิดีโอ>__<โมเดล>__AI[-720p][_partial-N%].mp4` |
| `ExportManager.start(render_id, quality)` | ต้องมี ffmpeg, เฟรมอย่างน้อยหนึ่งเฟรม และไฟล์ต้นฉบับ (`video_path` จาก VideoLibrary) · ทีละไฟล์ (`ExportBusy`) · ทำใน thread ใน `output/exports/<id 32 hex>/` |
| `status(id)` / `ready_file(id)` / `finished(id)` | ความคืบหน้าตามเฟรมที่เขียน · path ของไฟล์ที่เสร็จ · ลบหลังส่งครบ |

### `tracker_config(directory, confidence)`
อ่าน [configs/botsort.yaml](configs/botsort.yaml) แล้วตั้ง `track_high_thresh` = confidence และ
`new_track_thresh` = confidence + 0.15 (`NEW_TRACK_MARGIN`) เขียนเป็น `tracker.yaml` ในโฟลเดอร์ของ render
แล้วคืน path นั้น (ไม่ส่งโฟลเดอร์ = คืนไฟล์กลาง) ส่วน `UltralyticsDetector.track()` ส่งกรอบตั้งแต่
`TRACKER_FLOOR` (0.1) ให้ tracker เพื่อเปิดขั้นจับคู่รอบสอง แต่คืนเฉพาะกรอบที่ ≥ confidence และย่อภาพของ
GMC เป็น 1/4 (`GMC_DOWNSCALE`) หลังเฟรมแรกที่ tracker ถูกสร้าง ตัวเลขที่ใช้เลือกค่าอยู่ใน README

### `auto_capture.py` — Capture / ส่งเข้าไลน์อัตโนมัติ

| ส่วน | ทำอะไร |
|---|---|
| `AutoRules.from_request(body)` | ตรวจกฎจากหน้าเว็บ: `classes` (label → ความมั่นใจขั้นต่ำ), `send`, `reserve` (10), `min_seconds` (1) |
| `Selector.feed(frame, time_ms, detections)` | ต่อ Track ID: ช่วงที่เห็นต่อเนื่อง (ขาดได้ 0.5 วิ) ครบ `min_seconds` → `Candidate` ของเฟรมที่มั่นใจที่สุด · เลขใหม่ประเภทเดียวกันที่เริ่มภายใน 3 วิหลังเลขที่ settle แล้วหายไป และห่างไม่เกิน 1.5 เท่าขนาดกรอบ = คันเดิม (`same_vehicle`) |
| `AutoLineRun.run()` | อ่านบรรทัดใหม่ของ `tracks.jsonl` ทุก 1 วิ (เฉพาะบรรทัดสมบูรณ์) → capture → send · หยุดเมื่อ render ไม่ active แล้วอ่านรอบสุดท้าย · quota ต่ำกว่า reserve → หยุดส่งแต่ยัง capture |

ใน `webapp.py`: `take_capture()` และ `deliver()` แยกออกมาจาก handler ของปุ่ม Capture/ส่ง ให้ทั้งสองทางใช้
ร่วมกัน `start_render` รับ `auto` แล้ว `start_auto()` สร้าง `AutoLineRun` (pipeline OCR ของตัวเอง) ·
`GET /api/renders/auto` คืนสถานะ ภาพอัตโนมัติมี `origin: "auto"` และ `note` ใน `frames.jsonl`

### `training_export.py` — เลือกเฟรมเทรนและส่งออก label ร่าง (`telemetry-ocr training-frames`)

| ส่วน | ทำอะไร |
|---|---|
| `load_flights(renders, videos)` | render ที่ใหม่ที่สุดของแต่ละวิดีโอ พร้อมกรอบทุกเฟรมที่ประมวลผล |
| `load_feedback(sessions)` | ภาพที่ผู้ใช้แจ้ง (`feedback.jsonl` ของแต่ละ session ใช้ค่าล่าสุดต่อภาพ) |
| `select(...)` | แจ้งไว้ทั้งหมด → มอเตอร์ไซค์ 35% → รถได้เลขใหม่ 20% → กระจายทั่วเที่ยวบิน · ห่างกัน ≥ `spacing_s` · วิดีโอ holdout → test |
| `export(...)` | อ่านวิดีโอไปข้างหน้า เขียน `train|test/images|labels`, `data.yaml`, `manifest.csv` · label ร่างวางด้วย `boxes_at` |

`scripts/compare_models.py` เทียบ checkpoint บนชุดทดสอบ (`YOLO.val`) และบนวิดีโอจริงด้วย
`UltralyticsDetector` + `tracker_config` ของแอป
`scripts/train_local.py` เทรนบน GPU เครื่องนี้จาก 14th_opt และ/หรือ yolov8m-visdrone (สูตร GTX 1650: batch 2, ปิด AMP)

### `geocode.py` — พิกัด TAG → ถนน/ตำบล/หมุดอ้างอิง สำหรับข้อความไลน์

| ส่วน | ทำอะไร |
|---|---|
| `load_geocode_config()` | คีฯจาก env ก่อน แล้วค่อย `configs/line.json` · ไม่มีคีฯ = ปิดคุณสมบัตินี้เงียบๆ |
| `Geocoder.place()` | cache บนดิสก์ต่อพิกัดปัดเศษ 4 ตำแหน่ง (~11 ม.) · เพดานวันละ 500 ครั้ง · คำตอบที่ผิดพลาดไม่ถูกจำ |
| `parse_response()` | อ่าน address_components หลายชั้น (ที่ไทยคืนไม่คงที่) และหยิบสถานที่ในคำตอบเดิมมาเป็นหมุดอ้างอิง |
| `Place.lines()` | สองหรือสามบรรทัดใต้ลิงก์แผนที่ · ไม่มีชื่อถนนก็บอกว่าไม่มี |

`webapp.deliver()` เรียก `places_for(shots)` ก่อนสร้างข้อความ จึงใช้ทั้งกดส่งเองและส่งอัตโนมัติ
พิกัดมาจาก `line_client.tag_position(shot)` ที่อ่านค่าเดียวกันกับลิงก์แผนที่

### `choose_encoder(ffmpeg)`
ลอง `h264_nvenc` → `h264_qsv` → `libx264` ด้วยการ encode จริงสองเฟรม ใช้ตอนบันทึก MP4 เท่านั้น
(x264 แย่ง CPU 4 คอร์กับงานอื่น NVENC ใช้วงจรเข้ารหัสเฉพาะบนการ์ดจอ)

### `video_library.py` — `VideoLibrary` · `SourceFrames`
รายการวิดีโอใต้ `video_dir` (ค่าเริ่มต้น `videos/` ในโปรเจกต์ · env `TELEMETRY_OCR_VIDEOS`)
อ้างด้วย path สัมพัทธ์ (ไม่รับ path จาก request ตรงๆ) และอ่าน
เฟรมต้นฉบับทีละเฟรมให้ Capture โดยเปิดไฟล์ค้างไว้ ข้ามไปข้างหน้าไม่เกิน 30 เฟรมด้วยการอ่านต่อแทนการ seek

## 3.7b `detection_worker.py` — worker แบบสดสำหรับ CLI

หน้าเว็บเลิกใช้แล้ว แต่ `telemetry-ocr run --detect-model` (ผ่าน `detect_sidecar.py`) ยังใช้
`ProcessDetectionWorker` ส่งเฟรมเข้าคิว newest-wins ขนาด 1 ให้ลูกประมวลผลคู่กับ OCR แบบ realtime
`put_latest` ทิ้งเฟรมเก่าเมื่อคิวเต็ม เพื่อไม่ให้ผลตรวจจับล้าหลังภาพ

---

## 3.8 `capture_service.py` — บันทึกภาพและตรวจสอบซ้ำ ★

### `CaptureSession` (คลาส, บรรทัด 326)

| เมธอด | หน้าที่ |
|---|---|
| `save(frame, source_time_ms, records, detections, verify, boxes, box_source)` | เขียน JPEG q95 + สำเนาที่วาดกรอบ + ต่อท้าย `frames.jsonl` และ `telemetry.jsonl` |
| `all_shots()` | อ่าน manifest และ telemetry **ครั้งเดียว** แล้ว group — O(n) |
| `shot_at(index)` | สร้าง `ShotResult` ของเฟรมเดียวขึ้นใหม่จากไฟล์ |
| `discard(frame_index)` | ลบภาพ + สำเนา + thumbnail + ทุก record ที่อ้างถึง แล้วเขียน log ใหม่แบบ atomic |
| `record_verification(shot)` | เขียนผลการอ่านซ้ำลง `telemetry-verified.jsonl` |

> **`all_shots()` เคยเป็น O(n²)** เพราะ `shot_at` อ่านทั้งสองไฟล์ใหม่ทุกครั้ง
> วัดได้ 0.4 วิ ที่ 50 ภาพ · 5.8 วิ ที่ 200 · 33 วิ ที่ 500 และหน้าเว็บเรียกทุกครั้งที่ capture

### `verify_shot(image_path, live_records, pipeline) -> Verification` (บรรทัด 781)
- **Input:** path ของภาพที่บันทึก, ค่าที่อ่านสด, pipeline **ตัวใหม่** (ห้ามใช้ตัวเดิม ไม่งั้นจะไม่เป็นการตรวจอิสระ)
- **ทำอะไร:**
  1. อ่านภาพจากไฟล์
  2. `process_frame` **5 รอบ** — ไม่ใช่เพื่อหาหลักฐานใหม่ แต่เพื่อให้การโหวตมีองค์ประชุมพอจะเผยแพร่ค่า
  3. เรียก `_ensemble_read`
  4. เทียบกับค่าที่อ่านสด แยกเป็น `agreed` (ข้อความตรง) กับ `lagging` (ค่าต่างเพราะโหวตยังไม่ตาม)
- **Output:** `Verification(status, fields, agreed, disagreed, lagging, duration_ms)`

### `_ensemble_read(frame, pipeline, produced)` (บรรทัด 134) — **กลไกเพิ่มความแม่นยำ**
- **ทำอะไร:** อ่านซ้ำด้วย **3 โมเดล × 3 ช่องสัญญาณ = 9 ครั้ง** แล้วนับคะแนนโหวต
  - เสียงข้างมากชนะ · เสมอกันดูที่ผลรวม confidence
  - **ทับค่าเดิมได้เฉพาะฟิลด์ที่ตั้ง `ensemble_override`** ฟิลด์อื่นเติมเฉพาะช่องที่ว่าง
- **ทำไมจำกัดขอบเขต:** เปิดทุกฟิลด์แล้ววัดพบว่า `elevation_deg` แย่ลง (เครื่องหมายลบหาย)
  จึงเปิดเฉพาะฟิลด์ที่มีเฉลยยืนยันแล้ว

### `annotate_frame(frame, boxes, source_width, source_height)` ([annotate.py](src/telemetry_ocr/annotate.py))
วาดกรอบลง **สำเนา** เสมอ — ภาพต้นฉบับคือหลักฐานและเป็นภาพที่ใช้ OCR ซ้ำ
ถ้ากรอบพาดทับข้อความ OSD การอ่านซ้ำจะผิด

---

## 3.9 `webapp.py` — เว็บเซิร์ฟเวอร์ (ไฟล์ใหญ่ที่สุด)

### `create_app(config, recognizer, ...) -> web.Application` (บรรทัด 75)
ประกอบทุกอย่างไว้ในฟังก์ชันเดียว ใช้ closure เก็บ state ร่วม

**Middleware สี่ชั้น (เรียงตามลำดับการทำงาน)**

| Middleware | บรรทัด | หน้าที่ |
|---|---|---|
| `guard_middleware` | 136 | ตรวจ Host allowlist · Origin · **บังคับ header `X-Telemetry-OCR` สำหรับ POST/PUT/PATCH/DELETE** |
| `cors_middleware` | 174 | ตอบ preflight และใส่ CORS header ให้ origin ที่อนุญาต |
| `static_cache_middleware` | 190 | ใส่ `Cache-Control: no-cache` ให้ asset ของแอป (กันเบราว์เซอร์รันโค้ดเก่า) |
| `error_middleware` | 207 | แปลง exception เป็น JSON response |

> **`X-Telemetry-OCR` มีไว้ทำไม:** request ที่มี custom header จะไม่ใช่ CORS simple request
> เบราว์เซอร์จึงต้องส่ง preflight มาก่อน และ guard จะปฏิเสธได้
> ถ้าไม่มีชั้นนี้ เว็บใดก็ตามที่ผู้ใช้เปิดอยู่จะสั่ง `/api/renders` ให้ GPU ทำงานยาว หรือแก้
> `/api/models` ว่าเครื่องนี้จะโหลด checkpoint (pickle) ตัวไหน
>
> **GET ไม่ต้องมี header โดยตั้งใจ** เพราะ `<img>` ส่ง custom header ไม่ได้ — ถ้าบังคับ ภาพในหน้าเว็บจะไม่ขึ้น

**Endpoint สำคัญ**

| Route | ฟังก์ชัน | หน้าที่ |
|---|---|---|
| `POST /api/capture/shot` | `capture_shot` | ★ อ่านเฟรมต้นฉบับจากไฟล์ + OCR + กรอบจาก render + OCR ซ้ำ |
| `GET /api/capture/shots` | `capture_shots` (1223) | รายการภาพทั้งหมด ใหม่สุดก่อน พร้อม `stage` |
| `GET /api/capture/image` | `capture_image` (1301) | ส่งภาพ (มี `annotated=1`, `thumb=1`) |
| `DELETE /api/capture/shot` | `capture_discard` (1266) | ลบภาพและ record ที่เกี่ยวข้อง |
| `POST /api/line/send` | `line_send` (1373) | ส่งเข้า LINE (rate limit 3 วิ · สูงสุด 4 ภาพ) |
| `GET/POST /api/models`, `DELETE /api/models/{name}` | `list_models` · `add_model` · `remove_model` | ทะเบียนโมเดล (เพิ่มจาก candidate id เท่านั้น) |
| `GET /api/videos`, `GET /api/videos/source` | `list_videos` · `video_source` | รายการวิดีโอ · ส่งไฟล์ต้นฉบับแบบ Range |
| `POST /api/renders`, `GET/DELETE /api/renders/current` | `start_render` · `render_status` · `cancel_render` | เริ่มหรือทำต่อ · ความคืบหน้า · ยกเลิก (เก็บส่วนที่ทำแล้ว) |
| `GET /api/renders/auto` | `auto_status` | สถานะ Capture / ส่งเข้าไลน์อัตโนมัติ |
| `GET /api/renders/tracks` | `render_tracks` | สถานะ + กรอบของช่วงเวลาหนึ่ง (ไม่เกิน 2 นาที) ให้หน้าเว็บวาด |
| `DELETE /api/renders` | `delete_render` | ลบผลการประมวลผลทั้งโฟลเดอร์ |
| `POST/GET /api/renders/export`, `GET /api/renders/export/file` | `start_export` · `export_status` · `export_file` | เริ่มทำ MP4 ไฟล์เดียว · ความคืบหน้า · ดาวน์โหลดแล้วลบ |

### `capture_shot(request)` — เส้นทางที่ซับซ้อนที่สุด
ไม่มี body — หาเฟรมจาก `source_time_ms` × fps (ปัดตาม stride ของ render) แล้ว `SourceFrames.read()`
จากไฟล์ต้นฉบับ จากนั้น → OCR → `renders.detections_at()` → `session.save` → `verify_shot`
→ `record_verification`

---

## 3.10 `line_client.py` — จัดรูปแบบและส่ง

### `format_dms(degrees, is_latitude, decimals=2) -> str` (บรรทัด 360)
- **Input:** องศาทศนิยม, เป็นละติจูดหรือไม่, จำนวนทศนิยมของฟิลิปดา
- **ทำอะไร:** แปลงกลับเป็น DMS · หาซีกโลกจากเครื่องหมาย · ทดเมื่อวินาทีปัดถึง 60
- **Output:** `14°50'57.37"N`

### `maps_link(latitude, longitude) -> str` (บรรทัด 387)
ใช้ **7 ตำแหน่งทศนิยม** (≈1.1 ซม.) ซึ่งละเอียดกว่าความละเอียดของข้อมูลต้นทาง (0.01″ ≈ 31 ซม.) มาก
จึงรับประกันว่าแปลงกลับแล้วตรงกับ DMS ที่พิมพ์ในข้อความ

### `format_capture_message(shot, directory, dms_decimals) -> str` (บรรทัด ~296)
- **Input:** `ShotResult` + โฟลเดอร์เซสชัน + ความละเอียดพิกัดรายฟิลด์
- **ทำอะไร:** เรียงตาม `_LAYOUT` ตายตัว (ACFT/TAG/CAM/LRF/SENS) เพื่อให้สายตาหาบรรทัดได้จากตำแหน่ง
  - นโยบายค่าที่ใช้: **อ่านจากภาพก่อน ถ้าไม่ผ่านค่อยใช้ค่าจากการโหวต** (วัดแล้วให้ผลดีที่สุด 95.5%)
- **Output:** ข้อความพร้อมส่ง

### `push_capture(config, text, image_urls, timeout_s)` (บรรทัด ~500)
รวมข้อความ + ภาพ (สูงสุด 4) ไว้ใน **คำขอเดียว** เพราะ LINE นับโควตาตามจำนวนคำขอ

---

## 3.11 `public_share.py` + `quick_tunnel.py` — เผยแพร่ภาพให้ LINE

### `ShareStore.mint(path) -> str` (public_share.py:59)
สร้าง token 43 อักขระ อายุ 10 นาที ผูกกับไฟล์หนึ่งไฟล์
ตรวจ `is_relative_to` เพื่อกันการชี้ออกนอกโฟลเดอร์เซสชัน

### `QuickTunnel` (quick_tunnel.py:60)
คุมโปรเซส `cloudflared`
- **`_READY_PATTERN`** = `Registered tunnel connection` — สัญญาณว่าใช้ได้จริง
- `cloudflared` **พิมพ์ URL ออกมาก่อน** ที่อุโมงค์จะพร้อม ถ้าส่งทันทีจะได้ภาพเสียถาวร
  (LINE ดึงครั้งเดียวแล้วแคช) จึงต้องรอสัญญาณนี้ **และทดสอบดึงภาพเองก่อนส่ง**

**สามกฎที่คลาสนี้ต้องรักษา** — ทั้งสามข้อมาจากบั๊กจริงที่ทำให้รูปไม่ถูกแนบไปกับข้อความ
โดยไม่มีอะไรบอก แล้วผู้ใช้ได้ข้อความที่ลงท้ายด้วย path บนเครื่อง ซึ่งเปิดได้เฉพาะเครื่องนั้น:

1. **หนึ่ง object หนึ่ง cloudflared** — `start()` ที่เจอ process เดิมยังไม่ตายจะ*รอ*ตัวนั้น
   ไม่ spawn ตัวใหม่ เพราะฝั่ง server ขอ tunnel ทุกครั้งที่ `url` เป็น `None` และ `url` เป็น
   `None` ตลอดช่วงระหว่างที่ process ถูกสร้างกับที่มัน register connection
2. **reader thread พูดแทนเฉพาะ process ของตัวเอง** — `_read_output`/`_consume` เช็ค
   `_owns(process)` ภายใน lock เดียวกับที่เขียน state ไม่งั้น thread ของ process เก่าที่
   ตายไปแล้วจะไปล้าง `_url` ของ tunnel ที่ยังทำงานอยู่
3. **`stop()` ฆ่าทุก process ที่ object นี้เคยเปิด** ไม่ใช่แค่ตัวล่าสุด — handle ที่หลุดไป
   คือ hostname สาธารณะที่ยังตอบอยู่หลัง server ที่มันชี้ไปหายไปแล้ว

### `Quota` (line_client.py) — โควต้ารายเดือนของ LINE
- LINE นับ push ตาม**จำนวนปลายทาง ไม่ใช่จำนวน message object** (เอกสารของ LINE ยกตัวอย่าง:
  4 message object เข้าห้อง 5 คน = 5) การส่ง 1 ครั้งจากแอปนี้ไปหาปลายทางเดียวจึงกิน 1
  ไม่ว่าจะแนบกี่รูป — และนี่คือเหตุผลที่ `push_capture` ยัดทุกอย่างไว้ใน request เดียว
- `limit is None` มีสองความหมายที่ห้ามปนกัน: แพ็กเกจไม่จำกัด (`type: "none"` ของ LINE)
  กับ**อ่านโควต้าไม่ได้** ฟิลด์ `error` เป็นตัวแยก ถ้าปนกันบัญชีที่เหลือ 3 ข้อความจะแสดงว่าไม่จำกัด
- server ปฏิเสธเองด้วย 429 ตอนโควต้าเต็ม ก่อนจะยิงไปหา LINE

### `ImageMessage` (line_client.py) — เพดานขนาดสองชั้น
`originalContentUrl` จำกัด 10 MB แต่ `previewImageUrl` จำกัด **1 MB** และ push ที่ preview
เกินถูกปฏิเสธทั้งก้อน telemetry หายไปด้วย เฟรมที่พอดีเป็น preview ของตัวเอง เฟรมที่เกินได้
สำเนา 1024 px q80 แยกใน `thumbs/` — **ไม่ใช่ `frames/`** เพราะ index ของ capture ถัดไป
คำนวณจากการสแกนโฟลเดอร์นั้น

---

# 4. ลำดับการทำงานและการไหลของข้อมูล

## 4.1 เส้นทางเว็บ (ที่ใช้งานจริงบ่อยที่สุด)

```
[1] ผู้ใช้รัน:  telemetry-ocr web --config configs/roi_calibrated_cuda.yaml
     │
     └─► cli.py:main()
          ├─ load_config()            → AppConfig            [config.py]
          ├─ create_recognizer()      → PaddleRecognizer     [ocr.py]
          └─ run_web_app()                                   [webapp.py]
               ├─ _load_registry_and_videos() → ModelRegistry + โฟลเดอร์วิดีโอ  [vision_lab]
               └─ create_app() → aiohttp Application (+ RenderManager, SourceFrames)

[2] หน้าเว็บเปิด → GET /api/models · GET /api/videos · GET /api/renders
     ผู้ใช้เลือกวิดีโอ → <video src=/api/videos/source?video=..> (Range)

[3] ผู้ใช้กด "ประมวลผลด้วย AI"
     │  POST /api/renders {video, model, confidence, image_size, stride}
     └─► RenderManager.start() → spawn ลูก                   [render_job.py]
          ├─ load_ultralytics_detector()
          └─ run_render(): ทีละเฟรม → track (BoT-SORT) → tracks.jsonl · meta.json
     หน้าเว็บสลับไป "วิดีโอ AI" ทันที (วิดีโอต้นฉบับเดิม + canvas): poll GET /api/renders/tracks
     ทุก 2 วิ ขอกรอบ 30 วิข้างหน้า วาดทุกเฟรมที่แสดง [track-overlay.js] · poll /api/renders/current
     ทุก 1 วิ เพื่ออัปเดตแถบ

[4] ผู้ใช้กดปุ่ม Capture
     │  POST /api/capture/shot?video=..&source_time_ms=..&render=..   (ไม่มี body)
     └─► capture_shot()                                      [webapp.py]
          ├─ SourceFrames.read(video, frame_index)  → เฟรมต้นฉบับ (ไม่มีกรอบ)
          ├─ TelemetryPipeline.process_frame()   ← ล็อก _RECOGNIZER_LOCK
          │    ├─ extract_roi() × 11 · apply_recipe() × 11
          │    ├─ recognize_grouped() → PaddleOCR
          │    └─ validate_text() → TemporalStabilizer.add()
          ├─ renders.detections_at(render, frame_index) → กรอบ + track_id
          ├─ session.save(...)         → JPEG + .boxes.jpg + manifest
          └─ verify_shot(image_path, live_records, pipeline ใหม่)
               ├─ process_frame() × 5   (ให้การโหวตมีองค์ประชุม)
               └─ _ensemble_read()      → 3 โมเดล × 3 channel = 9 ครั้ง → โหวต
                    │
                    ▼
          telemetry-verified.jsonl + ShotResult (JSON กลับไปหน้าเว็บ)

[5] ผู้ใช้ติ๊กเลือกภาพแล้วกด "ส่งเข้า LINE"
     │  POST /api/line/send?frame_index=..&frame_index=..
     └─► line_send()                                         [webapp.py:1373]
          ├─ ตรวจ ≤ 4 ภาพ · rate limit 3 วิ
          ├─ format_capture_digest()                         [line_client.py]
          ├─ ensure_quick_tunnel() → cloudflared
          ├─ _share_images() → token 43 อักขระ อายุ 10 นาที
          ├─ _reachable() ← ดึงภาพเองก่อน ถ้าไม่ได้ รีสตาร์ตอุโมงค์
          └─ push_capture() → POST https://api.line.me/v2/bot/message/push
```

## 4.2 เส้นทาง CLI แบบแบตช์

```
telemetry-ocr run --video flight.mp4 --config roi.yaml --jsonl out.jsonl --csv out.csv
   └─ TelemetryPipeline.run()  → อ่านแบบเดียวกับ [4] ข้างบน ทุกเฟรมที่ตั้งเวลาไว้ และเขียนไฟล์อย่างเดียว

telemetry-ocr capture --video flight.mp4 --session output/sessions/s1
   └─ capture_session()        [capture.py:185]   บันทึกเฟรม + manifest (ยังไม่ OCR)

telemetry-ocr extract --session output/sessions/s1 --config roi.yaml
   └─ extract_session()        [extract.py:51]
        ├─ load_session()      อ่าน manifest กลับมา
        ├─ _warn_on_config_drift()  เตือนถ้า config ต่างจากตอน capture
        └─ วนทุกเฟรม → TelemetryPipeline.process_frame() → RecordWriter
```

## 4.3 รูปแบบข้อมูลบนดิสก์

```
output/sessions/<ชื่อ>/
├── session.json                {source, started_at, frame_count, config_fields}
├── frames.jsonl                หนึ่งบรรทัดต่อหนึ่งภาพ: frame_index, file, annotated,
│                               source_time_ms, captured_at, bytes, detections, verification
├── telemetry.jsonl             ค่าจากการอ่านสด (แต่ละบรรทัดระบุ image ที่มันมาจาก)
├── telemetry-verified.jsonl    ค่าที่อ่านซ้ำจากไฟล์ภาพ
├── frames/
│   ├── 00000000.jpg            ★ ต้นฉบับ ไม่เคยถูกวาดทับ
│   └── 00000000.boxes.jpg      สำเนาที่วาดกรอบแล้ว
└── thumbs/                     ภาพย่อ 320 px สร้างเมื่อมีการขอครั้งแรก
```

---

# 5. จุดสังเกตสำหรับการพัฒนาต่อ

## 5.1 ⛔ ข้อห้ามที่จะทำให้ระบบพังทันที

| ข้อห้าม | เหตุผล |
|---|---|
| **อย่า import torch ในโปรเซส `telemetry-ocr`** | cuDNN ชนกับ Paddle → `WinError 127` งาน YOLO ต้องอยู่ในโปรเซสลูกเท่านั้น |
| **อย่าเรียก PaddleOCR โดยไม่ผ่าน `_RECOGNIZER_LOCK`** | ไม่ thread-safe และมีหลายเธรดเรียกจริง (capture + การตรวจซ้ำจากไฟล์ภาพ) |
| **อย่าถอด `X-Telemetry-OCR` guard ออกจาก POST/DELETE** | เปิดทางให้เว็บอื่นยิง `.pt` เข้ามาโหลดผ่าน pickle |
| **อย่าบังคับ guard กับ GET** | `<img>` ส่ง custom header ไม่ได้ ภาพในหน้าเว็บจะหายหมด |
| **อย่าเปลี่ยนเวอร์ชัน OpenCV** | ตรึงไว้ให้ contrib กับ standard ตรงกัน |
| **อย่าวาดกรอบลงภาพต้นฉบับ** | เป็นหลักฐานและเป็นภาพที่ใช้ OCR ซ้ำ |

## 5.2 🎯 จุดที่ปรับปรุง "ความแม่นยำ" ได้ (เรียงตามความคุ้มค่า)

### 1. fine-tune โมเดลสำหรับสามฟิลด์ท้ายตาราง — **คุ้มที่สุดตอนนี้**
เฉลยครบแล้ว: `examples/validation_labels_15-31-25-0.csv` มี 913 crops ตรวจด้วยตาทีละแถว
ครอบคลุม 11 ฟิลด์ (ไม่นับ `timestamp` ที่คำนวณจากเวลา) วัดได้ **95.9% (840/876)**
ยังต่ำกว่าเป้า 97% อยู่ 1.1 จุด — ดูตารางเต็มใน README หัวข้อ "Accuracy ที่วัดได้จริง"

ช่องว่างกระจุกอยู่ที่สามฟิลด์: `sensor_angle_deg` 91.6% · `eo_detail` 90.4% · `ir_detail` 90.4%
รวม 249 จาก 876 ตัวอย่าง และกินส่วนใหญ่ของ 1.1 จุดที่ขาด

**preprocessing หมดทางแล้วกับสามตัวนี้** — การ calibrate ด้วย label จริงทำให้ทั้งชุดขึ้นจาก
93.8% เป็น 95.9% แต่ `eo_detail`/`ir_detail` ไม่ขยับเลย (90.4% ทั้งก่อนและหลัง) เพราะสาเหตุ
คือ**ตัวอักษรกำกวมของฟอนต์ OSD** (`1.⑩x`, `FOv`, `F0V 1.9²`) ไม่ใช่คอนทราสต์
ส่วน `sensor_angle_deg` อ่านหลักนำหน้าเกิน (`2.46` เป็น `32.46`)

ขั้นถัดไปคือ label เพิ่มเฉพาะสามฟิลด์นี้ → fine-tune `PP-OCRv6_tiny_rec` → วัดซ้ำด้วย
`telemetry-ocr calibrate --labels ...` ที่มีอยู่แล้ว ไม่ใช่การ calibrate รอบใหม่

### 2. อย่าไล่ปรับ ROI ของ `eo_detail` / `ir_detail` ต่อ
วัดด้วย label แล้วว่า `color_baseline` (ภาพสีดิบ ไม่ทำ CLAHE/unsharp) ชนะ 6 จาก 11 ฟิลด์
และสองฟิลด์นี้ไม่ตอบสนองต่อ recipe ใดเลย — เวลาที่ลงไปกับ ROI/preprocessing ตรงนี้
ไม่คืนอะไรกลับมา

### 3. อย่าเพิ่ม padding
วัดแล้วแย่ลง: 2% → ถูก 35/42 · 16% → 31 · 35% → 22

### 4. อย่าเปลี่ยนไปใช้โมเดลใหญ่กว่า
วัดแล้ว `en_PP-OCRv5_mobile_rec` ได้ 26/42 และ `en_PP-OCRv4_mobile_rec` ได้ 21/42
ขณะที่ `PP-OCRv6_tiny_rec` ได้ 35/42 — **tiny ดีที่สุดเมื่อใช้เดี่ยว**
(แต่สองตัวนั้นมีค่าในฐานะผู้โหวตใน ensemble เพราะผิดคนละที่)

## 5.3 ⚡ จุดที่ปรับปรุง "ความเร็ว" ได้

| จุด | สภาพปัจจุบัน | ข้อเสนอ |
|---|---|---|
| **capture แรกของแต่ละรอบช้า 4.7 วิ** | โหลดโมเดล ensemble แบบ lazy ตอนใช้ครั้งแรก | โหลดล่วงหน้าในเธรดพื้นหลังตอน server เริ่ม (`_ensemble_recognizers` มีล็อกอยู่แล้ว) |
| **โมเดล YOLO ต้องอัปโหลดใหม่ทุกครั้งที่ restart** | ผู้ใช้ต้องกดเอง | โหลด `models/yolo/best.pt` อัตโนมัติ — **แต่ต้องชั่งกับความเสี่ยง** เพราะเป็น pickle |
| **thumbnail สร้างตอนขอครั้งแรก** | ครั้งแรกช้า ครั้งต่อไปเร็ว (4 ms) | สร้างตอน `session.save` เลย |
| **`recognizer_workers = 1`** | PaddleOCR ตัวเดียว | เพิ่มได้ถ้า VRAM เหลือ (ตอนนี้ใช้แค่ 125 MiB สำหรับ 3 ตัว) |
| **ensemble อ่าน 9 ครั้งทุกฟิลด์** | ~30 ms/ROI | ข้ามฟิลด์ที่ค่าเดิม `confirmed` และ conf สูงอยู่แล้ว |

## 5.4 🧹 โค้ดที่ตายแล้ว

หน้าเว็บ**เลิกใช้เส้นทาง OCR แบบเรียลไทม์แล้ว** และโค้ดที่ค้างจากเส้นทางนั้นถูกลบไปแล้วทั้งหมด

| สิ่งที่ลบไป | เคยอยู่ที่ | ทำไม |
|---|---|---|
| `submitFrame()` · `captureRois()` | app.js | ไม่มีใครเรียก |
| `applyOcrResult()` · `state.records` | app.js | **ไม่มีใครอ่าน** — ดูด้านล่าง |

> **`state.records` เป็นกรณีที่ควรจำ** เอกสารฉบับก่อนเขียนไว้ว่ามัน "ยังใช้อยู่ เพื่อสะสม
> ให้ปุ่ม Download JSONL/CSV" ซึ่งฟังดูสมเหตุสมผลและ**ผิด**: ปุ่มทั้งสองอ่าน `state.records`
> จริง แต่มีแค่ตัวรับ telemetry ของ*สตรีม*เท่านั้นที่เติมมัน ในโหมดไฟล์ — เปิดวิดีโอ กด
> Capture กด Download — array ว่างเปล่า JSONL จึงมี newline ตัวเดียวและ CSV มีแต่หัวตาราง
> โดยปุ่มเปิดใช้งานอยู่และไม่มีอะไรเตือน ส่วนบนสตรีมมันโตถึง 200,000 รายการเพื่อไม่มีใคร
>
> ตอนนี้ export อ่านจาก `state.shots` ซึ่งเป็นสิ่งเดียวกับที่ปุ่มใช้ตัดสินว่าจะเปิดใช้งานไหม
> **บทเรียน: "ใครเรียกฟังก์ชันนี้" กับ "ใครอ่านค่าที่มันเขียน" เป็นคนละคำถาม** การ grep หา
> ตัวเรียกอย่างเดียวตอบข้อแรกได้ แต่พลาดข้อสอง

> endpoint `POST /api/session/{id}/frame` ยังคงเก็บไว้ เพราะมีเทสต์ครอบและเป็น API สาธารณะ

## 5.5 🔁 รูปแบบบั๊กที่เกิดซ้ำในโปรเจคนี้

สามอย่างนี้ทำให้เกิดบั๊กจริงมาแล้วหลายตัว และคุ้มที่จะมองหาเป็นอย่างแรกเวลาแก้อะไร

**1. คอมเมนต์ที่ขัดกับโค้ดใต้มัน** โปรเจคนี้คอมเมนต์ละเอียดผิดปกติ ซึ่งแปลว่าคอมเมนต์ที่
พูดไม่ตรงกับโค้ดคือสัญญาณที่เชื่อถือได้ ตัวอย่างจริง:
- `logging.basicConfig` ที่คอมเมนต์บอกว่า *"stderr ถูกทิ้ง"* แล้วส่ง log ไป stderr
- `capture_session` ที่คอมเมนต์บอกว่า *"manifest ถูกเขียนทั้งสองทาง"* แต่จริงเฉพาะ Ctrl+C
- `_publish` ที่ docstring บอกว่า control message ไม่ถูกเขี่ยทิ้ง แต่ `put_latest` เขี่ย head

**2. ค่าที่มีก็ต่อเมื่องานจบแล้ว ในโค้ดที่งานไม่มีวันจบ** สตรีมรันจนกว่าจะสั่งหยุด
`PipelineSummary` จึงไม่เคยถูกสร้าง และ `status()` รายงาน `reconnects: 0` ตลอดชีวิตของสตรีม

**3. ตัวนับที่นับผิดสิ่ง** `ocr_interval`/`ocr_phase` เคยนับจากเฟรมที่ decoder grab แทนเฟรม
ที่ส่ง OCR จริง residue ที่เข้าถึงได้จึงขึ้นกับอัตราส่วน fps/ocr_hz และฟิลด์อาจไม่ได้ตาเลย
— แล้วค่าที่ค้างถูกป้อนซ้ำจนถึง quorum และรายงานว่า `confirmed`

**4. state ที่ไม่ถูกล้างตอนเปลี่ยนแหล่งข้อมูล** `handleFile` และ `enterStreamMode` หมุน
`detectionSourceId` และ `detectionGeneration` แต่ `leaveStreamMode` ไม่ทำ — track id จาก
กล้องจึงไหลไปทับวิดีโอ ทุกจุดที่เปลี่ยนแหล่งภาพต้องทำครบทั้งชุด (โหมดสตรีมถูกถอดออกจาก
หน้าเว็บภายหลัง แต่หลักเดียวกันยังใช้กับ `handleFile` และการ seek)
