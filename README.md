# Telemetry Vision

โปรเจกต์นี้ครอบคลุมงานวิดีโอโดรนสองด้านที่ใช้ข้อมูลชุดเดียวกัน:

- **`telemetry-ocr`** — ถอด telemetry ที่อยู่ตำแหน่งคงที่ในวิดีโอด้วย PaddleOCR
  เน้น latency ต่ำและผลลัพธ์ที่นิ่งกว่าการ OCR เต็มเฟรมทุกเฟรม
- **`vision-lab`** — เตรียม dataset, ประเมินผล YOLO checkpoint กับวิดีโอทั้งไฟล์
  และสร้างรายงาน ซึ่งเป็นฝั่งที่ **ผลิต** โมเดลที่ web UI ของ OCR เอาไปใช้วาดกรอบ

Pipeline ของ OCR คือ:

```text
video decoder (30 FPS)
  -> scheduler + bounded newest-frame queue (10.1 Hz ตาม config ที่ shipped)
  -> fixed ROI crops
  -> per-field color/contrast preprocessing
  -> batched PaddleOCR TextRecognition
  -> fallback recipe when confidence/format fails
  -> regex/range validation + temporal voting
  -> JSONL and CSV writer
```

เพราะตำแหน่งข้อความคงที่ ระบบจะไม่โหลด text detection, document orientation หรือ unwarping model

## โครงสร้างโปรเจกต์

```text
src/telemetry_ocr/     realtime OCR + web UI + detection worker (ห้าม import torch)
src/vision_lab/        dataset, evaluation, report (import torch ได้)
configs/               ROI configs และ paths.yaml
configs/paths.yaml     ทะเบียน checkpoint และตำแหน่งของสื่อขนาดใหญ่
models/yolo/           YOLO checkpoints (~106 MB, ไม่เข้า git)
models/paddlex/        PaddleOCR model cache
runs/                  ผลการเทรนแต่ละรอบ — curve, results.csv, args.yaml เข้า git;
                       weights, TensorBoard log และภาพ batch ตัวอย่างไม่เข้า
notebooks/             train_colab.py สำหรับรันบน Google Colab
output/                ผลลัพธ์ทั้งหมดที่ generate ได้ใหม่
```

วิดีโอต้นฉบับ, เฟรมที่สกัดแล้วและ Roboflow dataset รวมกันราว 8 GB จึงไม่ได้อยู่ในโปรเจกต์
แต่อยู่ที่ `asset_root` ใน [configs/paths.yaml](configs/paths.yaml) ซึ่งชี้ไปที่
`../object_detection` ตามเดิม ย้ายที่เก็บได้โดยแก้ไฟล์นั้นหรือตั้ง environment variable
`TELEMETRY_VISION_ASSETS`

ตรวจว่าเครื่องนี้เห็นอะไรบ้างด้วยคำสั่งเดียว:

```powershell
.\.venv-cuda\Scripts\vision-lab.exe check
```

คำสั่งนี้รายงานทั้ง torch/CUDA, paddle/CUDA, checkpoint ที่มีจริงบนดิสก์ และโฟลเดอร์สื่อที่หาเจอ
โดย probe paddle ใน child process เพื่อไม่ให้ cuDNN ของสองฝั่งชนกันในตัว diagnostic เอง

## ความต้องการระบบ

- Production ที่แนะนำ: WSL2/Ubuntu, Python 3.10–3.12 (แนะนำ 3.11), CPU x86-64
- Native Windows ใช้ได้เมื่อปิด `enable_hpi` แต่ประสิทธิภาพอาจต่ำกว่า
- PaddleOCR/PaddlePaddle และ OpenCV
- ROI สูงสุดเริ่มต้น 10 ช่อง, วิดีโอเป้าหมาย 1080p30, OCR 10.1 Hz (เป้าหมายระยะยาว 12 Hz)

เครื่องนี้ยังไม่มี WSL และ system Python เป็น 3.13 แต่ได้สร้าง local Python 3.12 virtual environment และทดสอบ PaddleOCR CPU inference จริงบน native Windows แล้ว ส่วน HPI ยังต้องใช้ WSL/Docker ตามขั้นตอนด้านล่าง

## ติดตั้งใน WSL2

ติดตั้ง WSL จาก PowerShell แบบ Administrator แล้ว restart ตามที่ Windows แจ้ง:

```powershell
wsl --install -d Ubuntu-24.04
```

ภายใน Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv
cd /mnt/c/Users/Lenovo/Documents/Personal_Projects/telemetry-vision
python3 --version  # ต้องอยู่ในช่วง 3.10–3.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -e ".[ocr,web,test]"
paddleocr install_hpi_deps cpu
```

คำสั่งติดตั้ง PaddlePaddle อาจแตกต่างตาม platform; หาก package index ปกติไม่มี wheel ที่ตรง ให้ใช้คำสั่ง CPU ล่าสุดจากเอกสาร PaddlePaddle ก่อนติดตั้งโปรเจกต์

หากยังไม่ได้ติดตั้ง HPI dependencies ให้ตั้ง `runtime.enable_hpi: false`; MKL-DNN ยังเปิดได้ผ่าน `enable_mkldnn: true`

ค่าเริ่มต้นใน example config ใช้ `model_source: BOS` และเก็บโมเดลใน `models/paddlex` เพื่อใช้ cache เดิมทุกครั้ง บน WSL ที่ติดตั้ง HPI dependencies แล้วจึงเปลี่ยน `enable_hpi` เป็น `true`

`model_cache_dir` เป็น path แบบ relative ต่อ **working directory ที่รันคำสั่ง** ไม่ใช่ต่อไฟล์ config
ถ้ารันจากโฟลเดอร์อื่นจะได้ cache คนละที่และดาวน์โหลดโมเดลใหม่เงียบๆ ให้ `cd` เข้ามาที่ราก
โปรเจกต์ก่อนเสมอ (`start_windows_ocr_web.ps1` และ Scheduled Task ตั้ง working directory ให้แล้ว)
หรือใส่ absolute path ลงใน config โดยตรง เช่นเดียวกับ `output/ultralytics-config` ที่ YOLO worker ใช้

สำหรับ CPU ค่าเริ่มต้นใช้ `PP-OCRv6_tiny_rec` เพราะมี latency ต่ำกว่าอย่างชัดเจนในการ benchmark บนเครื่องนี้ ตั้ง `warmup_iterations: 2` เพื่อไม่ให้การ compile oneDNN ครั้งแรกปนกับ realtime latency หาก labeled accuracy ของวิดีโอจริงต่ำกว่าเป้าหมาย ให้เปลี่ยนเป็น `en_PP-OCRv5_mobile_rec` แล้ว benchmark ใหม่

## ติดตั้งแบบ CUDA (แนะนำถ้ามี NVIDIA GPU)

วัดบน GTX 1650 (4 GB, compute 7.5) แล้ว GPU เร็วกว่า CPU ทั้งสองงาน และ**ผ่านเกณฑ์
10 Hz / p95 < 100 ms ที่ native Windows ทำไม่ได้บน CPU**:

| งาน | CPU | CUDA |
|---|---|---|
| OCR update rate | 9.84 Hz | **10.10 Hz** |
| OCR latency p95 | 164.0 ms | **46.6 ms** |
| OCR dropped frames | 130 | **1** |
| YOLO track @416 px | 236.8 ms | **32.8 ms** |
| YOLO track @640 px | 507.3 ms | **41.4 ms** |

บน CUDA ให้ใช้ image size **640 px** ได้เลย เพราะถูกกว่า 416 px บน CPU เสียอีก

```powershell
py -3.12 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.\.venv-cuda\Scripts\python.exe -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv-cuda\Scripts\python.exe -m pip install -e ".[ocr,web,ai]" nvidia-cudnn-cu12==9.9.0.52
```

บรรทัดสุดท้ายจำเป็น: `paddlepaddle-gpu 3.2.0` ประกาศ pin ไว้ที่ `nvidia-cudnn-cu12==9.5.1.17`
แต่ตัว binary คอมไพล์มากับ cuDNN 9.9 และจะเตือนตอน runtime ว่า *"may cause serious
incompatible bug"* การลง 9.9.0.52 ทับทำให้คำเตือนหายและทั้งสอง framework ยังทำงานได้
(pip จะบ่นเรื่อง pin ไม่ตรง ซึ่งไม่เป็นไร)

ใช้ `configs/roi_calibrated_cuda.yaml` ซึ่งต่างจากตัว CPU แค่ `device: gpu`,
`enable_mkldnn: false` (oneDNN เป็นไลบรารี CPU) และ `recognizer_workers: 1`
เพราะ VRAM 4 GB ต้องแบ่งกับ YOLO

`start_windows_ocr_web.ps1` จะเลือก `.venv-cuda` ก่อนเสมอ และถอยไปใช้ `.venv` (CPU)
อัตโนมัติถ้าไม่มี โดยบันทึกว่าเลือกอันไหนลง log

### ทำไม Paddle กับ PyTorch อยู่ process เดียวกันบน CUDA ไม่ได้

ทั้งคู่พก cuDNN 9 มาเอง และบน Windows ตัวที่โหลดทีหลัง**พังเสมอ ไม่ว่าจะสลับลำดับอย่างไร**
(`WinError 127` ที่ `cudnn_cnn64_9.dll`) ปัญหาคือ `paddleocr` ลาก torch เข้ามาทางอ้อมผ่าน
`paddlex → modelscope` การเปิด GPU เฉยๆ จึงทำให้ Paddle พังก่อนจะสร้างโมเดล OCR ด้วยซ้ำ

ทางออกคือ process ที่ถือ Paddle จะซ่อน torch จากตัวเอง (`paddleocr` ทำงานได้ปกติเมื่อไม่มี torch)
ส่วน YOLO อยู่ใน spawn process แยกซึ่งโหลด cuDNN ของ torch ได้อิสระ

ข้อจำกัดที่ตามมาคือ **ข้อมูลที่ส่งข้าม process ต้องเป็น builtin ล้วน** เพราะการ unpickle
จะ import โมดูลที่นิยามคลาสนั้น เช่น `torch.__version__` เป็น subclass ของ `str`
การส่งกลับมาจะลาก torch เข้าไปใน process ของ Paddle และทำให้ OCR พัง —
มีเทสต์ `DetectionBoundaryTests` คุมเรื่องนี้ไว้แล้ว

## ตั้งค่า ROI

คัดลอก [examples/roi_config.yaml](examples/roi_config.yaml) แล้วแก้ `box` ของแต่ละ field ซึ่งอยู่ในรูป `[x, y, width, height]` และทุกค่าเป็นสัดส่วน `0.0–1.0` เทียบกับขนาดเฟรม

ตัวอย่างสำหรับกรอบ pixel `(826, 842)–(1095, 928)` บนเฟรม 1920×1080:

```text
x      = 826 / 1920
y      = 842 / 1080
width  = (1095 - 826) / 1920
height = (928 - 842) / 1080
```

ควร crop เฉพาะค่า เช่น `123.4` ไม่รวมคำว่า `SPEED` หรือ `km/h` เมื่อไม่จำเป็น จากนั้นกำหนด:

- `allowed_chars`: อักขระที่ field นี้ยอมรับ
- `regex`: รูปแบบเต็มของค่า
- `data_type`: `string`, `integer` หรือ `float`
- `min_value`/`max_value`: ช่วงตัวเลขที่เป็นไปได้
- `profile`: `fast` ใช้ 2 ใน 3 เฟรม, `stable` ใช้ 3 ใน 5 เฟรม
- `max_delta_per_second`: ใส่เฉพาะเมื่อทราบข้อจำกัดการเปลี่ยนค่าจริง

ตรวจ configuration โดยไม่โหลด PaddleOCR:

```bash
telemetry-ocr validate-config --config examples/roi_config.yaml
```

สร้างภาพตรวจตำแหน่ง ROI ก่อนเริ่ม calibration:

```bash
telemetry-ocr preview-rois \
  --video input.mp4 \
  --config examples/roi_config.yaml \
  --frame-index 0 \
  --output output/roi-preview.jpg
```

สร้าง validation crops 1,000 ภาพและ manifest สำหรับกรอกคำตอบจริง:

```bash
telemetry-ocr export-crops \
  --video input.mp4 \
  --config examples/roi_config.yaml \
  --output-dir validation \
  --crop-count 1000
```

เปิดภาพใน `validation/images` แล้วกรอกคอลัมน์ `expected` ใน `validation/labels.csv`; แถวที่ยังว่างจะถูกข้ามเมื่อ calibration อ่าน labels

## Calibration สีและ contrast

แบบไม่มี labels จะเลือกสูตรจาก valid-rate, confidence และ latency:

```bash
telemetry-ocr calibrate \
  --video input.mp4 \
  --config examples/roi_config.yaml \
  --output-config output/roi_calibrated.yaml \
  --report output/calibration.json \
  --samples 30
```

แบบแนะนำคือสร้าง CSV ตาม [examples/validation_labels.csv](examples/validation_labels.csv) แล้วให้ calibration เลือกจาก exact-match accuracy ก่อน:

```bash
telemetry-ocr calibrate \
  --video input.mp4 \
  --config examples/roi_config.yaml \
  --output-config output/roi_calibrated.yaml \
  --report output/calibration.json \
  --labels examples/validation_labels.csv
```

Calibration ทดลอง 17 recipes ต่อ field ครอบคลุม baseline, RGB/Lab/HSV channels, CLAHE, unsharp mask, top-hat, black-hat, Otsu และ adaptive threshold ทั้งสอง polarity จึงอาจใช้เวลานานบน CPU เมื่อมี labels จำนวนมาก

## เลือกจำนวน CPU threads

```bash
telemetry-ocr benchmark \
  --video input.mp4 \
  --config output/roi_calibrated.yaml \
  --threads 2,4,6 \
  --iterations 30 \
  --report output/benchmark.json \
  --update-config
```

การ benchmark ทำ warm-up แล้วเลือกจำนวน threads ที่มี p95 inference latency ต่ำสุด จากนั้น `--update-config` จะเขียนค่ากลับไปยัง config

## รัน realtime

```bash
telemetry-ocr run \
  --video input.mp4 \
  --config output/roi_calibrated.yaml \
  --jsonl output/telemetry.jsonl \
  --csv output/telemetry.csv
```

Decoder จะเดินตาม source timestamp ของไฟล์ และ OCR ประมาณ 12 ครั้ง/วินาที หาก inference ช้ากว่ากำหนด queue จะเก็บเฉพาะ scheduled frame ล่าสุดเพื่อไม่ให้ latency สะสม ใช้ `--no-realtime` ได้เมื่อต้องการ decode เร็วที่สุดและยอมให้ scheduled frames ที่ค้างถูกทิ้ง

แต่ละ record มีโครงสร้าง:

```json
{
  "source_time_ms": 1250.0,
  "frame_index": 38,
  "field_id": "speed",
  "value": 12.5,
  "raw_text": "12.5",
  "confidence": 0.94,
  "confirmed": true,
  "preprocess_variant": "lab_l_clahe",
  "latency_ms": 37.8
}
```

`value` คือค่าล่าสุดที่ temporal voting ยืนยันแล้วและจะเป็น `null` จนกว่าจะยืนยันครั้งแรก ส่วน `raw_text` คือ OCR ของรอบปัจจุบัน `confirmed=true` หมายถึงค่าที่แสดงมีเสียงสนับสนุนถึงเกณฑ์ใน voting window ปัจจุบัน ไม่ได้หมายความว่าเป็น event ที่เกิดเพียงครั้งเดียว

เมื่อ primary OCR มี confidence ต่ำกว่า `0.80` หรือไม่ผ่าน regex/range ระบบจะประมวลผล fallback recipe เฉพาะ ROI นั้น และเลือก validity ก่อน confidence การแก้ `O→0`, `I/l→1`, `S→5`, `B→8` จะทำเฉพาะเมื่อผลเดิมไม่ผ่าน validation

## รับ RTSP stream แบบ realtime

`run` รับได้ทั้งไฟล์และ stream URL — ตรวจจาก scheme (`rtsp://`, `rtsps://`, `http://`,
`udp://`, `srt://` ฯลฯ) ไม่ต้องบอกระบบเอง:

```powershell
.\.venv-cuda\Scripts\telemetry-ocr.exe run `
  --video rtsp://user:pass@192.168.1.50:554/stream1 `
  --config configs\roi_calibrated_cuda.yaml `
  --jsonl output\telemetry.jsonl --csv output\telemetry.csv
```

กด Ctrl+C เพื่อหยุด ซึ่งถือเป็นการจบปกติ — summary ยังถูกพิมพ์ออกมา หรือใช้
`--max-seconds` เพื่อกำหนดเวลาไว้ล่วงหน้า

### stream ต่างจากไฟล์อย่างไร

| เรื่อง | ไฟล์ | stream |
|---|---|---|
| การหน่วงให้ตรงเวลา | หน่วงตาม timestamp ของไฟล์ | **ไม่หน่วง** — เครือข่ายกำหนดจังหวะมาแล้ว |
| นาฬิกา | `CAP_PROP_POS_MSEC` | **wall clock** นับจากเฟรมแรก |
| เมื่ออ่านไม่ได้ | จบไฟล์ | **reconnect** พร้อม backoff 0.5/1/2/4 วินาที |
| เฟรมเสีย | เป็น error | ทนได้ถึง 30 เฟรมติดกันแล้วจึง reconnect |
| จุดจบ | ท้ายไฟล์ | Ctrl+C หรือ `--max-seconds` |

**ทำไมต้องใช้ wall clock:** timestamp ของ stream เริ่มนับใหม่ที่ศูนย์ทุกครั้งที่ reconnect
และบ่อยครั้งไม่มีมาเลย ขณะที่ temporal voting, `max_delta_per_second` และตัวเลข latency
ทั้งหมดต้องการนาฬิกาที่เดินหน้าทางเดียว

**ทำไมต้องไม่หน่วง:** การหน่วง stream ตาม timestamp ของตัวเองมีแต่จะเพิ่ม delay
และ delay นั้นสะสมไปเรื่อยๆ ตลอดเวลาที่ stream ยังไม่หลุด ระบบจึงขอ `CAP_PROP_BUFFERSIZE = 1`
จาก backend ด้วย เพราะอะไรที่ backend buffer ไว้คือ latency ที่ pipeline มองไม่เห็นและแก้ไม่ได้

summary จะรายงาน `reconnects` และ `decode_failures` เพิ่มมาด้วย ซึ่งเป็นตัวแยกระหว่าง
"OCR ตามไม่ทัน" กับ "สัญญาณไม่นิ่ง"

### รัน YOLO บนเฟรมเดียวกันพร้อมกัน

```powershell
.\.venv-cuda\Scripts\telemetry-ocr.exe run `
  --video rtsp://user:pass@192.168.1.50:554/stream1 `
  --config configs\roi_calibrated_cuda.yaml `
  --detect-model best50 --detections output\detections.jsonl
```

`--detect-model` รับได้ทั้งชื่อย่อจาก [configs/paths.yaml](configs/paths.yaml) และ path เต็ม
YOLO ทำงานใน **spawn process แยก** แบบเดียวกับที่ web UI ทำ ด้วยเหตุผลเดิม: process ที่ถือ
PaddleOCR ต้องไม่ import torch

คิวขาเข้าของ worker เป็น newest-wins เฟรมที่ YOLO ตามไม่ทันจะถูกทิ้ง ไม่ใช่ทำให้ OCR รอ
วัดบนวิดีโอจริง 12 วินาทีที่ 640 px: OCR ได้ 9.95 Hz (latency mean 48.4 ms, p95 91.5 ms)
ส่วน detection ได้ 66 จาก 120 เฟรมที่ส่งไป โดย OCR ไม่ตกเฟรม — เทียบกับ OCR อย่างเดียว
ที่ latency mean 33.4 ms ส่วนต่างคือราคาของการแบ่ง GPU 4 GB กัน

## capture แล้วค่อยถอดทีหลัง (ไม่ผ่านหน้าเว็บ)

ถ้าต้องการเก็บทั้งช่วงแล้วค่อยถอด ใช้สองคำสั่งนี้แทน `run`:

```powershell
.\.venv-cuda\Scripts\telemetry-ocr.exe capture `
  --video ..\object_detection\data\15-31-25-0.mp4 `
  --config configs\roi_calibrated_cuda.yaml `
  --session output\sessions\flight01 --max-seconds 60
```

```powershell
.\.venv-cuda\Scripts\telemetry-ocr.exe extract `
  --session output\sessions\flight01 `
  --config configs\roi_calibrated_cuda.yaml
```

ต่างจาก `run` ตรงที่ **ไม่ทิ้งเฟรม** — เมื่อตัวเขียนตามไม่ทัน decoder จะรอ แทนที่จะทิ้งแบบที่
เรียลไทม์ทำโดยตั้งใจ และถอดซ้ำด้วย ROI ที่แก้แล้วได้โดยไม่ต้องถ่ายใหม่

### ความแม่นยำ: วัดแล้วเท่าเดิม

| | `run` (เรียลไทม์) | `capture` → `extract` |
|---|---|---|
| ที่ 10 Hz | 98.7% confirmed (972 ค่า) | 98.6% confirmed (948 ค่า) |
| ที่ 25 Hz | 99.4% confirmed (2,304 ค่า) | 99.4% confirmed (2,340 ค่า) |

เหตุผล: เรียลไทม์บนเครื่องนี้**ไม่ได้ตกเฟรมเลย** จึงไม่มีอะไรให้กู้คืน แม้ดันเป็น 25 Hz
จนเริ่มตกเฟรม (2 จาก 195) temporal voting ก็ยังทนรูโหว่ระดับนั้นได้ ประโยชน์ที่ได้จริงคือ
**ตรวจย้อนหลังได้**และ**ถอดซ้ำได้** ส่วนความแม่นยำจะดีขึ้นต่อเมื่อเรียลไทม์ตามไม่ทันจริง

### รูปแบบภาพ: JPEG q95 เป็นค่าเริ่มต้น

| | PNG | JPEG q95 |
|---|---|---|
| capture rate | 7.3 Hz | **23.8 Hz** |
| ขนาด 195 เฟรม | 238 MB | **74 MB** |
| ค่าที่อ่านต่างกัน | — | **0 จาก 2,340** |

PNG **ช้ากว่าเรียลไทม์** ไฟล์รับได้เพราะ decoder รอได้ แต่กล้องสดไม่รอ — capture stream
เป็น PNG จะเสียเฟรมใน buffer ของ backend ใช้ `--format png` ได้ถ้า overlay เล็กหรือจางกว่านี้

**หมายเหตุเรื่อง voting:** เฟรมที่กดเก็บทีละครั้งห่างกันหลายวินาที จะ confirm ได้น้อยเมื่อ
`extract` ใหม่ เพราะ voting ต้องการเฟรมที่ต่อเนื่องกัน ปุ่ม Capture ในหน้าเว็บก็อ่านทีละเฟรม
จึงมีข้อจำกัดเดียวกัน ถ้าต้องให้ voting ยืนยันค่า ให้บันทึกต่อเนื่องด้วย `telemetry-ocr capture` แล้ว `extract`

## Web UI สำหรับทดสอบวิดีโอ

### ติดตั้งเป็น PWA และเปิด OCR server อัตโนมัติ

1. ดับเบิลคลิก `install-ocr-web-autostart.cmd` หนึ่งครั้ง ระบบจะสร้าง Windows
   Scheduled Task ชื่อ `Telemetry OCR Web Server` และเปิด server ให้ทันที
2. เปิด <http://127.0.0.1:8765> ด้วย Edge หรือ Chrome
3. กดปุ่ม **ติดตั้งแอป** ที่มุมขวาบน หรือกดไอคอน Install ใน address bar
4. หลังติดตั้งแล้วเปิด **Telemetry OCR** จาก Start Menu/Desktop ได้เหมือนแอปทั่วไป

ครั้งถัดไป OCR server จะเปิดแบบซ่อนอัตโนมัติหลัง Login Windows จึงเปิด PWA แล้วใช้งานได้เลย
Scheduled Task นี้เปิดเฉพาะ standalone OCR web เท่านั้น ไม่เปิดหรือแก้ไข Flight Plan

**Task ถูกตั้งให้ทำงานตอนใช้แบตเตอรี่ด้วย** (`-AllowStartIfOnBatteries`
`-DontStopIfGoingOnBatteries`) ค่าเริ่มต้นของ Task Scheduler คือ**ไม่** start ตอนใช้แบต
และ**สั่งหยุด**ถ้าถอดปลั๊กระหว่างรัน ซึ่งเหมาะกับงาน sync เบื้องหลัง แต่ผิดกับงานนี้ —
ภาพถ่ายจากภาคสนาม เครื่องที่อ่านมันก็อยู่ภาคสนามและใช้แบต และอาการที่ได้คือ PWA เปิดได้แต่
ไม่มีอะไรอยู่เบื้องหลัง โดยไม่มี log บรรทัดไหนบอกสาเหตุเลย เพราะ task ไม่ได้รันตั้งแต่แรก

ถ้าติดตั้ง task ไว้ก่อนหน้านี้ ให้รันไฟล์ติดตั้งซ้ำหนึ่งครั้ง — `Register-ScheduledTask -Force`
จะอัปเดตค่าเดิมให้ ตรวจได้ด้วย:

```powershell
(Get-ScheduledTask -TaskName "Telemetry OCR Web Server").Settings |
  Select-Object DisallowStartIfOnBatteries, StopIfGoingOnBatteries
```

ทั้งสองค่าต้องเป็น `False` และทุกครั้งที่ start ตัวสคริปต์จะบันทึกแหล่งจ่ายไฟลง log ไว้ด้วย
(`Power: running on AC (100% charge).`) เผื่อวันหนึ่งอาการนี้กลับมาจากสาเหตุอื่น

หากต้องการถอน auto-start ให้ดับเบิลคลิก `remove-ocr-web-autostart.cmd` หรือรัน:

```powershell
.\scripts\install_windows_ocr_web_autostart.ps1 -Remove
```

Service worker เก็บเฉพาะหน้าแอปและ static assets เพื่อให้ PWA เปิดหน้าจอได้เร็ว แต่การอ่าน OCR
ยังต้องมี local server ทำงานอยู่ เพราะโมเดล PaddleOCR ประมวลผลใน Python บนเครื่อง

### แหล่งวิดีโอ

หน้าเว็บเล่นวิดีโอจาก**โฟลเดอร์วิดีโอของโปรเจกต์** คือ `asset_dirs.videos` ใน
[configs/paths.yaml](configs/paths.yaml) (ค่าเริ่มต้น `../object_detection/data`) เลือกจากรายการ
**วิดีโอ** ด้านบนของหน้า ไม่มีปุ่มเปิดไฟล์จากเครื่องแล้ว เพราะ server ต้องอ่านไฟล์เองทั้งตอน
ประมวลผลด้วย AI และตอนดึงเฟรมต้นฉบับให้ Capture ขณะที่ไฟล์ที่เบราว์เซอร์เปิดไม่เคยบอก path ให้
server รู้ วิดีโอใหม่ให้คัดลอกเข้าโฟลเดอร์นั้นแล้วโหลดหน้าใหม่ ย้ายโฟลเดอร์ได้ด้วย
`TELEMETRY_VISION_ASSETS`

ถ้าต้องอ่านกล้องสด ใช้ `telemetry-ocr run` หรือ `telemetry-ocr capture` ทาง CLI ซึ่งยังรับ
`rtsp://` ได้ตามเดิม (ดูหัวข้อ "รับ RTSP stream แบบ realtime" ด้านบน)

### log ของ server

log มีสองไฟล์และคนละเจ้าของ:

| ไฟล์ | ใครเขียน | มีอะไร |
|---|---|---|
| `output/logs/ocr-web-autostart.log` | `start_windows_ocr_web.ps1` | เลือก build ไหน, แหล่งจ่ายไฟ, quick tunnel เปิดไหม |
| `output/logs/ocr-web.log` | ตัว server เอง | tunnel, share server, การส่ง LINE, request ที่ถูกปฏิเสธ — หมุนไฟล์ที่ 2 MB เก็บย้อนหลัง 4 ไฟล์ |

ไฟล์ที่สองเป็นที่เดียวที่บอกได้ว่าทำไมข้อความถึงไปโดยไม่มีรูป — scheduled task รันแบบซ่อน
จึงไม่มี console และทุกอย่างที่เขียนลง stderr หายไปก่อนจะมีใครเห็น

### ปุ่ม Capture: เก็บภาพเฟรมนี้ไว้พร้อม telemetry ของมัน

ส่วนอื่นทั้งหมดของหน้านี้มีไว้ **ดู** — ตอบขณะวิดีโอเล่นแล้วไม่เก็บอะไรไว้ ปุ่ม Capture เป็นตัวเดียว
ที่**เขียนอะไรถาวร** คือภาพเฟรมนั้นพร้อมสิ่งที่ OCR อ่านได้จากมัน

- **กดครั้งเดียว** = เก็บ 1 เฟรม
- **กดค้าง** = เก็บต่อเนื่องทุก 400 ms จนกว่าจะปล่อย (ค่าที่กำลังเปลี่ยนมักหลุดถ้ากดทีละครั้ง)

ไฟล์จะไปอยู่ที่ `output/sessions/session-<วันเวลา>/` ในรูปแบบเดียวกับที่ `extract` อ่านได้:

```text
session-20260818-121223/
  session.json             แหล่งที่มา, เวลาเริ่ม, รายชื่อ field
  frames.jsonl             หนึ่งบรรทัดต่อหนึ่งเฟรม
  telemetry.jsonl          ค่าที่ OCR อ่านได้จากเฟรมนั้นตอนกด
  telemetry-verified.jsonl ค่าที่อ่านซ้ำจากไฟล์ภาพโดยตรง
  frames/00000000.jpg      ภาพต้นฉบับ (ไม่มีอะไรวาดทับ)
  frames/00000000.boxes.jpg  ภาพเดียวกันพร้อมกรอบ + class + conf + Track ID
```

**กรอบวาดลงสำเนา ไม่ใช่ต้นฉบับ** เพราะต้นฉบับคือหลักฐานและเป็นไฟล์ที่ถูกอ่าน OCR ซ้ำ
กรอบที่บังตัวเลข OSD จะทำให้ค่าที่อ่านได้จากมันเพี้ยน ไฟล์ `.boxes.jpg` จะมีเฉพาะเมื่อ
render เจอวัตถุในเฟรมนั้น ใช้สีและป้ายเดียวกับวิดีโอ AI เพื่อให้เทียบกันได้ว่าเป็นตัวเดียวกัน

**ค่าที่ส่งเข้า LINE อ่านจากไฟล์ภาพโดยตรง** ไม่ใช่ค่าที่ผ่าน temporal voting — เพราะค่าที่โหวต
ยืนยันไว้ตั้งใจให้หน่วงตามฟิลด์ที่เปลี่ยนเร็ว ตอนหน้าเว็บยังมีโหมดสตรีม วัดได้ว่าข้อความ OCR
ตรงกัน 10/11 ฟิลด์ แต่ค่าที่ publish ต่างกัน 8/11

ภาพที่เก็บเป็น**ความละเอียดเต็ม** (1920×1080) ไม่ใช่ภาพ preview ที่ย่อแล้ว และเก็บทั้งเฟรม
ไม่ใช่เฉพาะแถบ ROI — เพราะคำถามที่ capture มีไว้ตอบคือ "ตอนนั้นภาพเป็นยังไง"

**เฟรมมาจากไหน** — server อ่านเฟรม**ต้นฉบับ**จากไฟล์วิดีโอเอง ณ เวลาที่หน้าเว็บกำลังแสดง
(ปัดเป็นเฟรมที่ใกล้ที่สุด) ไม่ใช่ภาพที่หน้าเว็บส่งมา เพราะภาพบนจออาจเป็นวิดีโอ AI ที่มีกรอบทับตัวเลข
OSD อยู่ กรอบของภาพ capture มาจาก `tracks.jsonl` ของผลการประมวลผลเฟรมนั้นโดยตรง จึงตรงกับวิดีโอ
AI โดยไม่ต้องถามโมเดลซ้ำ ถ้าวิดีโอยังไม่ได้ประมวลผลด้วย AI ภาพจะถูกบันทึกพร้อมสถานะ `no_render`

ชื่อ session ถูกจำกัดเป็นตัวอักษร ตัวเลข จุด ขีด และขีดล่าง เพราะมันกลายเป็นชื่อโฟลเดอร์

### ส่งเข้า LINE

ปุ่ม **ส่งเข้า LINE** ข้างปุ่ม Capture ส่งเฟรมล่าสุดที่เก็บไว้ — telemetry ที่ถอดได้
พร้อมจำนวน class ที่ตรวจจับได้ และรูปที่มีกรอบ

ข้อความเริ่มที่ค่าที่อ่านได้เลย ไม่มีหัวข้อเวลา/ชื่อ session และไม่มี path ของไฟล์ต่อท้าย เพราะ
สามบรรทัดนั้นกินพื้นที่ preview ของ notification บนมือถือ ทั้งที่รูปแนบมากับข้อความเดียวกันอยู่แล้ว

```
ACFT  14°52'00.7"N   102°01'08.0"E   ALT 388 m
TAG   14°51'53.1"N   102°01'15.0"E?   ALT 237 m
CAM   yaw 7.91°   pitch -24.96°
LRF   35.8 m
SENS  EO 1.0xFOV66.4°?   IR 1.0x?
      TAG on map: https://www.google.com/maps?q=14.864750,102.020833

ตรวจจับได้: car 3 (conf 0.75, ต่ำสุด 0.67) (รวม 3)
ตรวจซ้ำจากไฟล์ภาพ: ข้อความตรงกัน 3/11
```

ส่งหลายเฟรมพร้อมกันจะมีบรรทัด `[Capture x2]` นำหน้า เพราะเป็นบรรทัดเดียวที่บอกว่ามีค่ากี่ชุดตามมา

`?` ต่อท้ายค่า = ยังไม่ยืนยัน ไม่ได้ซ่อนทิ้ง เพราะจุดประสงค์ของ capture คือรายงานสิ่งที่อยู่บนจอ
การตัดฟิลด์ที่อ่านยากที่สุดออกจะบิดเบือนภาพรวม

พิกัดเป็น **DMS แบบเดียวกับที่ OSD แสดง** เพื่อเทียบกับหน้าจอได้ตรงๆ ส่วนลิงก์ Google Maps
ใช้ทศนิยมเต็มความละเอียด (DMS ปัดที่ 0.1 วินาที ราว 3 เมตร ลิงก์จึงแม่นกว่าที่แสดง)
ลิงก์จะไม่ขึ้นถ้าอ่านพิกัดได้ไม่ครบทั้งสองค่า — ลิงก์จากพิกัดครึ่งเดียวจะชี้ไปตำแหน่งที่มีอยู่จริง
แต่ผิด ซึ่งแย่กว่าไม่มีลิงก์

#### ค่าที่ส่งมาจากไหน

อ่านจาก**ไฟล์ภาพ**ก่อน ถ้าอ่านไม่ได้จึงใช้ค่าที่ OCR อ่านได้ตอนกด (temporal voting)
วัดจากทุก capture บนดิสก์:

| นโยบาย | confirmed | โชว์เป็น raw text |
|---|---|---|
| ภาพอย่างเดียว | 94.5% | 5.5% |
| ค่าตอนกดอย่างเดียว | 88.2% | 1.8% |
| **ภาพ แล้ว fallback ค่าตอนกด** | **95.5%** | **1.8%** |

ภาพชนะกรณีทั่วไปเพราะเป็นรูปที่ข้อความอ้างถึง และเพราะ voting ตั้งใจให้ตามหลัง
(68 จาก 110 ฟิลด์ confirmed ทั้งคู่แต่ค่าต่างกัน คือ voting ยังยึดเฟรมก่อนหน้า)
แต่ภาพเป็นตัวอย่างเดียว พออ่านไม่ผ่าน validation ก็ไม่มีอะไรรองรับ — **fallback คือส่วนที่
เคยขาดไป** ทำให้ฟิลด์ที่ภาพอ่านพลาดถูกส่งเป็น raw OCR เช่น `10z°00'23.7` ทั้งที่ voting
มีค่า 102.006658 อยู่แล้ว

#### ขนาดรูปที่ LINE ยอมรับ

image message พก URL สองอัน และ LINE วัดมันด้วยเพดานที่ต่างกันสิบเท่า:

| URL | เพดาน | รูปที่โปรเจกต์นี้สร้าง |
|---|---|---|
| `originalContentUrl` | 10 MB | 400–650 KB — ห่างมาก |
| `previewImageUrl` | **1 MB** | ฉากที่ภาพซับซ้อนข้ามได้ |

push ที่ preview เกินเพดานถูกปฏิเสธ**ทั้งก้อน** ข้อความ telemetry หายไปด้วย ระบบจึงวัดขนาด
ก่อนส่ง: เฟรมที่พอดีถูกใช้เป็น preview ของตัวเอง (ไม่เสีย encode และ token เพิ่มโดยเปล่าประโยชน์)
เฟรมที่เกินจะถูกย่อเป็นสำเนา 1024 px q80 แยกต่างหาก เก็บใน `thumbs/` ข้างๆ ไม่ใช่ใน `frames/`
เพราะ index ของ capture ถัดไปคำนวณจากการสแกนโฟลเดอร์นั้น

รูปที่เกิน 10 MB หรือย่อแล้วยัง preview ไม่ผ่าน จะถูกข้ามไปพร้อมบอกเหตุผลใน `share_error`
แทนที่จะทำให้ทั้ง push ล้ม — telemetry คือส่วนที่คนรออยู่

ส่งเฉพาะ**รูปที่มีกรอบ** รูปต้นฉบับยังอยู่บนดิสก์ข้างๆ แต่ไม่ถูกส่ง เพราะสองรูปที่ต่างกัน
แค่กรอบทำให้ผู้อ่านต้องเปิดทั้งคู่เพื่อดูว่าอันไหนเป็นอันไหน

#### ตั้งค่า

ต้องมี **LINE Official Account** ที่เปิด Messaging API แล้ว (LINE Notify ปิดบริการไปเมื่อ
มี.ค. 2025 จึงเหลือทางนี้ทางเดียว)

**ลำดับการสร้างเปลี่ยนไปเมื่อ 4 ก.ย. 2024** — สร้าง Messaging API channel จาก LINE
Developers Console ตรงๆ **ไม่ได้อีกแล้ว** ต้องเริ่มจาก Official Account:

1. สร้าง LINE Official Account ที่ <https://manager.line.biz>
2. ใน **LINE Official Account Manager** → Settings → **Messaging API** → เปิดใช้งาน
   (ขั้นนี้เองที่ทำให้ channel ถูกสร้างขึ้น)
3. channel จะไปโผล่ใน <https://developers.line.biz/console> — เข้าไปที่
   **Messaging API** tab → **Channel access token (long-lived)** → Issue
4. **Basic settings** tab → **Your user ID** (ขึ้นต้นด้วย `U`) คือค่า `to` สำหรับส่งหาตัวเอง
5. **เพิ่ม OA เป็นเพื่อน** ด้วย QR ใน Messaging API tab — push ไปยัง user ที่ยังไม่ได้เพิ่ม
   เป็นเพื่อนจะไม่ผ่าน

```powershell
$env:TELEMETRY_OCR_LINE_TOKEN = "<channel access token>"
$env:TELEMETRY_OCR_LINE_TO    = "<user id ที่ขึ้นต้นด้วย U>"
```

หรือคัดลอก [configs/line.example.json](configs/line.example.json) เป็น `configs/line.json`
แล้วกรอกค่า — **ไฟล์นี้อยู่ใน .gitignore** เพราะ channel access token เป็น bearer token
ใครถือก็ส่งข้อความในนาม OA นี้ได้จนกว่าจะ revoke ค่าจาก environment มาก่อนไฟล์เสมอ
เพราะไม่ต้องแตะดิสก์ แต่ไฟล์สะดวกกว่าตรงที่ **ไม่ต้อง restart server** — ถูกอ่านใหม่ทุกครั้ง
ที่กดส่ง ขณะที่ environment variable ต้อง restart scheduled task ก่อนถึงจะเห็น

`to` **ไม่ใช่** ชื่อที่แสดงหรือ @handle ของ OA ถ้าจะส่งเข้ากลุ่มต้องใช้ group id (`C...`)
ซึ่งได้จาก webhook เท่านั้น และ webhook ต้องมี public HTTPS ซึ่งเครื่องนี้ยังไม่มี —
ส่งหาตัวเองใช้ user id จาก Basic settings ได้เลย

#### ที่ควรรู้

**ส่งรูปได้ ถ้าตั้ง `public_base_url`** — ดูหัวข้อถัดไป ถ้าไม่ตั้งจะส่งเป็นข้อความอย่างเดียว
ซึ่งไม่เปิดอะไรออกสู่ภายนอกเลย

**แยกจากปุ่ม Capture โดยตั้งใจ** — Capture กดค้างได้ (เก็บทุก 400 ms) ถ้าผูกการส่งไว้ด้วยจะ
เผา quota รายเดือนของ OA ทีละหลายข้อความ ฝั่ง server มี rate limit 3 วินาทีต่อครั้งอีกชั้น
เพราะ endpoint เรียกตรงได้ ปิดปุ่มฝั่ง client อย่างเดียวไม่พอ

**จำนวน class นับตอน capture ไม่ใช่ตอนส่ง** — ถ้านับตอนกดส่งจะได้ผลของเวลาที่กดส่ง
ไม่ใช่ของเฟรมที่เก็บไว้ และตัวเลขถูกเขียนลง `frames.jsonl` ด้วย ทั้ง "ยังไม่ได้โหลดโมเดล",
"ยังไม่มีผล" และ "ตรวจแล้วไม่พบ" แยกจากกัน เพราะแก้คนละทาง

เมื่อ LINE ปฏิเสธ ข้อความของ LINE จะถูกส่งต่อให้เห็น — 401 คือ token ผิด, 400 มักเป็น `to` ผิด,
429 คือเกิน quota

#### โควต้ารายเดือน

**โควต้าเป็นรายเดือน ไม่ใช่รายวัน** และรีเซ็ตเมื่อขึ้นเดือนใหม่ — label ข้างปุ่ม
**ส่งเข้า LINE** จึงขึ้นต้นด้วยคำว่า "เดือนนี้" เสมอ ไม่ใช่ตัวเลขลอยๆ:

```
[ ส่งเข้า LINE ]  เดือนนี้เหลือ 298/300
```

อยู่ข้างปุ่มไม่ใช่บนปุ่ม เพราะโควต้าเป็นข้อเท็จจริงของบัญชี ไม่ใช่ส่วนหนึ่งของการกระทำ
ตัวเลขบนปุ่มอ่านเหมือนเป็นจำนวนที่การกดครั้งนี้จะใช้ไป

เหลือน้อยกว่าหนึ่งในสิบของแพ็กเกจ (ขั้นต่ำ 10) label เป็นสีเหลืองพร้อมขึ้นข้อความเตือน
หมดแล้วเป็นสีแดงและปุ่มกดไม่ได้ — server ปฏิเสธเองด้วย 429 ก่อนจะยิงไปหา LINE เพราะการไป
เจอตอนนั้นแปลว่าถ่ายภาพและกดส่งไปแล้ว ถ้าอ่านโควต้าไม่ได้ label จะซ่อนไปเลย ไม่โชว์ค่าว่าง
ซึ่งจะดูเหมือนศูนย์ (เหตุผลอยู่ใน tooltip ของปุ่ม)

ตัวเลขมาจาก LINE สองปลายทาง (`/message/quota` กับ `/message/quota/consumption`) cache ไว้
5 นาที และหลังส่งสำเร็จจะ +1 เองในเครื่องแทนที่จะถามใหม่ ดูดิบๆ ได้ที่
`GET /api/line/status` ในฟิลด์ `quota`

**การส่ง 1 ครั้งนับเป็น 1 ข้อความ ไม่ว่าจะแนบกี่รูป** — LINE นับตามจำนวน*ปลายทาง*
ไม่ใช่จำนวน message object (เอกสารของ LINE ยกตัวอย่าง: 4 message object ส่งเข้าห้อง 5 คน
นับเป็น 5) นี่คือเหตุผลที่ `push_capture` ยัดข้อความกับรูปทั้งหมดไว้ใน request เดียว

reply message ไม่นับ แต่โปรเจกต์นี้ใช้ push อย่างเดียวจึงนับทุกครั้ง

#### ส่งรูปด้วย (Cloudflare Tunnel)

Messaging API **รับ upload ไฟล์ตรงๆ ไม่ได้** ต้องให้ URL แบบ HTTPS ที่ server ของ LINE
ดึงเองได้ เครื่องนี้อยู่ที่ `127.0.0.1` จึงต้องมีทางให้ LINE เข้าถึงชั่วคราว

**tunnel ไม่ได้ชี้มาที่แอปหลัก** แต่ชี้ไปที่ **server แยกอีกตัวบนพอร์ต 8766 ที่มี route
เดียวคือรูป** เหตุผลคือ quick tunnel ตั้ง ingress ไม่ได้ — มัน forward ทั้งพอร์ต ถ้าชี้มาที่
8765 จะเท่ากับเอา API ทั้งชุดขึ้นเน็ตโดยมี middleware กั้นชั้นเดียว ชี้ไปที่ 8766 แม้ middleware
มีบั๊กก็ไปเจอ process ที่ทำอะไรไม่ได้นอกจากคืนไฟล์ตาม token

##### ทางที่ 1: quick tunnel (ไม่ต้องมีบัญชีหรือโดเมน)

```powershell
New-Item -ItemType File configs\quick-tunnel.enabled -Force
```

แล้ว restart scheduled task — จบ ไม่ต้องตั้ง `public_base_url`

URL เปลี่ยนทุกครั้งที่ tunnel เริ่มใหม่ ระบบจึง**อ่าน URL จาก cloudflared เองตอนจะส่ง**
ไม่ได้จดไว้ล่วงหน้า ทดสอบเองได้ด้วย `telemetry-ocr web --quick-tunnel`

##### ทางที่ 2: named tunnel (URL คงที่ ต้องมีโดเมนใน Cloudflare)

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel login
.\scripts\setup_cloudflare_tunnel.ps1 -Hostname drone.example.com
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" service install
```

แล้วใส่ใน `configs/line.json`: `"public_base_url": "https://drone.example.com"`

##### ทางที่ 2ข: named tunnel ที่ตั้งจาก Cloudflare dashboard

ถ้าสร้าง tunnel จาก Zero Trust dashboard แทนที่จะใช้สคริปต์ จะได้ service ที่รันด้วย
`tunnel run --token ...` ซึ่ง**ไม่มี `config.yml` และไม่มี `~/.cloudflared/` บนเครื่องเลย** —
ingress ทั้งหมดอยู่ฝั่ง Cloudflare ตรวจว่ามีอยู่ไหมด้วย:

```powershell
Get-CimInstance Win32_Service -Filter "Name='Cloudflared'" | Select-Object State, PathName
```

กรณีนี้ต้องทำสองอย่างให้ตรงกัน มิฉะนั้น tunnel จะรันอยู่เฉยๆ โดยที่แอปไม่รู้จักและไปเปิด
quick tunnel ใหม่ทุกครั้งแทน:

1. **ใน dashboard** → Networks → Tunnels → tunnel ของคุณ → Public Hostname
   ตั้ง path เป็น `/public/img/*` และ service เป็น `http://127.0.0.1:8766`
   — **8766 ไม่ใช่ 8765** ด้วยเหตุผลเดียวกับที่อธิบายไว้ข้างบน
2. **ในเครื่อง** ใส่ hostname นั้นลง `configs/line.json` แล้วลบ `configs/quick-tunnel.enabled` ทิ้ง

```json
{
  "channel_access_token": "...",
  "to": "U...",
  "public_base_url": "https://drone.example.com"
}
```

`public_base_url` มาก่อน quick tunnel เสมอ ดังนั้นตราบใดที่ค่านี้ไม่ว่าง marker file
จะไม่มีผล แต่ลบทิ้งไปเลยชัดเจนกว่า ไม่ต้อง restart server — ไฟล์ถูกอ่านใหม่ทุกครั้งที่กดส่ง

ตรวจว่าใช้เส้นทางไหนอยู่จริงด้วย `curl -H "X-Telemetry-OCR: 1" http://127.0.0.1:8765/api/line/status`
ถ้า `quick_tunnel` ไม่ใช่ `null` แปลว่ายังไปทาง quick tunnel อยู่

##### วัดจริงผ่าน tunnel

| ผ่าน public URL | ผล |
|---|---|
| `GET /public/img/<token จริง>` | **200** — 503,899 bytes ตรงกับไฟล์บนดิสก์ทุกไบต์ |
| `GET /public/img/<token ผิด>` | 404 |
| `/` , `/api/health` , `/api/config` , `/api/session` , `/assets/app.js` | **404 ทุกตัว** |

404 ไม่ใช่เพราะถูกปฏิเสธ แต่เพราะ **route พวกนั้นไม่มีอยู่ใน server ตัวนั้นเลย**

token สุ่ม 43 ตัวอักษร · **หมดอายุ 10 นาที** · ชี้ได้เฉพาะไฟล์ใน `output/sessions/`
(resolve path จริงแล้วเทียบ ไม่ใช่เช็คสตริง) · GET เท่านั้น

⚠️ **ข้อจำกัดที่แก้ไม่ได้:** LINE ดึงรูปแล้ว **เก็บสำเนาไว้ที่ฝั่งเขา** token หมดอายุไม่ไปถึง
cache ของ LINE — ส่งรูปไปแล้วคือส่งไปแล้ว ถ้าภาพจากกล้องตัวนี้เปิดเผยไม่ได้ อย่าเปิดฟีเจอร์นี้
ส่งข้อความอย่างเดียวไม่เปิดอะไรออกเลย

### เปิด server เองโดยไม่ใช้ auto-start

เปิด local web server พร้อมโหลด PaddleOCR และ configuration ที่ calibrate จากวิดีโอจริง:

```powershell
.\.venv\Scripts\telemetry-ocr.exe web `
  --config configs\roi_calibrated.yaml `
  --open-browser
```

หากไม่ใช้ `--open-browser` ให้เปิด `http://127.0.0.1:8765` เอง เลือกวิดีโอจากรายการ **วิดีโอ**
ต้นฉบับจะเล่นทันที (server ส่งไฟล์แบบ Range จึง seek ได้โดยไม่ต้องโหลดทั้งไฟล์) กด **Capture**
เพื่อเก็บเฟรมพร้อมค่าที่ OCR อ่านได้ และดาวน์โหลดค่าของภาพที่เก็บไว้เป็น JSONL/CSV ได้

วิดีโอที่เลือกต้องใช้ layout telemetry ตรงกับ configuration หากตำแหน่ง overlay เปลี่ยนต้องสร้าง
หรือ calibrate config ใหม่ ต้นฉบับที่เบราว์เซอร์เล่นไม่ได้เพราะ codec ยังประมวลผลด้วย AI ได้ แล้วเล่น
วิดีโอ AI ซึ่งเข้ารหัสเป็น H.264 เสมอ การเปิดครั้งแรกอาจใช้เวลาสักครู่ระหว่างโหลดโมเดล OCR

### ประมวลผลวิดีโอด้วย AI (YOLO + BoT-SORT)

หน้าเว็บ**ไม่วาดกรอบสดทับวิดีโอแล้ว** แต่ให้ server รันวิดีโอทั้งไฟล์ผ่านโมเดล**ทุกเฟรมตามลำดับ
ครั้งเดียว** วาดกรอบลงในภาพ แล้วเข้ารหัสเป็น H.264 ที่เขียนออกมาเป็น**ชิ้นละ 2 วินาที** หน้าเว็บจึง
เล่นส่วนที่เสร็จแล้ว**ระหว่างที่ยังประมวลผลอยู่**ได้ ไม่ต้องรอทั้งไฟล์

เหตุผลที่เปลี่ยน: การวาดสดส่งเฟรมให้ BoT-SORT ได้ราว 10–16 จาก 30 เฟรมต่อวินาทีและห่างไม่เท่ากัน
ตามจังหวะที่ GPU ว่าง tracker จึงหลุดทุกครั้งที่กล้องเคลื่อน และกรอบที่วาดยังช้ากว่าภาพ 1–3 เฟรมเสมอ
การประมวลผลทั้งไฟล์ให้ tracker เห็นทุกเฟรมเรียงกัน และสิ่งที่เห็นบนจอคือสิ่งที่โมเดลเห็นในเฟรมนั้นพอดี
โดยดูไปพร้อมกับที่โมเดลทำงานได้

**วิธีใช้**

1. เลือกวิดีโอ แล้วเลือก **โมเดล**, **Confidence**, **Image size** และ **ประมวลผล**
   (ค่าเริ่มต้น **ทุก 2 เฟรม** ซึ่งเร็วพอ ๆ กับความเร็ววิดีโอบนเครื่องนี้ · ทุกเฟรมได้ภาพลื่นกว่าแต่ช้ากว่าเท่าตัว)
2. กด **ประมวลผลด้วย AI** หน้าเว็บสลับไป **วิดีโอ AI** ทันทีและเริ่มเล่นส่วนที่เสร็จแล้ว (ราว 10–15 วินาที
   แรกเป็นการโหลดโมเดล) เล่นทันจุดที่ประมวลผลถึงเมื่อไรจะขึ้น *กำลังประมวลผลส่วนถัดไป…* แล้วไปต่อเอง
3. แถบความคืบหน้าแสดง %, เฟรม/วินาที และเวลาที่เหลือ กด **ยกเลิก** ได้ทุกเมื่อ —
   **ส่วนที่ทำไปแล้วไม่หาย** รายการจะขึ้นว่า *ประมวลผลไว้ N% · หยุดไว้* พร้อมปุ่ม **ทำต่อ**
4. ปุ่ม **ต้นฉบับ / วิดีโอ AI** สลับได้โดยคงเวลาที่เล่นอยู่ · เลือกวิดีโอ + โมเดล + ค่าเดิมครั้งต่อไป
   ระบบเล่นผลเดิมทันทีโดยไม่ประมวลผลซ้ำ
5. กดที่หัวข้อ **ประมวลผลด้วย AI** เพื่อย่อแผงทั้งแผงให้วิดีโอเต็มพื้นที่ (หัวข้อยังบอกโมเดลและ
   ความคืบหน้า) ค่าที่เลือกถูกจำไว้ใน `localStorage` ของเครื่องนั้น

**ผลลัพธ์บนดิสก์** — `output/renders/<วิดีโอ>__<โมเดล>__c0.45-s640-st1/`

```text
run-000/        รอบแรก: init.mp4 + seg_00000.m4s … + index.m3u8 (ffmpeg เขียนชิ้นละ 2 วินาที)
run-001/        รอบที่ทำต่อหลังยกเลิก (ชิ้นในรอบนี้เริ่มนับเวลาใหม่ หน้าเว็บบวก offset ให้เอง)
tracks.jsonl    หนึ่งบรรทัดต่อเฟรม: frame_index, time_ms, detections (มี track_id)
meta.json       status (running/partial/complete), next_frame, runs, ค่าที่ใช้, encoder, fps,
                เวลาที่ใช้, เวลาเฉลี่ยต่อขั้นตอน, fingerprint ของวิดีโอกับโมเดล
```

ผลที่ประมวลผลไว้ก่อนหน้านี้ (ไฟล์ `annotated.mp4` ไฟล์เดียว) ยังอยู่ในรายการและเล่นได้เหมือนเดิม

**ยกเลิกแล้วทำต่อ** — ยกเลิก งานล้ม หรือ process ถูก kill ระบบจะเก็บชิ้นที่เขียนเสร็จแล้วไว้ทั้งหมด
บันทึก `next_frame` แล้วทำเครื่องหมายว่า `partial` กดประมวลผลอีกครั้งด้วยไฟล์และค่าเดิมจะ seek ไป
เฟรมนั้นแล้วทำต่อในโฟลเดอร์ `run-NNN` ใหม่ (ต่อ playlist เดิมไม่ได้ เพราะ ffmpeg จะเขียน init segment
ทับและทำให้เวลาของชิ้นเก่าเลื่อนทั้งหมด) Track ID ของรอบใหม่ถูกบวก offset ให้พ้นเลขที่ใช้ไปแล้ว

ผลที่ขนาดหรือเวลาแก้ไขของวิดีโอหรือไฟล์ `.pt` เปลี่ยนไปหลังประมวลผลถือว่า**เก่า** (ทำต่อไม่ได้ ต้องเริ่มใหม่) และรายการจะบอกว่า
ควรประมวลผลใหม่ ประมวลผลได้ทีละงานเพราะ GPU 4 GB (งานที่สองได้ `409`) ปิด server ระหว่างทาง
จะยกเลิกงานและปิดโปรเซสลูกให้ YOLO ยังรันใน spawn process แยกเหมือนเดิม เพราะ process ที่ถือ
PaddleOCR ห้าม import torch

**ต้องมี ffmpeg** — หาจาก `TELEMETRY_OCR_FFMPEG`, PATH แล้วจึงโฟลเดอร์ของ winget ถ้าไม่พบ
หน้าเว็บจะแจ้งและยังเล่นผลที่มีอยู่ได้ encoder เลือกเองตามลำดับ `h264_nvenc` → `h264_qsv` →
`libx264` โดยทดสอบ encode จริงสองเฟรมก่อนเริ่มงาน บังคับตัวใดตัวหนึ่งได้ด้วย
`TELEMETRY_OCR_RENDER_ENCODER`

**วัดจริง** บน GTX 1650 + Core i5-11320H คลิป 1080p30 ยาว 60 วินาที (1,801 เฟรม, `best50`, 640 px):

| encoder | เวลาทั้งหมด | ความเร็ว | ms ต่อเฟรม (decode / track / draw / encode) |
|---|---|---|---|
| `libx264` veryfast | 360.8 s | 5.0 เฟรม/วินาที | — |
| **`h264_nvenc`** | **112.5 s** | **16.0 เฟรม/วินาที** | 5.2 / 53.4 / 1.2 / 2.3 |

เบราว์เซอร์เล่น playlist ที่ยังเขียนไม่เสร็จเองไม่ได้ (Chrome 152 และ Edge 153 ขึ้น
`DEMUXER_ERROR_COULD_NOT_PARSE` ทั้งคู่) หน้าเว็บจึงดึงชิ้นวิดีโอมาต่อเข้า **Media Source** เอง
โดยโหลดล่วงหน้าราว 30 วินาทีและทิ้งส่วนที่ห่างจากจุดที่ดูอยู่เกิน 1 นาที — วิดีโอ 25 นาทีใหญ่เกินกว่าจะ
ถือไว้ทั้งไฟล์ในบัฟเฟอร์

x264 ไม่ได้ช้าเพราะตัวมันเอง (วัดแยก ถอดและเข้ารหัสได้ ~105 เฟรม/วินาที) แต่แย่ง CPU 4 คอร์กับ
tracker จนทั้งสายช้าลง ระหว่างนั้น GPU ใช้งานแค่ราว 21% NVENC ใช้วงจรเข้ารหัสเฉพาะบนการ์ดจอ
จึงคืน CPU ให้ tracker ตอนนี้เวลาเกือบทั้งหมดอยู่ที่ YOLO + BoT-SORT (~53 ms) วิดีโอ 25 นาทีจึงใช้
ราว 47 นาที ตัวเลือก **ทุก 2 เฟรม** (ค่าเริ่มต้น) ลดเวลาลงราวครึ่งหนึ่ง ทำให้ความเร็วประมวลผลใกล้เคียง
ความเร็ววิดีโอ จึงดูต่อเนื่องได้เกือบเหมือน real-time วิดีโอ AI ของคลิป 60 วินาทีนี้มีขนาด 53 MB (NVENC, CQ 27)

- ค่าเริ่มต้น confidence `0.45`, IoU `0.45`, image size `640` ซึ่งควรตรงกับขนาดที่ checkpoint เทรนมา
- `best50` (`models\yolo\best.pt`) รุ่น 50 epochs ตรวจพบวัตถุบน `15-31-25-0.mp4` ที่ `0.45` ส่วน
  `run6` รุ่น 100 epochs ไม่คืน detection แม้ลด confidence เป็น `0.10` จึงไม่แนะนำสำหรับวิดีโอนี้
- BoT-SORT ช่วยให้ track ID ต่อเนื่อง แต่แก้ false positive หรือวัตถุที่โมเดลตรวจไม่เจอไม่ได้ ในคลิป
  ทดสอบข้างบนมีกรอบเพียง 96 จาก 1,801 เฟรมและ track ยาวสุด 14 เฟรม ข้อจำกัดที่เหลือบนฟุตเทจนี้จึง
  อยู่ที่ recall ของโมเดลกับวัตถุเล็กจากมุมสูง ไม่ใช่ที่ tracker
- FP16 ใช้ไม่ได้บนการ์ดนี้ (เฉลี่ย 164.1 ms เทียบ FP32 30.7 ms เพราะ GTX 1650 ไม่มี tensor core)
  จึงใช้ FP32

ติดตั้ง dependency ด้วย:

```powershell
.\.venv-cuda\Scripts\python.exe -m pip install -e ".[ai]"
```

`best.pt` บางไฟล์ที่บันทึกด้วย custom pickle ต้องใช้ `dill` ซึ่งรวมอยู่ในชุด `ai` แล้ว พร้อม
`lap>=0.5.12` สำหรับ BoT-SORT

### จัดการโมเดลจากหน้าเว็บ

กด **จัดการโมเดล** ข้างรายการโมเดล:

- **โมเดลในรายการ** — ดูชื่อ ไฟล์ และขนาด กด **ตั้งเป็นค่าเริ่มต้น** หรือ **เอาออกจากรายการ**
  (ไฟล์ `.pt` ยังอยู่บนดิสก์เสมอ)
- **เพิ่มโมเดลจากไฟล์บนเครื่อง** — แสดง `.pt` ใน `models/` และ `runs/*/weights/` ที่ยังไม่อยู่ใน
  รายการ พร้อมชื่อย่อที่แนะนำ แก้ชื่อได้ก่อนกด **เพิ่ม** (ตัวอักษร ตัวเลข `. _ -` ไม่เกิน 40 ตัว)

การเปลี่ยนแปลงถูกเขียนลง `configs/models.local.yaml` (อยู่ใน `.gitignore`) ซึ่ง `load_paths`
ซ้อนทับบน `configs/paths.yaml` ไฟล์หลักจึงไม่ถูกเขียนทับและ comment ไม่หาย และ CLI
(`vision-lab models`, `run --detect-model`) เห็นรายการเดียวกับหน้าเว็บ ลบ `models.local.yaml`
ทิ้งเพื่อกลับไปใช้ `paths.yaml` อย่างเดียวได้

**ไม่มีการอัปโหลด `.pt`** — checkpoint ของ PyTorch เป็น pickle ที่รันโค้ดตอนโหลด หน้าเว็บจึงเพิ่มได้
เฉพาะไฟล์ที่ server เจอเองใต้โฟลเดอร์ของโปรเจกต์ โดยอ้างด้วย id ของรายการนั้น ไม่รับ path จาก
request ใส่ไฟล์ในโฟลเดอร์เหล่านั้นเฉพาะที่เชื่อถือได้

### API ที่หน้าเว็บใช้

- `GET /api/models` · `GET /api/models/candidates` · `POST /api/models` `{candidate_id, name}` ·
  `POST /api/models/default` `{name}` · `DELETE /api/models/{name}`
- `GET /api/videos` — วิดีโอในโฟลเดอร์ พร้อม fps, จำนวนเฟรม และความยาว
- `GET /api/videos/source?video=<id>` — ไฟล์ต้นฉบับ (รองรับ Range)
- `POST /api/renders` `{video, model, confidence, image_size, stride}` — เริ่มประมวลผล หรือ
  **ทำต่อ**จากผลที่ค้างไว้ (`202`) · คืน `existing` เมื่อมีผลที่เสร็จแล้วและยังใช้ได้ · `force` เพื่อเริ่มใหม่ทั้งหมด
- `GET /api/renders/current` · `DELETE /api/renders/current` — ความคืบหน้า (มี `start_frame`,
  `stages_ms` และ `encoder`) · ยกเลิก (ส่วนที่ทำแล้วยังอยู่)
- `GET /api/renders?video=<id>` — ผลทั้งหมดของวิดีโอนั้น พร้อม `status` และ `processed_ms`
- `GET /api/renders/stream?render=<id>` — รายการชิ้นวิดีโอที่เขียนเสร็จแล้ว (หน้าเว็บ poll ทุก 2 วินาที)
- `GET /api/renders/segment?render=<id>&run=run-000&file=seg_00000.m4s` — ชิ้นวิดีโอหนึ่งชิ้น
- `GET /api/renders/video?render=<id>` — ผลรุ่นเก่าที่เป็น MP4 ไฟล์เดียว · `DELETE /api/renders?render=<id>`
- `POST /api/capture/shot?session&session_id&video&source_time_ms&render` — ไม่มี body เพราะ server
  อ่านเฟรมเอง

### การป้องกัน API

Server ตรวจสามชั้นก่อนเข้า handler เพราะทุก endpoint มี side effect ที่เกิดขึ้นก่อนที่
CORS จะตัดสินว่าใครอ่าน response ได้:

- `Host` ต้องเป็น `127.0.0.1`, `localhost`, `::1` หรือค่าที่ส่งผ่าน `--host` มิฉะนั้นตอบ `403`
  เพื่อกัน DNS rebinding
- request ที่มี `Origin` แต่ไม่ผ่าน allowlist ถูก **ปฏิเสธ** ที่ `403` ไม่ใช่แค่ไม่ใส่ CORS header
- ทุก `POST`/`DELETE` ใต้ `/api/` ต้องมี header `X-Telemetry-OCR: 1`

ข้อสุดท้ายสำคัญที่สุด: request แบบ form หรือ `text/plain` เป็น CORS-safelisted จึงไม่เกิด preflight
ถ้าไม่บังคับ header นี้ เว็บใดก็ตามที่ผู้ใช้เปิดอยู่จะสั่ง `/api/renders` ให้ GPU ทำงานยาว หรือแก้
`/api/models` ว่าเครื่องนี้จะโหลด checkpoint (pickle) ตัวไหนได้ header ที่ไม่อยู่ใน safelist บังคับให้
browser ส่ง preflight ที่ guard ปฏิเสธได้

หากเรียก API จาก web app อื่นต้องใส่ทั้ง `--allow-origin` ตอนเปิด server และส่ง header
`X-Telemetry-OCR: 1` ทุก request ที่เปลี่ยนสถานะ

## ฝั่ง detection: เตรียมข้อมูล วัดผล และเทรน

คำสั่งชุดนี้อยู่ใต้ `vision-lab` แยกจาก `telemetry-ocr` เพราะมันโหลด torch ตรงๆ ซึ่งเป็นสิ่งที่
process ของ PaddleOCR ทำไม่ได้ (ดูหัวข้อ cuDNN ด้านบน) ทั้งสองคำสั่งติดตั้งมาจาก package
เดียวกันและใช้ venv เดียวกันได้

### ทะเบียน checkpoint

`configs/paths.yaml` ตั้งชื่อย่อให้ checkpoint เพื่อไม่ต้องพิมพ์ path ยาวทุกครั้ง:

```powershell
.\.venv-cuda\Scripts\vision-lab.exe models
```

| ชื่อย่อ | ไฟล์ | หมายเหตุ |
|---|---|---|
| `best50` | `models/yolo/best.pt` | 50 epochs — ค่าเริ่มต้น เป็นตัวเดียวที่วัดแล้วคืน detection บนวิดีโอเป้าหมาย |
| `best01` | `models/yolo/best01.pt` | |
| `yolov8n` | `models/yolo/yolov8n.pt` | pretrained baseline |
| `run4` / `run6` / `run7` | `runs/visdrone_custom_*/weights/best.pt` | ผลจากแต่ละรอบเทรน |

ที่ไหนก็ตามที่รับ `--model` จะรับได้ทั้งชื่อย่อและ path เต็ม

หน้าเว็บเพิ่ม เอาออก และตั้งค่าเริ่มต้นได้จาก **จัดการโมเดล** โดยเขียนลง `configs/models.local.yaml`
ที่ซ้อนทับไฟล์นี้ (ดูหัวข้อ "จัดการโมเดลจากหน้าเว็บ")

### สกัดเฟรมไปทำ dataset

```powershell
.\.venv-cuda\Scripts\vision-lab.exe extract-frames --interval 1.0
```

อ่านวิดีโอทุกไฟล์จากโฟลเดอร์ `videos` ของ asset root แล้วเขียน JPEG แยกโฟลเดอร์ตามชื่อวิดีโอ
ลงในโฟลเดอร์ `frames` ระบุ `--source`/`--output` เพื่อ override ได้

### วัดผล checkpoint กับวิดีโอทั้งไฟล์

```powershell
.\.venv-cuda\Scripts\vision-lab.exe evaluate 15-31-25-0.mp4 --track --imgsz 640
```

ชื่อไฟล์เปล่าๆ จะถูกค้นในโฟลเดอร์ `videos` ของ asset root แต่ละรอบเขียนโฟลเดอร์ใหม่ที่มี:

- `frame_summary.csv` — จำนวน detection และ inference latency รายเฟรม แยกตามคลาส
- `summary.json` — ค่า setting ที่ใช้จริง, throughput และจำนวน track ID ไม่ซ้ำเมื่อเปิด `--track`
- วิดีโอที่วาดกรอบแล้ว (ปิดด้วย `--no-save-video`)

`--imgsz` ควรตรงกับ image size ตอนเทรน มิฉะนั้นความแม่นยำจะลดลง ส่วน `--vid-stride 3`
ช่วยให้สำรวจวิดีโอยาวได้เร็วขึ้น

ต่างจากการประมวลผลในหน้าเว็บตรงที่คำสั่งนี้ให้ตัวเลขสรุปที่เทียบ checkpoint กันได้
(`frame_summary.csv`, `summary.json`) ส่วนหน้าเว็บเก็บวิดีโอ AI กับ `tracks.jsonl` ไว้ดูและ capture ต่อ

### เทรนรอบใหม่

`notebooks/train_colab.py` เป็นสคริปต์แบบแบ่งเซลล์สำหรับ Google Colab (ใช้ syntax `!pip`
จึงไม่ใช่ Python ที่ import ได้ และถูกกันออกจาก ruff/mypy ไว้แล้ว) เมื่อเทรนเสร็จให้ก๊อป
โฟลเดอร์ผลลัพธ์มาไว้ใน `runs/` แล้วเพิ่มชื่อย่อลง `configs/paths.yaml`
โดย curve และ `results.csv` จะเข้า git ส่วน `weights/` ไม่เข้า

### คู่มือฉบับเต็ม (PDF ภาษาไทย)

```powershell
.\.venv-cuda\Scriptsision-lab.exe handbook
```

สร้างคู่มือที่อธิบาย **ทั้งสองเส้นทาง** — OCR และ object detection — ไล่ทีละขั้นตอน
พร้อมบอกว่าพารามิเตอร์แต่ละตัวคืออะไร ทำไมถึงเลือก และทำไมถึงเป็นค่านั้น รวมถึงตัวเลข
ที่วัดได้จริงทั้งหมด ตัวเลขทุกตัวดึงจาก config, `args.yaml`, `results.csv` และค่าคงที่ในโค้ด

ต่างจาก `report` ด้านล่างซึ่งครอบคลุมเฉพาะทฤษฎี YOLOv8 กับ dataset

### รายงาน PDF ภาษาไทย

```powershell
.\.venv-cuda\Scripts\vision-lab.exe report --run visdrone_custom_6th
```

สร้าง PDF อธิบายทฤษฎีและ dataset โดยฝังภาพจากโฟลเดอร์ผลการเทรนที่ระบุ ต้องมี Chrome หรือ
Edge ในเครื่องเพราะ ReportLab ไม่มี complex text shaping จึงวางวรรณยุกต์ไทยที่ซ้อนบนสระทับกัน
(`ที่` กลายเป็น `ที`) ส่วน ReportLab ยังใช้ประทับเลขหน้าซึ่งเป็นตัวเลขล้วน ติดตั้ง dependency
ด้วย `pip install -e ".[report]"`

## ทดสอบ

หลังติดตั้ง test dependencies:

```bash
python -m pytest
```

หรือรัน core tests ด้วย standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

ชุดเทสต์ไม่ต้องใช้ PaddlePaddle เพราะ dependency หนักถูก import แบบ lazy ทั้งหมด
เทสต์ที่ต้องใช้ aiohttp/OpenCV จะ skip เองถ้าไม่มี

เทสต์ฝั่งหน้าเว็บใช้ `node:test` ที่มากับ Node ไม่ต้องติดตั้งอะไรเพิ่ม:

```bash
node --test tests/format.test.js tests/web-shell.test.js
```

Lint และ type check (ติดตั้งด้วย `pip install -e ".[lint]"`):

```bash
python -m ruff check . && python -m mypy
```

GitHub Actions ใน `.github/workflows/ci.yml` รัน pytest บน Python 3.11/3.12 พร้อม ruff
และ mypy ส่วน `ruff format` ยังไม่ได้บังคับใช้เพราะจะ rewrite ทุกไฟล์จนกลบ diff ของงานจริง

ก่อนใช้งานจริงควรมี labeled set อย่างน้อย 1,000 ROI crops และตรวจ:

- exact-field accuracy เป้าหมายอย่างน้อย 97%
- sustained update rate อย่างน้อย 10 Hz, เป้าหมาย 12 Hz
- p95 frame-to-result latency ไม่เกิน 100 ms
- `dropped_frames` ไม่เพิ่มต่อเนื่องใน workload ปกติ

หาก preprocessing ยังไม่ถึงเป้าหมาย ให้ fine-tune `en_PP-OCRv5_mobile_rec` ด้วยภาพจริงและภาพสังเคราะห์อย่างน้อย 5,000 crops โดยคง character dictionary เดิมก่อน และเพิ่ม augmentation สำหรับ blur, video compression, low contrast และสีที่กลืนพื้นหลัง

## ผล smoke test บนเครื่องนี้

ทดสอบ native Windows CPU ด้วย PaddlePaddle 3.2.0, PaddleOCR 3.7.0 และ `PP-OCRv6_tiny_rec` กับวิดีโอสังเคราะห์ 1080p30 จำนวนสาม ROI:

- 12.04 OCR updates/second
- p95 latency 37.9 ms หลัง warm-up
- dropped frames 0
- อ่านถูก 180/180 field records (`123.4`, `2450`, `D`)

ผลนี้ยืนยัน runtime path แต่ไม่ใช่ accuracy ของวิดีโอจริง ซึ่งต้องวัดใหม่ด้วย `export-crops` และ labels จากวิดีโอเป้าหมาย

## ข้อจำกัด

- ROI ต้องคงที่และวิดีโอไม่มีการหมุนหรือเปลี่ยน layout
- ถ้าตัวอักษรและพื้นหลังเหมือนกันจนไม่มี spatial/color signal เหลืออยู่ preprocessing ไม่สามารถกู้ข้อมูลที่หายไปได้ ต้องใช้ต้นฉบับคุณภาพสูงขึ้น, multi-frame reconstruction หรือ fine-tuning
- Exact temporal voting ไม่เหมาะกับค่าที่เปลี่ยนทุก OCR tick; field ดังกล่าวควรลด `temporal.fast.required` เป็น `1` หลังวัด false-read rate จาก validation set
- ค่าที่เปลี่ยนทุก tick ใช้ median ข้ามเฟรมแทน exact vote และต้องมี observation ที่ valid
  อย่างน้อย **3 ค่า** จึงจะ `confirmed` ได้ เพราะ median ของ 2 ค่าคือค่าที่ใหม่กว่าเสมอ
  ไม่ได้กันค่าผิดเลย ผลคือ field แบบ `fast` ยืนยันช้าลงหนึ่ง tick และวัดจาก
  `telemetry-9.78hz.jsonl` แล้วมี tick ที่ `confirmed` ลดลง 0.3–8.3% แล้วแต่ field
  โดยค่าที่รายงานยังเป็นค่า confirmed ล่าสุดเหมือนเดิม เพียงแต่ `confirmed=false`
- ตั้ง `max_delta_per_second` ให้ field ที่มีขีดจำกัดทางกายภาพชัดเจน จะได้ด่านที่สอง:
  หน้าต่าง voting ที่กระจายเกินกว่าที่ค่าจะเปลี่ยนทันจะไม่ถูกยืนยัน ค่าที่วัดได้จาก run
  ที่มีอยู่ยังปนค่าผิดจึงใช้ตั้ง threshold ไม่ได้ ควรใช้ข้อจำกัดจริงของ gimbal และอากาศยาน

## Real-video CPU tuning (15-31-25-0.mp4)

The calibrated configuration for the supplied 1920x1080 flight video is
`configs/roi_calibrated.yaml`. It is tracked in git because the Windows
auto-start task depends on it and `output/` is ignored. It uses the following
runtime controls:

- `recognizer_workers: 2` with one CPU thread per independent Paddle model.
- `derive_from_source_time: true` for the clock after its first valid OCR anchor.
- `ocr_interval`/`ocr_phase` to stagger the two slowly changing altitude fields.
- `enable_fallback: false` in the native-Windows real-time profile. Set it to
  `true` for a slower accuracy-first pass that retries invalid/low-confidence ROIs.
- Numeric fields that change every tick use a multi-frame median when an exact
  2-of-3 vote is impossible; string fields retain exact voting.
- The decoder calls `grab()` for every source frame and `retrieve()` only for
  scheduled OCR frames, avoiding unnecessary full-HD BGR copies.

The full 513-second run over the 12 fields this config now defines produced
60,624 records from 5,052 processed OCR frames: 9.84 updates/second, 92.7 ms
mean end-to-end latency, 164.0 ms p95 latency, and 130 newest-frame queue drops.
Native Windows on the CPU is therefore below the strict 10 Hz / 100 ms-p95
acceptance target. The same run on CUDA clears it comfortably at 10.10 Hz and
46.6 ms p95, so a GPU, not WSL2/HPI, is now the recommended route to that
target. PaddleX HPI is Linux-only for this setup and remains the option for
CPU-only deployments that must hit it.

An earlier figure of 9.90 Hz and 136.2 ms p95 is quoted in some notes. It was
measured when the config still had 10 fields (50,780 records over 5,078 frames
is exactly 10 per frame), so it is not comparable: 12 fields is 20% more OCR
work per frame and p95 rose 20.4%, which accounts for the whole difference.

## Accuracy ที่วัดได้จริง

`examples/validation_labels_15-31-25-0.csv` คือ label ที่ตรวจด้วยตาทีละแถวจากเฟรมจริง
ครบ 913 crops (ไม่นับ `timestamp` ที่คำนวณจากเวลา ไม่ได้ OCR) โดยดูภาพก่อนแล้วค่อยเทียบกับ
ผล OCR เพื่อไม่ให้คำตอบเดิมชี้นำ

**exact-field accuracy = 95.9% (840/876)** ยังต่ำกว่าเป้า 97% อยู่ 1.1 จุด

| field | n | recipe เดิม | recipe ที่เลือกด้วย label |
|---|---|---|---|
| `tag_lat_dms` | 83 | 97.6% | **100.0%** |
| `acft_lon_dms` | 83 | 98.8% | 98.8% |
| `tag_lon_dms` | 83 | 96.4% | 98.8% |
| `tag_alt_m` | 83 | 96.4% | 98.8% |
| `lrf_m` | 46 | 95.7% | 97.8% |
| `elevation_deg` | 83 | 95.2% | 97.6% |
| `acft_lat_dms` | 83 | 95.2% | 96.4% |
| `acft_alt_m` | 83 | 90.4% | 95.2% |
| `sensor_angle_deg` | 83 | 86.7% | **91.6%** |
| `eo_detail` | 83 | 90.4% | 90.4% |
| `ir_detail` | 83 | 90.4% | 90.4% |
| **รวม** | **876** | **93.8%** | **95.9%** |

การ calibrate ครั้งแรกเลือก recipe จาก *valid rate* จึงเลือก preprocessing ที่ทำให้ผลแย่ลง
พอวัดด้วย label จริง `color_baseline` (ภาพสีดิบ ไม่ทำ CLAHE/unsharp) ชนะ 6 จาก 11 field
และได้ +2.1 จุด รันซ้ำได้ด้วย:

```bash
telemetry-ocr calibrate --video <video> --config configs/roi_calibrated_cuda.yaml \
  --output-config configs/roi_calibrated_cuda.yaml --report output/calibration.json \
  --labels examples/validation_labels_15-31-25-0.csv
```

`lrf_m` นับ 46 แถวเพราะอีก 37 แถวเป็น `0.00` (ไม่มีสัญญาณสะท้อน) ซึ่งต่ำกว่า `min_value` แล้ว
จึงไม่มีระยะให้ตรวจ

ที่เหลือให้ไล่ต่อคือสองตัวท้ายตาราง: `sensor_angle_deg` ยังอ่านหลักนำหน้าเกินในบางเฟรม
(`2.46` เป็น `32.46`) และ `eo_detail`/`ir_detail` พลาดจากตัวอักษรกำกวมของฟอนต์ OSD
(`1.⑩x`, `FOv`, `F0V 1.9²`) ซึ่งน่าจะต้อง fine-tune โมเดลมากกว่าปรับ preprocessing

### รอบ label ถัดไป (สำหรับดัน 95.9% → 97%)

ช่องว่างกระจุกอยู่ที่สามฟิลด์ — `sensor_angle_deg` 91.6%, `eo_detail` 90.4%, `ir_detail` 90.4%
รวม 249 จาก 876 ตัวอย่าง จึงไม่มีเหตุผลจะ label ครบสิบสองฟิลด์อีกรอบ

**1. export เฉพาะสามฟิลด์ และข้ามเฟรมที่ตอบไปแล้ว**

```bash
telemetry-ocr export-crops --video <video> --config configs/roi_calibrated_cuda.yaml --output-dir output/labels-pass2 --crop-count 900 --fields sensor_angle_deg,eo_detail,ir_detail --exclude-labels examples/validation_labels_15-31-25-0.csv
```

`--fields` ทำให้โควตา crop ไปลงที่สามฟิลด์แทนที่จะถูกหารด้วยสิบสอง — `--crop-count` เท่าเดิม
จึงครอบคลุมเฟรมได้มากกว่าเดิมสี่เท่า ส่วน `--exclude-labels` ข้ามคู่ (frame, field) ที่รอบแรก
ตอบไปแล้ว รอบสองจึงเป็นการ**เพิ่ม** label ไม่ใช่ถามซ้ำ (แถวที่ `expected` ว่างไม่นับว่าตอบแล้ว
— นั่นคือ crop ที่ export ไปแล้วไม่ได้กรอก ซึ่งต้อง export ใหม่)

900 crops ÷ 3 ฟิลด์ = 300 เฟรม เทียบกับรอบแรกที่ได้ 83 เฟรม

**2. สร้าง contact sheet แล้วกรอกช่อง `expected`**

```bash
python scripts/label_contact_sheet.py output/labels-pass2
```

เปิดไฟล์ 900 ไฟล์ทีละไฟล์คือส่วนที่ช้าที่สุดของงานนี้ สคริปต์นี้เรียง crop ลงแผ่นละ 25 ภาพ
พร้อม frame index อยู่ข้างซ้าย อ่านทีเดียวทั้งจอแล้วพิมพ์ลง CSV ทีเดียว
(`--field ir_detail` ทำทีละฟิลด์ · โดยปกติข้ามแถวที่กรอกแล้ว ใช้ `--all-rows` ถ้าอยากเห็นทั้งหมด)

crop ถูกขยาย 3 เท่าแบบ nearest ไม่ resample ให้พอดีกริด เพราะภาพที่ถูกปรับขนาดคือภาพที่อ่านผิดได้

ไฟล์ `output/labels-pass2/labels.csv` มีคอลัมน์ชุดเดียวกับ
`examples/validation_labels_15-31-25-0.csv` พอดี จึงต่อท้ายกันได้ตรงๆ

| คอลัมน์ | ใครกรอก |
|---|---|
| `frame_index` `field_id` `crop_path` | export กรอกให้ |
| `expected` | **คุณ** — สิ่งที่เห็นในภาพ |
| `ocr_raw` `verified` | บันทึกของคุณเอง กรอกทีหลังหรือเว้นไว้ก็ได้ |

`ocr_raw` **ถูกเว้นว่างไว้ตั้งใจ** ไม่ได้เติมผล OCR ไว้ให้ — ตัวเลข 95.9% ที่วัดไว้เกิดจากการ
ดูภาพก่อนแล้วค่อยเทียบ ถ้าเห็นคำตอบของเครื่องก่อนจะถูกชี้นำโดยไม่รู้ตัว

เว้นแถวที่อ่านไม่ออกจริงๆ ไว้ว่าง `load_labels` ข้ามให้เอง ดีกว่าเดา

**3. รวมไฟล์แล้ววัดใหม่**

```bash
python -c "import pathlib; a=pathlib.Path('examples/validation_labels_15-31-25-0.csv').read_text(encoding='utf-8'); b=pathlib.Path('output/labels-pass2/labels.csv').read_text(encoding='utf-8'); pathlib.Path('examples/validation_labels_pass2.csv').write_text(a.rstrip()+chr(10)+chr(10).join(b.splitlines()[1:])+chr(10), encoding='utf-8')"
```

แล้วรัน `calibrate --labels examples/validation_labels_pass2.csv` เพื่อดูว่า preprocessing
ยังพอมีที่ให้ขยับไหม — **คาดว่าไม่มี** รอบแรก `eo_detail`/`ir_detail` อยู่ที่ 90.4% ทั้งก่อนและ
หลัง calibrate คือไม่ตอบสนองต่อ recipe ใดเลย ถ้ายืนยันซ้ำด้วยข้อมูลที่มากขึ้น ก็แปลว่าเหลือ
ทางเดียวคือ **fine-tune `PP-OCRv6_tiny_rec`** ด้วย crops ที่ label ไว้แล้ว ซึ่งเป็นงานฝั่ง
PaddleOCR ไม่ใช่ฝั่งโปรเจกต์นี้
