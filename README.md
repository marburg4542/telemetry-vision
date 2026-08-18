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
`extract` ใหม่ เพราะ voting ต้องการเฟรมที่ต่อเนื่องกัน ส่วน `telemetry.jsonl` ที่ปุ่มเขียนไว้
ตอน capture ใช้ผล voting ของสตรีมที่กำลังเดินอยู่ จึงเชื่อถือได้มากกว่าในกรณีนั้น

## Web UI สำหรับทดสอบวิดีโอ

### ติดตั้งเป็น PWA และเปิด OCR server อัตโนมัติ

1. ดับเบิลคลิก `install-ocr-web-autostart.cmd` หนึ่งครั้ง ระบบจะสร้าง Windows
   Scheduled Task ชื่อ `Telemetry OCR Web Server` และเปิด server ให้ทันที
2. เปิด <http://127.0.0.1:8765> ด้วย Edge หรือ Chrome
3. กดปุ่ม **ติดตั้งแอป** ที่มุมขวาบน หรือกดไอคอน Install ใน address bar
4. หลังติดตั้งแล้วเปิด **Telemetry OCR** จาก Start Menu/Desktop ได้เหมือนแอปทั่วไป

ครั้งถัดไป OCR server จะเปิดแบบซ่อนอัตโนมัติหลัง Login Windows จึงเปิด PWA แล้วใช้งานได้เลย
Scheduled Task นี้เปิดเฉพาะ standalone OCR web เท่านั้น ไม่เปิดหรือแก้ไข Flight Plan

หากต้องการถอน auto-start ให้ดับเบิลคลิก `remove-ocr-web-autostart.cmd` หรือรัน:

```powershell
.\scripts\install_windows_ocr_web_autostart.ps1 -Remove
```

Service worker เก็บเฉพาะหน้าแอปและ static assets เพื่อให้ PWA เปิดหน้าจอได้เร็ว แต่การอ่าน OCR
ยังต้องมี local server ทำงานอยู่ เพราะโมเดล PaddleOCR ประมวลผลใน Python บนเครื่อง

### ดูสตรีมใน PWA

เบราว์เซอร์เปิด RTSP ไม่ได้ ดังนั้นในโหมดสตรีม **หน้าที่จะสลับกัน**: server เป็นคนถอดภาพ
และรัน OCR ส่วนหน้าเว็บกลายเป็นผู้ชมผลลัพธ์ที่ตัวเองไม่ได้ผลิต

1. เปิด <http://127.0.0.1:8765>
2. ใส่ URL ในช่องบนซ้าย แล้วกด **เปิดสตรีม**
3. ภาพจะขึ้นทันที พร้อม telemetry ทางขวาแบบเดียวกับตอนเล่นไฟล์

ภาพถูกส่งเป็น **MJPEG** เข้า `<img>` โดยตรง ทำให้หน้าเว็บไม่ต้อง poll ภาพเลย — หนึ่ง
request ต่อหนึ่งสตรีม ไม่ใช่หนึ่ง request ต่อหนึ่งเฟรม ส่วน telemetry ใช้ long poll
ที่รอบน event จริง วัดจากเบราว์เซอร์ได้ 1 request ต่อ 1 revision พอดี

**Object detection ใช้เส้นทางเดิมทุกอย่าง** หน้าเว็บจับภาพจาก `<img>` แล้วส่งเข้า endpoint
เดิมที่โหมดไฟล์ใช้อยู่ YOLO จึงทำงานกับสตรีมผ่านโค้ดที่มีเทสต์อยู่แล้ว

ข้อจำกัดที่ตั้งใจ: **สตรีมที่กำลังรันเป็นเจ้าของ OCR** ถ้าเปิดอีกแท็บแล้วส่งเฟรมจากไฟล์
จะได้ 409 เพราะทั้งเครื่องมี PaddleOCR ตัวเดียว และสตรีมเรียกมันจาก thread ของตัวเอง

telemetry ยังถูกเขียนลง `output/stream-telemetry.jsonl` และ `.csv` เสมอ — หน้าเว็บเป็น
ผู้ชม ไม่ใช่ตัวบันทึก

### ให้เปิดสตรีมเองตอน login

ใส่ URL ลงใน [configs/stream-url.txt](configs/stream-url.txt) บรรทัดเดียว (บรรทัดที่ขึ้นต้น
ด้วย `#` ถูกข้าม) จากนั้น scheduled task จะเปิดกล้องให้เองตอน login — ไม่ต้องเปิดเบราว์เซอร์
ไม่ต้องพิมพ์คำสั่ง

```
rtsp://user:password@192.168.1.50:554/stream1
```

ระบบจะ**ลองใหม่ทุก 30 วินาที**ตราบใดที่ยังต่อไม่ได้ เพราะกล้องมักยังไม่พร้อมตอน Windows
login พอดี และสตรีมอาจหลุดนานกว่า backoff ของ decoder เอง หากกด **หยุดสตรีม** ใน UI
ตัว supervisor จะถูกยกเลิกด้วย — หยุดแล้วคือหยุด ไม่ใช่กลับมาเองใน 30 วินาที

ถ้าไม่ได้ตั้งค่าไฟล์นี้ server จะทำงานเหมือนเดิมทุกประการ

ดู log ได้ที่ `output/logs/ocr-web-autostart.log` ซึ่งบันทึกว่าเปิดสตรีมไหนไว้

### ปุ่ม Capture: เก็บภาพเฟรมนี้ไว้พร้อม telemetry ของมัน

ส่วนอื่นทั้งหมดของหน้านี้มีไว้ **ดู** — ตอบขณะวิดีโอเล่นแล้วไม่เก็บอะไรไว้ ปุ่ม Capture เป็นตัวเดียว
ที่**เขียนอะไรถาวร** คือภาพเฟรมนั้นพร้อมสิ่งที่ OCR อ่านได้จากมัน

- **กดครั้งเดียว** = เก็บ 1 เฟรม
- **กดค้าง** = เก็บต่อเนื่องทุก 400 ms จนกว่าจะปล่อย (ค่าที่กำลังเปลี่ยนมักหลุดถ้ากดทีละครั้ง)

ไฟล์จะไปอยู่ที่ `output/sessions/session-<วันเวลา>/` ในรูปแบบเดียวกับที่ `extract` อ่านได้:

```text
session-20260818-121223/
  session.json      แหล่งที่มา, เวลาเริ่ม, รายชื่อ field
  frames.jsonl      หนึ่งบรรทัดต่อหนึ่งเฟรม
  telemetry.jsonl   ทุกแถวมีคอลัมน์ image ชี้ไฟล์ภาพของมัน
  frames/00000000.jpg ...
```

ภาพที่เก็บเป็น**ความละเอียดเต็ม** (1920×1080) ไม่ใช่ภาพ preview ที่ย่อแล้ว และเก็บทั้งเฟรม
ไม่ใช่เฉพาะแถบ ROI — เพราะคำถามที่ capture มีไว้ตอบคือ "ตอนนั้นภาพเป็นยังไง"

**telemetry มาจากไหน** ขึ้นกับว่าใครถือวิดีโออยู่:

- **สตรีม** — server อ่านอยู่แล้ว จึงใช้ค่าที่เพิ่งอ่านได้ ไม่รัน OCR ซ้ำ เพราะจะเป็นสอง thread
  ใน PaddleOCR พร้อมกัน ซึ่งเป็นสิ่งที่ guard ของสตรีมมีไว้กัน และจะได้คำตอบต่างจากภาพเดียวกัน
- **ไฟล์ในเบราว์เซอร์** — ยังไม่มีใครอ่าน จึงส่งเฟรมมาให้ server อ่านตรงนั้น

ชื่อ session ถูกจำกัดเป็นตัวอักษร ตัวเลข จุด ขีด และขีดล่าง เพราะมันกลายเป็นชื่อโฟลเดอร์

### เปิด server เองโดยไม่ใช้ auto-start

เปิด local web server พร้อมโหลด PaddleOCR และ configuration ที่ calibrate จากวิดีโอจริง:

```powershell
.\.venv\Scripts\telemetry-ocr.exe web `
  --config configs\roi_calibrated.yaml `
  --open-browser
```

หากไม่ใช้ `--open-browser` ให้เปิด `http://127.0.0.1:8765` เอง จากนั้นกด **เลือกวิดีโอ** เพื่อ browse ผ่าน Explorer หรือ drag and drop ไฟล์ลงบนตัวเล่น วิดีโอต้นฉบับจะเล่นใน browser และค่าจาก OCR จะอัปเดตในแผงด้านขวา สามารถปรับ sampling rate เป็น 5–15 Hz, reset temporal voting และดาวน์โหลดผลสะสมเป็น JSONL/CSV ได้

ไฟล์วิดีโอเต็มไม่ถูกอัปโหลดไปที่ server: browser เล่นไฟล์จากเครื่องและส่งเฉพาะบริเวณ ROI ของเฟรมปัจจุบันไปยัง `127.0.0.1` เท่านั้น การส่ง OCR ใช้ one-request-in-flight จึงทิ้งรอบเก่าเมื่อ CPU ทำงานไม่ทันและไม่สร้าง backlog อย่างไรก็ตามวิดีโอที่เลือกต้องใช้ layout telemetry ตรงกับ configuration; หากตำแหน่ง overlay เปลี่ยนต้องสร้างหรือ calibrate config ใหม่

หมายเหตุ: codec ต้องเป็นชนิดที่ browser รองรับ โดย MP4/H.264 หรือ WebM ใช้งานได้กว้างที่สุด และการเปิดครั้งแรกอาจใช้เวลาสักครู่ระหว่างโหลดโมเดล OCR

### วาดกรอบจากโมเดล YOLO บนวิดีโอ

ในแผง **AI ANNOTATION** กด **Browse .pt** แล้วเลือก Ultralytics YOLO checkpoint เช่น
`models\yolo\best.pt` จาก Explorer ระบบจะตอบรับการอัปโหลดทันทีและแสดง
`loading` ระหว่าง child process โหลดโมเดล เมื่อสถานะเป็น `ready` จึงเริ่มส่งเฟรมและวาดกรอบ
ทับบนวิดีโอต้นฉบับ โดยไม่แก้ไขไฟล์วิดีโอจริง

จากการทดสอบกับ `15-31-25-0.mp4` checkpoint `models\yolo\best.pt` (ชื่อย่อ `best50`) รุ่น 50 epochs
ตรวจพบวัตถุที่ค่าเริ่มต้น `0.45` และรักษา Track ID ต่อเนื่องได้ ส่วน
`runs\visdrone_custom_6th\weights\best.pt` (ชื่อย่อ `run6`) รุ่น 100 epochs ไม่คืน detection แม้ลด confidence เป็น
`0.10` และสุ่มตรวจตลอดวิดีโอ จึงไม่แนะนำ checkpoint 100 epochs ตัวนั้นสำหรับวิดีโอนี้

YOLO/BoT-SORT ทำงานใน Windows child process ที่สร้างด้วย `spawn` แยกจาก web server และ
PaddleOCR โดยสิ้นเชิง จึงไม่ใช้ model object, inference thread หรือ lock ร่วมกัน หาก worker ล่ม
OCR และหน้าเว็บจะทำงานต่อ พร้อมแสดงปุ่ม **Restart worker**

- ใช้ `model.track(..., persist=True, tracker="botsort.yaml")` บน CPU และส่ง `track_id` กลับมาพร้อม class, confidence และ bounding box
- ค่าเริ่มต้นของหน้าเว็บคือ Frame submit **Auto**, confidence `0.45`, IoU `0.45`,
  image size ตามที่ checkpoint เทรนมา และจำกัด PyTorch/OpenCV ไว้ 2 CPU threads
- **Auto** กำหนดจังหวะส่งจาก round trip ที่วัดได้จริง เพราะการส่งเร็วกว่าที่ worker
  ระบายคิวได้ทำให้ผล **ช้าลง** ไม่ใช่เร็วขึ้น — เมื่อคิวล้น inference จะเริ่มบนเฟรมที่รออยู่แล้วเสมอ
  วัดที่ cycle 117 ms ได้ staleness 117 ms เทียบกับ 134 ms ที่ 10 Hz และ 184 ms ที่ 20 Hz
  เลือกอัตราคงที่ 5/10/15/20 Hz เองได้ถ้าต้องการเทียบ
- **image size ถูกตั้งให้ตรงกับที่ checkpoint เทรนมาโดยอัตโนมัติ** (อ่านจาก `trained_image_size`
  ของ worker) เพราะการรันคนละขนาดกับตอนเทรนทำให้ตำแหน่งกรอบคลาดเคลื่อนโดยตรง ไม่เกี่ยวกับ latency
  ถ้าเลือกเองแล้วไม่ตรง แผงสถานะจะเตือน บน CPU ค่าอัตโนมัติถูกจำกัดไว้ที่ 640 px
  เพราะ 960 px บน CPU ช้าเกินใช้งาน
- browser ย่อด้านยาวของภาพตาม image size ก่อนสร้าง JPEG; server รับเฟรมแบบ non-blocking และ queue เก็บได้หนึ่งเฟรม โดยเฟรมใหม่แทนเฟรมที่ค้างอยู่
- overlay คาด**เฉพาะจุดศูนย์กลาง** (คงขนาดกรอบจากผลล่าสุด เพราะ jitter ของ BoT-SORT
  อยู่ที่ขนาดเป็นหลัก) ระยะคาดผูกกับ cadence จริงที่ 2 เท่าของช่วงห่างระหว่างผล จำกัดที่ 250–600 ms
  และใช้คู่ตัวอย่างเฉพาะเมื่อห่างกัน 30–600 ms
- **กล่องที่เพิ่งเห็นครั้งแรกใช้การเคลื่อนที่ร่วมของกล้อง** ซึ่งประมาณจาก median ของทุกกล่องบนจอ
  เพราะเวลาอากาศยานเคลื่อน วัตถุทุกชิ้นในเฟรมเลื่อนไปด้วยกัน ค่านี้นิ่งกว่าการใช้สอง sample
  ของกล่องเดียว และใช้ได้ทันทีกับ track ID ที่เพิ่งตั้งใหม่หรือ detection ที่ tracker ไม่เคยให้ ID
  (สองกรณีนี้เดิมไม่ขยับเลย) เมื่อไม่มีอะไรวัดได้ ค่าจะสลายเข้าหาศูนย์แทนที่จะค้าง
- เวลาที่ใช้วัดความเก่าของผลมาจาก `requestVideoFrameCallback` ไม่ใช่ `video.currentTime`
  ซึ่งถูกปัดเป็นขอบเฟรมและอ่านช้ากว่าภาพจริง บวกล่วงหน้าอีกหนึ่งเฟรมเพราะสิ่งที่วาดจะปรากฏที่ repaint ถัดไป
  ส่วน fade กรอบที่หายภายใน 150 ms และซ่อนผลที่ช้ากว่าวิดีโอเกิน 750 ms ยังเหมือนเดิม
- tracker จะ reset เมื่อเลือก/seek วิดีโอ, เปลี่ยน confidence หรือ image size, เปลี่ยนโมเดล และเมื่อ source timestamp กระโดดเกินหนึ่งวินาที
- แผงสถานะแสดง detection FPS ที่ได้จริง, inference latency, result lag, **Capture**
  (เวลา encode JPEG ในเบราว์เซอร์), **Wait** (เวลาที่ผลนอนรออยู่ใน server) และจำนวนเฟรมที่ queue ทิ้ง
  ส่วนบรรทัดสถานะแยกให้เห็นทุกช่วง: `rate · cap · post · infer · transport · wait`
- checkbox **Track ID** เปิดให้แสดงเลขประจำตัวของ BoT-SORT ต่อท้ายชื่อ class ค่าเริ่มต้นปิด
  เลขนี้มีไว้ดีบัก tracker — ถ้ามันกระโดดไปมาแปลว่า tracker หลุดและจับเป็นวัตถุใหม่
- หากไม่มีกรอบ แผงสถานะจะระบุว่าโมเดลไม่พบวัตถุ หรือพบผลแต่ผลเกิน 750 ms จึงถูกซ่อน
- รอบ inference แรกมักช้ากว่ารอบถัดไป เพราะ PyTorch ต้อง warm up โมเดล
- อัปโหลดเฉพาะ `.pt` ที่เชื่อถือได้ เพราะ checkpoint ของ PyTorch สามารถบรรจุโค้ดที่ทำงานตอนโหลดได้
- โมเดลถูกเก็บใน temporary directory และลบเมื่อกดล้างโมเดลหรือปิด OCR server
- BoT-SORT ช่วยให้ track ID และตำแหน่งต่อเนื่องขึ้น แต่ไม่สามารถแก้ false positive หรือ class ที่โมเดลทำนายผิดได้

ติดตั้ง dependency สำหรับ AI annotation เพิ่มด้วย:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ai]"
```

`best.pt` บางไฟล์ที่บันทึกด้วย custom pickle ต้องใช้ `dill`; dependency นี้รวมอยู่ในชุด `ai` แล้ว
ชุดนี้ติดตั้ง `lap>=0.5.12` สำหรับ BoT-SORT ด้วย

Detection API ที่หน้าเว็บใช้งานมี lifecycle ดังนี้:

- `POST /api/detection/model` — เก็บโมเดลชั่วคราว, เปิด worker และตอบ `202 loading`
- `GET /api/detection/model` — อ่าน `loading/ready/error`, PID, classes, epochs, trained image size, tracker และ device
- `POST /api/detection/frame` — enqueue JPEG แล้วคืนผลล่าสุดทันที หรือ `202` หากยังไม่มีผล
- `GET /api/detection/result` — อ่านผลล่าสุดโดยไม่ส่งเฟรมซ้ำ รับ `after_revision`
  และ `wait_ms` (จำกัดที่ 1000) เพื่อค้าง request ไว้จนกว่าจะมีผลใหม่กว่า `after_revision`
  โดย `revision` เป็นตัวนับของ server เองที่ไม่ถอยหลัง client ส่งค่าที่ได้รับล่าสุดกลับไปตรงๆ
- `POST /api/detection/reset` — reset tracker เมื่อ source หรือเวลาเล่นเปลี่ยน
- `POST /api/detection/restart` — เริ่ม worker ใหม่หลัง process ล่ม

### การป้องกัน API

Server ตรวจสามชั้นก่อนเข้า handler เพราะทุก endpoint มี side effect ที่เกิดขึ้นก่อนที่
CORS จะตัดสินว่าใครอ่าน response ได้:

- `Host` ต้องเป็น `127.0.0.1`, `localhost`, `::1` หรือค่าที่ส่งผ่าน `--host` มิฉะนั้นตอบ `403`
  เพื่อกัน DNS rebinding
- request ที่มี `Origin` แต่ไม่ผ่าน allowlist ถูก **ปฏิเสธ** ที่ `403` ไม่ใช่แค่ไม่ใส่ CORS header
- ทุก `POST`/`DELETE` ใต้ `/api/` ต้องมี header `X-Telemetry-OCR: 1`

ข้อสุดท้ายสำคัญที่สุด: `multipart/form-data` เป็น CORS-safelisted content type จึงไม่เกิด
preflight ถ้าไม่บังคับ header นี้ เว็บใดก็ตามที่ผู้ใช้เปิดอยู่จะ POST ไฟล์ `.pt` เข้า
`/api/detection/model` ได้ และ worker โหลด checkpoint ด้วย pickle จึงเท่ากับรันโค้ดได้ทันที
header ที่ไม่อยู่ใน safelist บังคับให้ browser ส่ง preflight ที่ guard ปฏิเสธได้

หากเรียก API จาก web app อื่นต้องใส่ทั้ง `--allow-origin` ตอนเปิด server และส่ง header
`X-Telemetry-OCR: 1` ทุก request ที่เปลี่ยนสถานะ

ตัวเลขเดิมในหัวข้อนี้วัดจาก CPU ที่ 416 px (2.2 results/วินาที, lag 0.37–0.58 วินาที)
บน CUDA และหลังแก้เส้นทางส่งผลลัพธ์แล้ววัดใหม่ที่ **640 px, 10 Hz** บนวิดีโอจริงได้:

| ช่วง | ก่อนแก้ (ผลกลับมาพร้อม POST ถัดไป) | หลังแก้ (pump + long-poll) |
|---|---|---|
| `harvest` — ผลนอนรอใน parent | 64.8 ms (p95 81.6) | **0.2 ms** (p95 0.3) |
| round trip ทั้งหมด | 124.9 ms (p95 135.5) | **60.1 ms** (p95 74.9) |
| `inference` | 42.0 ms | 41.3 ms |
| `transport` — queue wait + IPC | 5.6 ms | 6.3 ms |

`harvest` คือขนาดของปัญหาโดยตรง: ไม่มีอะไร drain คิวผลลัพธ์จนกว่าจะมี HTTP request เข้ามา
ผลจึงรอครบหนึ่ง submit interval ก่อนจะเอาไปวาดได้ ตอนนี้ background pump เก็บให้ทันที

OCR ไม่ถูกกระทบ วัดพร้อมกันบนเครื่องเดียวกัน: latency median 39.1 ms เมื่อรัน OCR อย่างเดียว
เทียบกับ 39.8 ms ขณะมี detection ทำงานเต็มที่ (p95 44.1 → 55.0 ms ซึ่งยังต่ำกว่าเกณฑ์ 100 ms)

บนหน้าเว็บจริงที่ OCR ทำงานพร้อมกัน `inference` ขึ้นไปถึงราว 111 ms เพราะ PaddleOCR
กับ YOLO แย่ง GPU 4 GB ใบเดียวกัน ไม่ใช่เพราะโมเดลช้าลง — วัดตอน OCR ว่างได้ 30–41 ms
ถ้าต้องการ lag ต่ำกว่านี้อีกให้ลด sampling rate ของ OCR ลงเป็น 5–8 Hz ซึ่งคืน GPU
ให้ detection โดยตรง

**FP16 ใช้ไม่ได้บนการ์ดนี้** วัดที่ 640 px แล้ว FP32 เฉลี่ย 30.7 ms ส่วน FP16 เฉลี่ย
164.1 ms (ช้ากว่า 5 เท่า) โดยได้ track เท่ากันและกรอบต่างกันหนึ่งใบจาก 256 เฟรม
เพราะ GTX 1650 ไม่มี tensor core จึง emulate half precision `half=False` จึงเป็นค่าที่ถูกแล้ว

`post` วัดได้ 12.7 ms ซึ่งต่ำกว่า submit interval ที่ 100 ms มาก จึง**ไม่**เพิ่มจำนวน request
ที่ค้างได้พร้อมกัน เพราะ input queue เป็น newest-wins ขนาด 1 การส่งซ้อนจะไล่เฟรมที่ worker
ยังไม่ได้หยิบทิ้งโดยไม่ได้ latency ลดลง

ตัวเลขข้างบนวัดจาก harness ที่ไม่มีภาระฝั่ง browser ในหน้าเว็บจริงยังมี `Capture`
(การ encode JPEG) เพิ่มเข้ามา ซึ่งแผงสถานะแสดงแยกไว้แล้ว

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

ต่างจาก overlay ใน web UI ตรงที่คำสั่งนี้เดินทั้งไฟล์แบบไม่ต้องทัน realtime จึงใช้ image size
เต็มและได้ตัวเลขที่เทียบ checkpoint กันได้ ส่วน web UI ตอบคำถามว่ากรอบทันวิดีโอหรือไม่

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
