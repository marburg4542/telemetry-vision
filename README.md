# Telemetry OCR

ระบบถอด telemetry ที่อยู่ตำแหน่งคงที่ในวิดีโอด้วย PaddleOCR โดยเน้น CPU, latency ต่ำ และผลลัพธ์ที่นิ่งกว่าการ OCR เต็มเฟรมทุกเฟรม

Pipeline หลักคือ:

```text
video decoder (30 FPS)
  -> scheduler + bounded newest-frame queue (12 Hz)
  -> fixed ROI crops
  -> per-field color/contrast preprocessing
  -> batched PaddleOCR TextRecognition
  -> fallback recipe when confidence/format fails
  -> regex/range validation + temporal voting
  -> JSONL and CSV writer
```

เพราะตำแหน่งข้อความคงที่ ระบบจะไม่โหลด text detection, document orientation หรือ unwarping model

## ความต้องการระบบ

- Production ที่แนะนำ: WSL2/Ubuntu, Python 3.10–3.12 (แนะนำ 3.11), CPU x86-64
- Native Windows ใช้ได้เมื่อปิด `enable_hpi` แต่ประสิทธิภาพอาจต่ำกว่า
- PaddleOCR/PaddlePaddle และ OpenCV
- ROI สูงสุดเริ่มต้น 10 ช่อง, วิดีโอเป้าหมาย 1080p30, OCR 12 Hz

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
cd /mnt/c/Users/Lenovo/Documents/ChatGPT/ocr
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
`object_detection\best.pt` จาก Explorer ระบบจะตอบรับการอัปโหลดทันทีและแสดง
`loading` ระหว่าง child process โหลดโมเดล เมื่อสถานะเป็น `ready` จึงเริ่มส่งเฟรมและวาดกรอบ
ทับบนวิดีโอต้นฉบับ โดยไม่แก้ไขไฟล์วิดีโอจริง

จากการทดสอบกับ `15-31-25-0.mp4` checkpoint `object_detection\best.pt` รุ่น 50 epochs
ตรวจพบวัตถุที่ค่าเริ่มต้น `0.45` และรักษา Track ID ต่อเนื่องได้ ส่วน
`visdrone_custom_6th\weights\best.pt` รุ่น 100 epochs ไม่คืน detection แม้ลด confidence เป็น
`0.10` และสุ่มตรวจตลอดวิดีโอ จึงไม่แนะนำ checkpoint 100 epochs ตัวนั้นสำหรับวิดีโอนี้

YOLO/BoT-SORT ทำงานใน Windows child process ที่สร้างด้วย `spawn` แยกจาก web server และ
PaddleOCR โดยสิ้นเชิง จึงไม่ใช้ model object, inference thread หรือ lock ร่วมกัน หาก worker ล่ม
OCR และหน้าเว็บจะทำงานต่อ พร้อมแสดงปุ่ม **Restart worker**

- ใช้ `model.track(..., persist=True, tracker="botsort.yaml")` บน CPU และส่ง `track_id` กลับมาพร้อม class, confidence และ bounding box
- ค่าเริ่มต้นของหน้าเว็บคือส่งเฟรม 10 Hz, confidence `0.45`, IoU `0.45`, image size 416 px และจำกัด PyTorch/OpenCV ไว้ 2 CPU threads
- เลือก 416 px เพื่อให้กรอบทันวิดีโอเมื่อรันพร้อม OCR บน CPU เครื่องนี้; ยังเลือก 640/960 px ได้เมื่อต้องการเน้นความแม่นยำหรือใช้ CPU ที่เร็วกว่า
- browser ย่อด้านยาวของภาพตาม image size ก่อนสร้าง JPEG; server รับเฟรมแบบ non-blocking และ queue เก็บได้หนึ่งเฟรม โดยเฟรมใหม่แทนเฟรมที่ค้างอยู่
- overlay ใช้สองตำแหน่งล่าสุดของแต่ละ `track_id` เพื่อคาดตำแหน่งต่อไม่เกิน 250 ms, fade กรอบที่หายภายใน 150 ms และซ่อนผลที่ช้ากว่าวิดีโอเกิน 750 ms
- tracker จะ reset เมื่อเลือก/seek วิดีโอ, เปลี่ยน confidence หรือ image size, เปลี่ยนโมเดล และเมื่อ source timestamp กระโดดเกินหนึ่งวินาที
- แผงสถานะแสดง detection FPS ที่ได้จริง, inference latency, result lag และจำนวนเฟรมที่ queue ทิ้ง
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
- `GET /api/detection/result` — poll ผลล่าสุดโดยไม่ส่งเฟรมซ้ำ
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

ผลทดสอบพร้อม OCR บน laptop เครื่องนี้ด้วยโมเดล YOLOv8m 50 epochs ที่ 416 px ได้ประมาณ
2.2 detection results/วินาที, inference ราว 0.3–0.45 วินาที และ result lag ราว 0.37–0.58 วินาที
จึงยังอยู่ภายในเกณฑ์แสดงกรอบ 750 ms ส่วน 640 px เมื่อมี OCR ทำงานพร้อมกันวัด lag ได้ประมาณ
1.15 วินาทีและหน้าเว็บจะซ่อนกรอบเพื่อไม่ให้วาดผิดตำแหน่ง หากต้องการทั้ง 640 px และกรอบที่ลื่นจริง
ควรพิจารณา GPU ในรุ่นถัดไป

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
Native Windows is therefore below the strict 10 Hz / 100 ms-p95 acceptance
target. PaddleX HPI is Linux-only for this setup; WSL2/Ubuntu with HPI remains
the recommended deployment route when that strict target is mandatory.

An earlier figure of 9.90 Hz and 136.2 ms p95 is quoted in some notes. It was
measured when the config still had 10 fields (50,780 records over 5,078 frames
is exactly 10 per frame), so it is not comparable: 12 fields is 20% more OCR
work per frame and p95 rose 20.4%, which accounts for the whole difference.

These timing numbers are measured. Accuracy is not claimed from format validity:
the delivery probe passed 79/80 sampled ROIs, but a manually labeled 1,000-crop
validation set is still required before claiming the 97% exact-field target.
Crops for that set are exported at `output/validation/` (996 images across the
12 fields) and need their `expected` column filled in; rows for the `timestamp`
field can be left blank because it is derived from source time, not recognized.
