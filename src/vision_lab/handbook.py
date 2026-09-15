"""คู่มือฉบับเต็มของโปรเจกต์: ทฤษฎี หลักการทำงาน และเหตุผลของทุกค่าที่ตั้งไว้

ต่างจาก report.py ซึ่งอธิบายเฉพาะทฤษฎี YOLOv8 กับ dataset เอกสารนี้ครอบคลุม
**ทั้งสองฝั่ง** — ทั้งเส้นทาง OCR และเส้นทาง object detection — และไล่อธิบายทุก
พารามิเตอร์ว่าคืออะไร ทำไมถึงเลือก และทำไมถึงเป็นค่านั้น

ทุกตัวเลขในเอกสารนี้ดึงมาจากไฟล์จริงในโปรเจกต์ (configs, args.yaml, results.csv,
ค่าคงที่ในโค้ด) หรือจากการวัดที่บันทึกไว้ใน README ไม่มีตัวเลขไหนที่เขียนจากความจำ

ใช้โครงสร้าง HTML→Chrome→PDF ร่วมกับ report.py ด้วยเหตุผลเดียวกัน: ReportLab
ไม่มี complex text shaping จึงวางวรรณยุกต์ไทยที่ซ้อนบนสระทับกันจนอ่านไม่ออก
"""

from __future__ import annotations

from pathlib import Path

from telemetry_ocr.errors import ConfigError

from .paths import ProjectPaths
from .report import CSS, figure, formula, h1, h2, h3, note, p, page_break, table, ul

DEFAULT_PDF_NAME = "Telemetry_Vision_Handbook_TH.pdf"
FIGURE_RUN = "visdrone_custom_7th"


def _cover() -> str:
    """หน้าปกต้องจบด้วย page_break เสมอ

    `@page :first` ตั้ง margin เป็น 0 เพื่อให้แถบสีบนสุดชนขอบกระดาษ และ `.cover`
    ชดเชยด้วย padding ของตัวเอง ถ้าเนื้อหาบทแรกไหลต่อลงมาในหน้าเดียวกัน มันจะถูก
    จัดหน้าโดยไม่มี margin เลย — ตารางและข้อความจะชนขอบกระดาษทั้งสองด้าน
    """
    meta = table([], [
        ["ขอบเขต", "เส้นทาง OCR และเส้นทาง object detection ทั้งสองฝั่ง"],
        ["โมเดล", "PaddleOCR PP-OCRv6_tiny_rec · Ultralytics YOLOv8m + BoT-SORT"],
        ["ฮาร์ดแวร์ที่วัด", "NVIDIA GTX 1650 (4 GB, compute 7.5), Windows"],
        ["ที่มาของตัวเลข", "configs, args.yaml, results.csv และค่าคงที่ในโค้ด"],
    ]).replace("<thead><tr></tr></thead>", "")
    return (
        '<div class="cover-bar"></div>'
        '<div class="cover">'
        "<h1>คู่มือหลักการและทฤษฎี</h1>"
        '<div class="sub">Telemetry Vision — OCR และ Object Detection</div>'
        "<p>อธิบายการทำงานทุกขั้นตอน พร้อมเหตุผลว่าทำไมทุกค่าถึงเป็นค่านั้น</p>"
        f"{meta}"
        "</div>"
        f"{page_break()}"
    )


def _overview() -> str:
    s = [h1("1. ภาพรวมของระบบ")]
    s.append(p(
        "โปรเจกต์นี้ทำงานกับวิดีโอจากอากาศยานไร้คนขับ และแยกเป็นสองเส้นทางที่เป็นอิสระ"
        "จากกันโดยสิ้นเชิง ทำงานพร้อมกันบนเฟรมเดียวกัน แต่ตอบคำถามคนละข้อ"
    ))
    s.append(table(
        ["", "เส้นทาง OCR", "เส้นทาง Object Detection"],
        [
            ["คำถามที่ตอบ", "ตัวเลข telemetry บนหน้าจอคืออะไร", "ในภาพมีวัตถุอะไรอยู่ตรงไหน"],
            ["ตำแหน่งที่สนใจ", "<b>คงที่</b> — OSD วางตำแหน่งเดิมทุกเฟรม", "<b>ไม่คงที่</b> — วัตถุอยู่ที่ไหนก็ได้"],
            ["โมเดล", "PaddleOCR PP-OCRv6_tiny_rec", "Ultralytics YOLOv8m + BoT-SORT"],
            ["Framework", "PaddlePaddle", "PyTorch"],
            ["ที่รัน", "process ของ web server", "spawn child process แยก"],
            ["ผลลัพธ์", "ค่าที่ผ่านการยืนยันแล้ว (JSONL/CSV)", "กรอบวาดทับวิดีโอ"],
        ],
        widths=["18%", "41%", "41%"],
    ))
    s.append(h3("ความต่างที่กำหนดสถาปัตยกรรมทั้งหมด"))
    s.append(p(
        "ข้อแตกต่างสำคัญที่สุดคือ <b>ตำแหน่งของสิ่งที่สนใจ</b> ข้อความ telemetry อยู่ที่เดิม"
        "ทุกเฟรมเพราะเป็น overlay ที่กล้องวาดลงไป ส่วนวัตถุในภาพจะอยู่ที่ไหนก็ได้"
    ))
    s.append(p(
        "ผลคือฝั่ง OCR <b>ไม่ต้องใช้ text detection เลย</b> — ไม่ต้องหาว่าข้อความอยู่ตรงไหน"
        "เพราะเรารู้อยู่แล้ว จึงข้ามโมเดล detection, document orientation และ unwarping ทิ้งทั้งหมด "
        "เหลือแค่ text recognition ทำงานบน crop เล็กๆ ที่กำหนดไว้ ซึ่งเป็นเหตุผลว่าทำไม OCR "
        "ถึงทำงานได้ที่ 10 Hz บนการ์ดจอระดับล่าง ขณะที่การ OCR เต็มเฟรมทุกเฟรมทำไม่ได้"
    ))
    s.append(note("ข้อจำกัดที่ตามมา", [
        "เพราะระบบเชื่อว่าตำแหน่งคงที่ ถ้า layout ของ OSD เปลี่ยน หรือวิดีโอถูกหมุน/ครอป "
        "ค่า ROI ทั้งหมดจะผิดทันทีและต้อง calibrate ใหม่ — นี่คือราคาที่จ่ายเพื่อความเร็ว",
    ]))
    s.append(h3("ทำไมสอง framework ถึงอยู่คนละ process"))
    s.append(p(
        "PaddlePaddle และ PyTorch ต่างพก cuDNN 9 มาเอง บน Windows ตัวที่โหลดทีหลัง"
        "<b>พังเสมอ ไม่ว่าจะสลับลำดับอย่างไร</b> (<code>WinError 127</code> ที่ "
        "<code>cudnn_cnn64_9.dll</code>) ปัญหาซับซ้อนขึ้นเพราะ <code>paddleocr</code> ลาก torch "
        "เข้ามาทางอ้อมผ่าน <code>paddlex → modelscope</code> การเปิด GPU เฉยๆ จึงทำให้ Paddle "
        "พังก่อนจะสร้างโมเดลด้วยซ้ำ"
    ))
    s.append(p(
        "ทางออกมีสองชั้น: process ที่ถือ Paddle จะ<b>ซ่อน torch จากตัวเอง</b> "
        "(<code>sys.modules[\"torch\"] = None</code>) และ YOLO ถูกย้ายไปอยู่ใน process ที่สร้าง"
        "ด้วย <code>spawn</code> แยกต่างหาก ซึ่งโหลด cuDNN ของ torch ได้อย่างอิสระ"
    ))
    s.append(note("ผลข้างเคียงที่ต้องระวังตลอด", [
        "ข้อมูลที่ส่งข้าม process ต้องเป็น <b>builtin ล้วน</b> เพราะการ unpickle จะ import "
        "โมดูลที่นิยามคลาสนั้น ตัวอย่างเช่น <code>torch.__version__</code> เป็น subclass ของ "
        "<code>str</code> การส่งมันกลับมาจะลาก torch เข้าไปใน process ของ Paddle และทำให้ OCR พัง",
        "มีเทสต์ <code>DetectionBoundaryTests</code> เดินตรวจทุก field ของทุกข้อความที่ข้าม "
        "boundary เพื่อคุมเรื่องนี้โดยเฉพาะ",
    ], kind="warn"))
    return "".join(s)


def _ocr_pipeline() -> str:
    s = [page_break(), h1("2. เส้นทาง OCR — ทีละขั้นตอน")]
    s.append(formula(
        "video decoder (30 FPS) → scheduler + newest-frame queue (10.1 Hz) → ROI crops → "
        "preprocessing ต่อ field → PaddleOCR batch → validation → temporal voting → JSONL/CSV"
    ))

    s.append(h2("2.1 Decoder และ scheduler"))
    s.append(p(
        "วิดีโอ 1080p30 ถูก decode ทุกเฟรม แต่ OCR ทำงานแค่ 10.1 ครั้งต่อวินาที "
        "scheduler จึงเลือกเฉพาะเฟรมที่ถึงกำหนด ส่วนเฟรมที่เหลือถูกข้ามโดยไม่แปลงสี "
        "(การ copy และแปลง BGR ทำเฉพาะเฟรมที่จะเข้า OCR จริงเท่านั้น)"
    ))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "คืออะไร และทำไมเป็นค่านี้"],
        [
            ["<code>ocr_hz</code>", "10.1",
             "จำนวนครั้งที่ OCR ทำงานต่อวินาที เลือกจากการวัดจริง: เกณฑ์คือ 10 Hz และ p95 "
             "ไม่เกิน 100 ms ค่านี้ทำได้ 10.10 Hz จริงโดย dropped frames เท่ากับ 1 "
             "ส่วน 12 Hz เป็นเป้าหมายระยะยาวที่ฮาร์ดแวร์ปัจจุบันยังไปไม่ถึงอย่างเสถียร"],
            ["<code>queue_size</code>", "1",
             "ความลึกของคิวระหว่าง decoder กับ OCR <b>ตั้งเป็น 1 โดยตั้งใจ</b> เพื่อให้เก็บ"
             "เฉพาะเฟรมล่าสุด ถ้าคิวลึกกว่านี้ เมื่อ OCR ช้ากว่ากำหนด latency จะสะสมขึ้นเรื่อยๆ "
             "แทนที่จะทิ้งเฟรมเก่า — ยอมทิ้งเฟรมดีกว่ายอมให้ผลลัพธ์ล้าหลังภาพ"],
            ["<code>realtime</code>", "true",
             "เดินตาม timestamp ของไฟล์ ไม่ decode เร็วกว่าเวลาจริง ตั้งเป็น false เมื่อต้องการ"
             "ประมวลผลไฟล์ให้เร็วที่สุดและยอมให้เฟรมที่ค้างถูกทิ้ง"],
            ["<code>max_rois</code>", "12",
             "เพดานจำนวน ROI ต่อเฟรม กันไม่ให้ config ที่ใส่ field มากเกินไปทำให้ batch "
             "ใหญ่จน latency พุ่ง ค่าปัจจุบันพอดีกับ 12 field ที่ใช้จริง"],
        ],
        widths=["20%", "10%", "70%"],
    ))

    s.append(h2("2.2 การครอป ROI"))
    s.append(p(
        "แต่ละ field กำหนดด้วย <code>box = [x, y, width, height]</code> ที่เป็น<b>สัดส่วน "
        "0.0–1.0</b> เทียบขนาดเฟรม ไม่ใช่พิกเซล เพื่อให้ config เดิมใช้ได้กับวิดีโอความละเอียดอื่น"
    ))
    s.append(formula("x = 826 / 1920 = 0.4302 &nbsp;&nbsp; y = 842 / 1080 = 0.7796"))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "คืออะไร และทำไมเป็นค่านี้"],
        [
            ["<code>padding</code>", "0.02",
             "ขยายกรอบออกรอบด้านเป็นสัดส่วนของขนาด ROI เอง (2%) เผื่อให้ตัวอักษรไม่ถูกตัดขอบ "
             "ค่าเริ่มต้นในโค้ดคือ 0.08 แต่ config ที่ calibrate แล้วลดเหลือ 0.02 เพราะ ROI "
             "ถูกวัดมาแม่นแล้ว การเผื่อมากเกินไปจะดึงพื้นหลังและตัวอักษรข้างเคียงเข้ามารบกวน"],
            ["<code>aspect_bucket_base</code>", "10.0",
             "ฐานสำหรับจัดกลุ่ม crop ตามอัตราส่วนก่อนเข้า batch เพราะ PaddleOCR ปรับความสูง"
             "ให้เท่ากันแล้วยืดความกว้าง crop ที่อัตราส่วนใกล้กันจึง pad น้อยกว่าและเร็วกว่า"],
        ],
        widths=["24%", "10%", "66%"],
    ))
    s.append(note("หลักการเลือกกรอบ", [
        "ควรครอบเฉพาะ<b>ตัวค่า</b> เช่น <code>123.4</code> ไม่รวมคำว่า <code>SPEED</code> "
        "หรือหน่วย <code>km/h</code> เมื่อไม่จำเป็น เพราะทุกอักขระที่เกินมาคือโอกาสที่ regex "
        "จะไม่ผ่านและต้องเข้า fallback",
    ]))

    s.append(page_break())
    s.append(h2("2.3 Preprocessing — หัวใจของความแม่นยำ"))
    s.append(p(
        "ตัวอักษร OSD มักบางและมีสีใกล้พื้นหลัง การส่ง crop ดิบเข้าโมเดลจึงมักอ่านผิด "
        "ทุก field มี <b>recipe</b> ของตัวเองที่เลือกมาจากการวัดจริง ไม่ใช่ค่ากลางค่าเดียวใช้ทุกช่อง"
    ))
    s.append(h3("ขั้นตอนใน recipe หนึ่งชุด"))
    s.append(table(
        ["ขั้น", "พารามิเตอร์", "ค่าเริ่มต้น", "ทำอะไร และทำไม"],
        [
            ["1. เลือก channel", "<code>channel</code>", "lab_l",
             "เลือกช่องสีที่แยกตัวอักษรออกจากพื้นหลังได้ดีที่สุด มีให้เลือก 10 แบบ "
             "(color, gray, b, g, r, lab_l, lab_a, lab_b, hsv_s, hsv_v) — ตัวอักษรสีเหลือง"
             "บนพื้นเขียวอาจแยกได้ชัดใน channel เดียวและจมหายในอีก channel"],
            ["2. ขยายภาพ", "<code>scale</code>", "2.0",
             "ขยาย crop สองเท่าด้วย interpolation ก่อนเข้าโมเดล เพราะ crop จาก ROI เล็กมาก "
             "(สูงราว 30 px) และโมเดลถูกเทรนกับตัวอักษรที่ใหญ่กว่านั้น"],
            ["3. CLAHE", "<code>clahe_clip</code> / <code>clahe_grid</code>", "2.0 / 8",
             "ปรับ contrast แบบเฉพาะที่ (Contrast Limited Adaptive Histogram Equalization) "
             "clip 2.0 จำกัดการขยาย contrast ไม่ให้ noise ระเบิด ส่วน grid 8×8 คือความละเอียด"
             "ของการแบ่งพื้นที่ — ละเอียดกว่านี้จะไวต่อ noise เกินไปบน crop ขนาดนี้"],
            ["4. Unsharp mask", "<code>unsharp_amount</code>", "0.8",
             "เพิ่มความคมของขอบตัวอักษรด้วยการลบภาพเบลอออกจากภาพต้นฉบับ 0.8 คือความแรง "
             "มากกว่านี้จะสร้าง halo รอบตัวอักษรซึ่งทำให้ recognition แย่ลง"],
            ["5. Morphology", "<code>morphology</code>", "none",
             "<code>tophat</code> ดึงวัตถุสว่างบนพื้นมืด <code>blackhat</code> ทำตรงข้าม "
             "ใช้เมื่อพื้นหลังมีการไล่เฉดที่รบกวน ค่าเริ่มต้นปิดไว้เพราะส่วนใหญ่ไม่จำเป็น"],
            ["6. Threshold", "<code>threshold</code>", "none",
             "<code>otsu</code> หาเกณฑ์ขาวดำอัตโนมัติทั้งภาพ <code>adaptive</code> หาเป็นบริเวณ "
             "(<code>block_size</code> 21, <code>C</code> 5.0) เหมาะเมื่อแสงไม่สม่ำเสมอ "
             "การทำ threshold ทิ้งข้อมูลไปมาก จึงเปิดเฉพาะเมื่อวัดแล้วว่าช่วยจริง"],
        ],
        widths=["14%", "20%", "10%", "56%"],
    ))
    s.append(h3("การ calibrate เลือก recipe อย่างไร"))
    s.append(p(
        "คำสั่ง <code>telemetry-ocr calibrate</code> ทดลอง <b>17 recipe ต่อ field</b> "
        "ครอบคลุม baseline, RGB/Lab/HSV channels, CLAHE, unsharp, top-hat, black-hat, "
        "Otsu และ adaptive threshold ทั้งสอง polarity แล้วเลือกตัวที่ดีที่สุด"
    ))
    s.append(note("เกณฑ์การเลือกสำคัญกว่าจำนวน recipe", [
        "ถ้ามี labels: เลือกจาก <b>exact-match accuracy</b> ก่อน แล้วค่อยดู confidence และ latency",
        "ถ้าไม่มี labels: เลือกจาก valid-rate (ผ่าน regex/range) ซึ่ง<b>อ่อนกว่ามาก</b> "
        "เพราะค่าที่ผิดแต่รูปแบบถูกจะผ่านไปได้ — จึงควรทำ labeled set อย่างน้อย 1,000 crops",
    ]))
    s.append(p(
        "ผลลัพธ์ที่ calibrate ได้จริงบนวิดีโอเป้าหมายกระจายตัวชัดเจน ไม่ได้ลงเอยที่ recipe "
        "เดียวกันทุก field ซึ่งเป็นหลักฐานว่าการเลือกต่อ field มีความหมาย:"
    ))
    s.append(table(
        ["recipe ที่ถูกเลือก", "field ที่ใช้"],
        [
            ["<code>color_baseline</code>", "timestamp, eo_detail, ir_detail, sensor_angle_deg, "
             "elevation_deg, tag_lat_dms, tag_lon_dms, tag_alt_m, acft_lon_dms"],
            ["<code>color_clahe</code>", "lrf_m, acft_alt_m"],
            ["<code>gray_clahe</code>", "acft_lat_dms"],
        ],
        widths=["25%", "75%"],
    ))

    s.append(page_break())
    s.append(h2("2.4 โมเดล recognition"))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "คืออะไร และทำไมเป็นค่านี้"],
        [
            ["<code>model_name</code>", "PP-OCRv6_tiny_rec",
             "โมเดล text recognition เลือกเพราะ benchmark บนเครื่องนี้ให้ latency ต่ำกว่า "
             "<code>en_PP-OCRv5_mobile_rec</code> อย่างชัดเจน ถ้า labeled accuracy ของวิดีโอจริง"
             "ต่ำกว่าเป้าหมาย ให้สลับไป v5 แล้ว benchmark ใหม่"],
            ["<code>batch_size</code>", "12",
             "จำนวน crop ที่ส่งเข้าโมเดลพร้อมกัน ตั้งให้เท่ากับจำนวน ROI พอดี เพื่อให้ทั้งเฟรม"
             "เข้าโมเดลครั้งเดียว ไม่ต้องเสียค่า overhead ของการเรียกหลายรอบ"],
            ["<code>warmup_iterations</code>", "2",
             "รันโมเดลทิ้ง 2 รอบตอนเริ่ม เพื่อไม่ให้การ compile kernel ครั้งแรกไปปนกับการวัด "
             "latency จริง รอบแรกช้ากว่ารอบถัดไปเสมอ"],
            ["<code>device</code>", "gpu",
             "config CUDA ใช้ gpu ส่วน config CPU ใช้ cpu — วัดแล้ว GPU ให้ p95 46.6 ms "
             "เทียบกับ CPU 164.0 ms ซึ่งเป็นความต่างระหว่างผ่านเกณฑ์กับไม่ผ่าน"],
            ["<code>enable_mkldnn</code>", "false (GPU) / true (CPU)",
             "oneDNN เป็นไลบรารีเร่งความเร็วสำหรับ <b>CPU</b> จึงต้องปิดเมื่อรันบน GPU"],
            ["<code>enable_hpi</code>", "false",
             "High Performance Inference ต้องติดตั้ง dependency เพิ่มซึ่งบน native Windows "
             "ยังไม่พร้อม จึงปิดไว้ เปิดได้เมื่อรันบน WSL/Docker ที่ลง HPI แล้ว"],
            ["<code>cpu_threads</code>", "4",
             "จำนวน thread ที่ Paddle ใช้บน CPU เลือกจากคำสั่ง <code>benchmark</code> ที่ทดลอง"
             "หลายค่าแล้วเลือกตัวที่ p95 ต่ำสุด"],
            ["<code>recognizer_workers</code>", "1",
             "จำนวนอินสแตนซ์ของโมเดลที่ทำงานขนาน บน CUDA ตั้งเป็น 1 เพราะ <b>VRAM 4 GB "
             "ต้องแบ่งกับ YOLO</b> การเพิ่มเป็น 2 จะแย่ง VRAM จนทั้งสองฝั่งช้าลง"],
        ],
        widths=["22%", "16%", "62%"],
    ))

    s.append(h2("2.5 Validation — จากข้อความเป็นค่าที่เชื่อถือได้"))
    s.append(p(
        "ข้อความที่ OCR อ่านได้จะยังไม่ถูกใช้ทันที ต้องผ่านการตรวจหลายชั้นก่อน"
    ))
    s.append(table(
        ["พารามิเตอร์", "ทำอะไร"],
        [
            ["<code>allowed_chars</code>", "อักขระที่ field นี้ยอมรับ ตัวอื่นถือว่าอ่านผิด"],
            ["<code>regex</code>", "รูปแบบเต็มของค่า เช่น พิกัด DMS ต้องมีองศา ลิปดา ฟิลิปดาครบ"],
            ["<code>data_type</code>", "string, integer, float หรือ dms — กำหนดวิธีแปลงเป็นค่า"],
            ["<code>min_value</code> / <code>max_value</code>",
             "ช่วงที่เป็นไปได้จริง ค่าที่หลุดช่วงถูกปฏิเสธแม้รูปแบบจะถูก"],
            ["<code>max_delta_per_second</code>",
             "อัตราการเปลี่ยนสูงสุดที่เป็นไปได้ ใส่เฉพาะเมื่อทราบข้อจำกัดจริงของค่านั้น"],
            ["<code>dms_degrees_digits</code>",
             "จำนวนหลักขององศา — ละติจูด 2 หลัก ลองจิจูด 3 หลัก ใช้ตรวจความถูกต้องของรูปแบบ"],
        ],
        widths=["28%", "72%"],
    ))
    s.append(h3("การซ่อมอักขระที่ฟอนต์ OSD อ่านกำกวม"))
    s.append(p(
        "ฟอนต์ OSD ทำให้เกิดความสับสนซ้ำๆ แบบเดิม เช่น <code>O→0</code>, <code>I/l→1</code>, "
        "<code>S→5</code>, <code>B→8</code> ระบบจะพยายามซ่อมเหล่านี้ "
        "<b>เฉพาะเมื่อผลเดิมไม่ผ่าน validation เท่านั้น</b>"
    ))
    s.append(note("ทำไมต้องซ่อมทีหลัง ไม่ใช่ซ่อมทุกครั้ง", [
        "ถ้าซ่อมทุกครั้งโดยไม่ดูว่าผลเดิมผ่านหรือไม่ ค่าที่อ่านถูกอยู่แล้วอาจถูกแก้ให้ผิด "
        "การซ่อมจึงเป็นทางเลือกสุดท้าย ไม่ใช่ขั้นตอนปกติ",
        "มีกรณีที่ระบบ<b>ปฏิเสธที่จะเดา</b> โดยเจตนา เช่นเมื่อจุดทศนิยมหายไปและมีความเป็นไปได้"
        "หลายทางที่ผ่าน regex เท่ากัน การเดาผิดอันตรายกว่าการไม่มีค่า",
    ], kind="warn"))
    s.append(h3("Fallback recipe"))
    s.append(p(
        "เมื่อ confidence ต่ำกว่า <code>confidence_threshold</code> = <b>0.65</b> หรือไม่ผ่าน "
        "regex/range ระบบจะประมวลผล ROI นั้นซ้ำด้วย fallback recipe แล้ว<b>เลือกความถูกต้อง"
        "ก่อน confidence</b> — ผลที่ผ่าน validation ชนะผลที่ confidence สูงกว่าแต่ไม่ผ่านเสมอ"
    ))
    s.append(p(
        "ค่าเริ่มต้นในโค้ดคือ 0.80 แต่ config ที่ calibrate แล้วลดเหลือ 0.65 และตั้ง "
        "<code>enable_fallback: false</code> เพราะเมื่อ recipe หลักถูกเลือกมาดีแล้ว "
        "การเปิด fallback ทำให้เสียเวลาประมวลผลซ้ำโดยได้ประโยชน์น้อย"
    ))
    return "".join(s)


def _temporal() -> str:
    s = [page_break(), h1("3. Temporal voting — ชั้นที่กันการอ่านผิดรายเฟรม")]
    s.append(p(
        "OCR อ่านผิดเป็นครั้งคราวเป็นเรื่องปกติ ถ้าแสดงผลทุกค่าที่อ่านได้ทันที ตัวเลขบนหน้าจอ"
        "จะกระพริบไปมา ระบบจึงไม่ยืนยันค่าจนกว่าจะเห็นค่าเดียวกันซ้ำในหน้าต่างเวลาที่กำหนด"
    ))
    s.append(table(
        ["profile", "window", "required", "หมายความว่า", "ใช้กับ"],
        [
            ["<code>fast</code>", "3", "2",
             "เห็นค่าเดียวกัน 2 ใน 3 เฟรมล่าสุดจึงยืนยัน",
             "11 field ที่เหลือทั้งหมด"],
            ["<code>stable</code>", "5", "3",
             "เห็นค่าเดียวกัน 3 ใน 5 เฟรมล่าสุดจึงยืนยัน",
             "timestamp"],
        ],
        widths=["12%", "10%", "10%", "38%", "30%"],
    ))
    s.append(h3("ราคาที่จ่ายคือ latency และวัดได้"))
    s.append(p(
        "การยืนยันต้องรอเฟรมเพิ่ม ซึ่งแปลงเป็นเวลาได้ตรงๆ ที่ <code>ocr_hz</code> = 10.1 "
        "หนึ่ง tick กินเวลา 99 ms:"
    ))
    s.append(table(
        ["profile", "ต้องรอเพิ่ม", "เวลาจริงที่วัดได้"],
        [
            ["<code>fast</code>", "1 tick", "<b>100 ms</b> (median และ p95 เท่ากัน)"],
            ["<code>stable</code>", "2 ticks", "<b>200 ms</b> (median และ p95 เท่ากัน)"],
        ],
        widths=["20%", "20%", "60%"],
    ))
    s.append(note("ค่าที่วัดตรงกับทฤษฎีพอดี", [
        "วัดจากคลิปจริง 60 วินาที (606 เฟรม, 7,272 records) — ค่าที่ได้ตรงกับ 1 และ 2 tick "
        "เป๊ะ ซึ่งยืนยันว่ากลไก voting ทำงานตามที่ออกแบบไว้ ไม่มีความล่าช้าแฝงอื่น",
    ], kind="ok"))
    s.append(h3("กรณีพิเศษ: ค่าที่เปลี่ยนทุก tick"))
    s.append(p(
        "การโหวตแบบ exact match ใช้ไม่ได้กับค่าที่ไม่เคยซ้ำเลย เช่นพิกัดที่ขยับตลอด "
        "ระบบจึงใช้ <b>median ข้ามเฟรม</b> แทน และต้องมี observation ที่ valid "
        "<b>อย่างน้อย 3 ค่า</b> จึงจะยืนยันได้"
    ))
    s.append(formula(
        "median ของ 2 ค่า = ค่าที่ใหม่กว่าเสมอ → ไม่ได้กันค่าผิดเลย → ต้องมีอย่างน้อย 3"
    ))
    s.append(p(
        "ผลคือ field แบบ fast ยืนยันช้าลงหนึ่ง tick และวัดแล้วมี tick ที่ <code>confirmed</code> "
        "ลดลง 0.3–8.3% แล้วแต่ field โดยค่าที่รายงานยังเป็นค่า confirmed ล่าสุดเหมือนเดิม"
    ))
    s.append(h3("สิ่งที่พบจากการวัดจริง"))
    s.append(table(
        ["field", "สัดส่วน tick ที่ confirmed"],
        [
            ["<code>tag_alt_m</code>, <code>acft_alt_m</code>", "99.2%"],
            ["<code>timestamp</code>", "99.5%"],
            ["<code>elevation_deg</code>", "98.0%"],
            ["<code>acft_lon_dms</code>", "97.9%"],
            ["<code>sensor_angle_deg</code>", "95.0%"],
            ["<code>ir_detail</code>", "93.6%"],
            ["<code>eo_detail</code>", "86.3%"],
            ["<code>lrf_m</code>", "<b>30.0%</b>"],
        ],
        widths=["45%", "55%"],
    ))
    s.append(note("lrf_m คือจุดที่ควรตรวจต่อ", [
        "field อื่นอยู่ที่ 86–99% แต่ <code>lrf_m</code> ผ่านเกณฑ์แค่ 30% แปลว่าการอ่าน LRF "
        "ไม่นิ่งพอจะผ่าน 2 ใน 3 เกือบตลอดเวลา ค่าที่แสดงจึงเป็นค่าเก่าที่ยังยืนยันอยู่เสียส่วนใหญ่",
        "นี่เป็นปัญหาคุณภาพการอ่าน ไม่ใช่ latency — ควรตรวจ ROI, recipe หรือ threshold ของ field นี้",
    ], kind="warn"))
    return "".join(s)


def _detection(run_dir: Path) -> str:
    s = [page_break(), h1("4. เส้นทาง Object Detection")]

    s.append(h2("4.1 YOLOv8 ทำงานอย่างไรโดยย่อ"))
    s.append(p(
        "YOLO (You Only Look Once) มองภาพทั้งภาพครั้งเดียวแล้วทำนายกรอบและคลาสพร้อมกัน "
        "ต่างจากวิธีสองขั้นที่ต้องเสนอบริเวณก่อนแล้วค่อยจำแนก จึงเร็วกว่ามากและเหมาะกับงาน realtime"
    ))
    s.append(p(
        "YOLOv8 เป็นแบบ <b>anchor-free</b> คือทำนายระยะจากจุดกึ่งกลางไปยังขอบกรอบโดยตรง "
        "ไม่ต้องออกแบบ anchor box ล่วงหน้า และใช้ <b>Distribution Focal Loss (DFL)</b> "
        "ซึ่งทำนายขอบกรอบเป็นการแจกแจงความน่าจะเป็นแทนตัวเลขเดียว ทำให้ขอบกรอบแม่นขึ้น"
    ))
    s.append(h3("Loss สามส่วนและน้ำหนักที่ใช้"))
    s.append(table(
        ["ส่วน", "น้ำหนัก", "ควบคุมอะไร"],
        [
            ["<code>box</code>", "7.5", "ความแม่นของตำแหน่งและขนาดกรอบ (CIoU loss)"],
            ["<code>cls</code>", "0.5", "ความถูกต้องของคลาส"],
            ["<code>dfl</code>", "1.5", "ความคมชัดของการแจกแจงขอบกรอบ"],
        ],
        widths=["16%", "12%", "72%"],
    ))
    s.append(p(
        "ทั้งสามค่าเป็นค่าเริ่มต้นของ Ultralytics ที่ไม่ได้ปรับ เพราะการปรับ loss weight "
        "ควรทำหลังจากแก้ปัญหาเชิงโครงสร้าง (ข้อมูล, augmentation, learning rate) จนหมดแล้ว"
    ))

    s.append(h2("4.2 พารามิเตอร์การเทรน"))
    s.append(p(
        f"ตัวเลขทั้งหมดในหัวข้อนี้ดึงจาก <code>runs/{FIGURE_RUN}/args.yaml</code> ซึ่งเป็น"
        "รอบล่าสุดและให้ผลดีที่สุด"
    ))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "คืออะไร และทำไมเป็นค่านี้"],
        [
            ["<code>model</code>", "yolov8m.pt",
             "ขนาด medium เป็นจุดสมดุลระหว่างความแม่นกับความเร็วบน GTX 1650 "
             "ตัวเล็กกว่าเร็วกว่าแต่พลาดวัตถุเล็กในภาพมุมสูง ตัวใหญ่กว่าช้าเกินไปสำหรับ realtime"],
            ["<code>epochs</code>", "100",
             "เพิ่มจาก 50 ในรอบก่อนหน้า เพราะกราฟยังไม่อิ่มตัวที่ 50"],
            ["<code>patience</code>", "30",
             "หยุดเทรนถ้าไม่ดีขึ้นติดต่อกัน 30 epoch กัน overfit และประหยัดเวลา GPU"],
            ["<code>batch</code>", "16", "จำกัดด้วย VRAM ของ Tesla T4 บน Colab"],
            ["<code>imgsz</code>", "640",
             "<b>ค่าที่สำคัญที่สุดต่อการใช้งานจริง</b> ลดจาก 960 ในรอบที่ 4 เพราะ 960 ทำให้ "
             "inference ช้าเกินไปบน GPU ที่ใช้จริง และ<b>ต้องใช้ค่าเดียวกันตอน inference</b> "
             "มิฉะนั้นตำแหน่งกรอบจะคลาดเคลื่อน"],
            ["<code>optimizer</code>", "SGD",
             "เสถียรกว่า Adam สำหรับ fine-tune งาน detection และเป็นค่าที่ Ultralytics แนะนำ"],
            ["<code>lr0</code> / <code>lrf</code>", "0.001 / 0.01",
             "learning rate เริ่มต้น และตัวคูณของ lr สุดท้าย (lr สุดท้าย = lr0 × lrf) "
             "lr0 ต่ำเพราะเป็นการ fine-tune จากน้ำหนักที่เทรนมาแล้ว ไม่ใช่เทรนจากศูนย์"],
            ["<code>cos_lr</code>", "true",
             "ลด lr แบบ cosine แทนแบบเชิงเส้น ทำให้ช่วงท้ายการเทรนนิ่งขึ้น "
             "<b>เปิดเพิ่มในรอบที่ 6</b> และเป็นหนึ่งในสาเหตุที่กราฟเสถียรขึ้นชัดเจน"],
            ["<code>warmup_epochs</code>", "3.0",
             "ค่อยๆ เพิ่ม lr ใน 3 epoch แรก กันไม่ให้ gradient ก้อนใหญ่ตอนเริ่มทำลายน้ำหนักเดิม"],
            ["<code>momentum</code>", "0.937", "ค่ามาตรฐานของ Ultralytics ไม่ได้ปรับ"],
            ["<code>weight_decay</code>", "0.0005", "regularisation มาตรฐาน กัน overfit"],
        ],
        widths=["18%", "14%", "68%"],
    ))

    s.append(page_break())
    s.append(h3("Augmentation — ปรับให้ตรงกับมุมมองโดรน"))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "เหตุผล"],
        [
            ["<code>degrees</code>", "180.0",
             "<b>เปลี่ยนจาก 0 ในรอบที่ 4</b> — ภาพจากโดรนมองลงมา วัตถุหันไปทางไหนก็ได้ "
             "การหมุนเต็มช่วงจึงสมจริง ต่างจากภาพระดับพื้นที่รถมักตั้งตรงเสมอ"],
            ["<code>flipud</code>", "0.5",
             "<b>เปิดเพิ่มจาก 0</b> ด้วยเหตุผลเดียวกัน — มุมมองจากด้านบนไม่มี 'บน' ที่แน่นอน"],
            ["<code>fliplr</code>", "0.5", "พลิกซ้ายขวา เป็นค่ามาตรฐานที่สมเหตุสมผลอยู่แล้ว"],
            ["<code>scale</code>", "0.5",
             "สุ่มย่อ/ขยาย ±50% ช่วยให้ทนต่อความสูงบินที่ต่างกัน ซึ่งเปลี่ยนขนาดวัตถุในภาพโดยตรง"],
            ["<code>translate</code>", "0.1", "เลื่อนภาพ ±10% ช่วยเรื่องวัตถุที่อยู่ติดขอบเฟรม"],
            ["<code>hsv_h/s/v</code>", "0.015 / 0.7 / 0.4",
             "สุ่มเปลี่ยนสี ความอิ่มสี และความสว่าง ครอบคลุมสภาพแสงและเวลาที่ต่างกัน "
             "ค่า h ต่ำเพราะการเปลี่ยนเฉดสีมากทำให้ภาพไม่สมจริง"],
            ["<code>mosaic</code>", "1.0",
             "ต่อสี่ภาพเป็นภาพเดียว เพิ่มความหลากหลายของบริบทและช่วยเรื่องวัตถุเล็กมาก"],
            ["<code>close_mosaic</code>", "10",
             "<b>เปิดเพิ่มในรอบที่ 6</b> — ปิด mosaic ใน 10 epoch สุดท้าย เพื่อให้โมเดลเห็น"
             "ภาพจริงที่ไม่ถูกต่อก่อนจบ ซึ่งตรงกับสิ่งที่จะเจอตอน inference"],
        ],
        widths=["18%", "14%", "68%"],
    ))
    s.append(note("ทำไมต้องแก้ letterbox ด้วย", [
        "รอบก่อนหน้ามีปัญหาภาพบิดเบี้ยวเพราะการ resize ไม่รักษาอัตราส่วน การใช้ letterbox "
        "(เติมขอบแทนการยืด) ทำให้สัดส่วนของวัตถุถูกต้อง ซึ่งสำคัญมากกับการทำนายขนาดกรอบ",
    ]))

    s.append(h3("ผลการเทรนทั้งสามรอบ"))
    s.append(table(
        ["รอบ", "epochs", "imgsz", "mAP50", "mAP50-95", "precision", "recall"],
        [
            ["visdrone_custom_4th", "50", "960", "0.9101", "0.6773", "0.8511", "0.8854"],
            ["visdrone_custom_6th", "100", "640", "0.9041", "0.6201", "0.8442", "0.9095"],
            ["<b>visdrone_custom_7th</b>", "100", "640",
             "<b>0.9174</b>", "0.6328", "0.8688", "0.8873"],
        ],
        widths=["28%", "12%", "12%", "12%", "14%", "11%", "11%"],
    ))
    s.append(p(
        "<b>mAP50</b> คือความแม่นเฉลี่ยเมื่อถือว่ากรอบถูกต้องเมื่อซ้อนทับของจริงเกิน 50% "
        "ส่วน <b>mAP50-95</b> เฉลี่ยข้ามเกณฑ์ 50% ถึง 95% จึงเข้มงวดกว่ามากและสะท้อน"
        "ความแม่นของขอบกรอบ ไม่ใช่แค่การเจอวัตถุ"
    ))
    s.append(note("เหตุผลที่ต้องอ่านสองตัวเลขคู่กัน", [
        "รอบที่ 4 มี mAP50-95 สูงสุด (0.6773) เพราะเทรนที่ 960 px ซึ่งให้ขอบกรอบแม่นกว่า "
        "แต่ imgsz นั้นช้าเกินไปสำหรับการใช้งานจริงบน GTX 1650",
        "รอบที่ 7 จึงเป็นตัวที่เลือกใช้: mAP50 ดีที่สุด (0.9174) ที่ imgsz ใช้งานได้จริง "
        "— เป็นการแลกความแม่นของขอบกรอบกับความเร็ว ซึ่งเป็นการแลกที่ถูกต้องสำหรับการประมวลผลวิดีโอบน GTX 1650",
    ]))
    s.append(figure(run_dir / "results.png", f"กราฟการเทรนรอบ {FIGURE_RUN}"))
    s.append(page_break())
    s.append(figure(run_dir / "PR_curve.png", "Precision-Recall curve — พื้นที่ใต้กราฟคือ mAP50"))
    s.append(figure(run_dir / "confusion_matrix_normalized.png",
                    "Confusion matrix แบบ normalized — แนวทแยงคือการทำนายถูก"))
    return "".join(s)


def _inference() -> str:
    s = [page_break(), h1("5. Inference และการติดตามวัตถุ")]
    s.append(h2("5.1 พารามิเตอร์ตอน inference"))
    s.append(table(
        ["พารามิเตอร์", "ค่า", "คืออะไร และทำไมเป็นค่านี้"],
        [
            ["<code>imgsz</code>", "640",
             "ค่าเริ่มต้นของการประมวลผลวิดีโอ เลือก 416/960/1280 ได้ในหน้าเว็บ ควรตรงกับขนาดตอนเทรน "
             "เพราะการรันคนละขนาดทำให้ตำแหน่งกรอบคลาดเคลื่อนโดยตรง "
             "รอบที่ 4 เทรนที่ 960 ส่วน best50 และรอบที่ 6–7 เทรนที่ 640"],
            ["<code>conf</code>", "0.45",
             "เกณฑ์ความมั่นใจขั้นต่ำ ต่ำกว่านี้จะได้กรอบผิดเยอะ สูงกว่านี้จะพลาดวัตถุจริง"],
            ["<code>iou</code>", "0.45",
             "เกณฑ์ของ Non-Maximum Suppression — กรอบที่ซ้อนทับกันเกินค่านี้จะถูกยุบเหลือใบเดียว "
             "ต่ำเกินไปจะลบกรอบของวัตถุที่อยู่ชิดกันจริง สูงเกินไปจะเหลือกรอบซ้ำ"],
            ["<code>max_det</code>", "300", "จำนวนกรอบสูงสุดต่อเฟรม กันกรณีผิดปกติที่กรอบระเบิด"],
            ["<code>half</code>", "<b>ไม่ส่ง</b>",
             "GTX 1650 ไม่มี tensor core จึง emulate half precision วัดที่ 640 px แล้ว FP32 "
             "เฉลี่ย 30.7 ms ส่วน FP16 เฉลี่ย 164.1 ms — <b>ช้ากว่า 5 เท่า</b> "
             "และ Ultralytics 8.4 เลิกใช้ argument นี้แล้วจึงเตือนทุกครั้งที่ส่งไม่ว่าค่าจะเป็นอะไร"],
            ["<code>TELEMETRY_OCR_YOLO_THREADS</code>", "2",
             "จำกัด thread ของ PyTorch/OpenCV ไม่ให้ YOLO กินทุกคอร์จนแย่ง CPU ของ OCR "
             "ยังจำเป็นแม้รันบน CUDA เพราะ letterbox และ NMS ทำบน CPU"],
        ],
        widths=["24%", "14%", "62%"],
    ))

    s.append(h2("5.2 BoT-SORT — การติดตามข้ามเฟรม"))
    s.append(p(
        "detection บอกว่าเฟรมนี้มีอะไรบ้าง แต่ไม่บอกว่ารถคันนี้คือคันเดียวกับเฟรมก่อนหรือไม่ "
        "tracker ทำหน้าที่นั้น โดยใช้ Kalman filter ทำนายตำแหน่งถัดไปแล้วจับคู่กับ detection ใหม่"
    ))
    s.append(p(
        "ผลคือแต่ละวัตถุได้ <b>track ID</b> ที่คงเดิมข้ามเฟรม ซึ่งมีประโยชน์สองอย่าง: "
        "นับจำนวนวัตถุที่ไม่ซ้ำได้ และถ้าเลขกระโดดไปมาแปลว่า tracker หลุด ซึ่งเป็นสัญญาณว่า "
        "detection ไม่นิ่งพอหรือเฟรมห่างเกินไป"
    ))
    s.append(note("สิ่งที่ tracker ทำไม่ได้", [
        "BoT-SORT ช่วยให้ ID และตำแหน่งต่อเนื่องขึ้น แต่<b>แก้ false positive หรือคลาสที่โมเดล"
        "ทำนายผิดไม่ได้</b> ถ้าโมเดลเห็นเงาเป็นรถ tracker จะติดตามเงานั้นอย่างซื่อสัตย์",
    ], kind="warn"))
    s.append(h3("tracker เริ่มใหม่เมื่อไร"))
    s.append(p(
        "การประมวลผลวิดีโอหนึ่งครั้งใช้ tracker หนึ่งชุด เดินจากเฟรมแรกถึงเฟรมสุดท้ายโดยไม่ reset "
        "เพราะเฟรมเรียงต่อกันเสมอ การประมวลผลใหม่ (วิดีโอ โมเดล หรือค่าใหม่) คือโปรเซสใหม่และ tracker ใหม่ "
        "ส่วน worker แบบสดของ CLI <code>run --detect-model</code> ยัง reset เมื่อ source เปลี่ยน "
        "หรือเวลากระโดดเกิน <b>1000 ms</b>"
    ))
    return "".join(s)


def _realtime() -> str:
    s = [page_break(), h1("6. ประมวลผลวิดีโอทั้งไฟล์")]
    s.append(p(
        "หน้าเว็บไม่วาดกรอบสดทับวิดีโออีกแล้ว server รันวิดีโอผ่าน YOLO + BoT-SORT "
        "<b>ทุกเฟรมตามลำดับครั้งเดียว</b> วาดกรอบลงในภาพ แล้วเข้ารหัสเป็น MP4 (H.264) "
        "ที่หน้าเว็บเล่นได้ พร้อมเก็บผลรายเฟรมไว้ใน <code>tracks.jsonl</code>"
    ))
    s.append(note("ทำไมเลิกวาดกรอบสด", [
        "การวาดสดส่งเฟรมให้ tracker ได้ราว 10–16 จาก 30 เฟรมต่อวินาที และห่างไม่เท่ากันตามจังหวะ"
        "ที่ GPU ว่าง BoT-SORT จึงจับคู่วัตถุหลุดทุกครั้งที่กล้องเคลื่อน",
        "กรอบที่วาดยังอธิบายภาพในอดีตเสมอ (ช้ากว่าภาพ 1–3 เฟรม) ต้องมีทั้ง auto pacing "
        "การคาดตำแหน่ง long-poll และการซ่อนผลที่เก่าเกิน 750 ms เพื่อให้พอดูได้ ทั้งหมดคือการชดเชย"
        "ปัญหาที่ไม่ควรมีตั้งแต่ต้น",
        "การประมวลผลทั้งไฟล์ตัดทั้งสองปัญหาทิ้ง: tracker เห็นทุกเฟรม และสิ่งที่อยู่บนจอคือสิ่งที่"
        "โมเดลเห็นในเฟรมนั้นพอดี แลกกับการต้องรอก่อนดู",
    ], kind="warn"))

    s.append(h2("6.1 ขั้นตอนต่อเฟรม"))
    s.append(p(
        "วัดจริงบน GTX 1650 + Core i5-11320H คลิป 1080p30 ยาว 60 วินาที (1,801 เฟรม) "
        "ด้วย best50 ที่ 640 px:"
    ))
    s.append(table(
        ["ขั้นตอน", "ทำอะไร", "ms ต่อเฟรม"],
        [
            ["decode", "<code>cv2.VideoCapture</code> grab/retrieve ถอดเฟรม 1080p", "5.2"],
            ["track", "YOLO inference + BoT-SORT (รวม global motion compensation)", "<b>53.4</b>"],
            ["draw", "<code>annotate_frame</code> วาดกรอบ ป้าย และ Track ID ลงสำเนา", "1.2"],
            ["encode", "ส่งภาพ BGR เข้า ffmpeg ทาง stdin (รวมเวลารอเมื่อ encoder ตามไม่ทัน)", "2.3"],
        ],
        widths=["14%", "66%", "20%"],
    ))
    s.append(p(
        "รวมได้ <b>16.0 เฟรม/วินาที</b> (112.5 วินาทีสำหรับคลิป 60 วินาที) วิดีโอ 25 นาทีจึงใช้ราว "
        "47 นาที เวลาเกือบทั้งหมดอยู่ที่ขั้น track ตัวเลือก 'ทุก 2 เฟรม' ลดงานลงราวครึ่งหนึ่ง "
        "โดย tracker ยังเห็นเฟรมที่ห่างเท่ากันเสมอ"
    ))

    s.append(h2("6.2 การเลือก encoder"))
    s.append(table(
        ["encoder", "ถอด+เข้ารหัส 600 เฟรม", "ในการประมวลผลจริง", "หมายเหตุ"],
        [
            ["<code>h264_nvenc</code>", "~275 เฟรม/วินาที", "<b>16.0 เฟรม/วินาที</b>",
             "วงจรเข้ารหัสเฉพาะบน GPU ไม่แย่ง CPU"],
            ["<code>h264_qsv</code>", "~135 เฟรม/วินาที", "—", "Intel QuickSync บน iGPU"],
            ["<code>libx264</code> veryfast", "~105 เฟรม/วินาที", "5.0 เฟรม/วินาที",
             "แย่ง CPU 4 คอร์กับ tracker"],
        ],
        widths=["20%", "24%", "24%", "32%"],
    ))
    s.append(note("บทเรียน: เร็วเมื่อวัดแยก ไม่ได้แปลว่าเร็วในสายงานจริง", [
        "x264 วัดแยกได้ ~105 เฟรม/วินาทีซึ่งดูเหลือเฟือ แต่ในการประมวลผลจริงมันแย่ง CPU กับ YOLO "
        "(letterbox, NMS) และ optical flow ของ BoT-SORT จนทั้งสายเหลือ 5 เฟรม/วินาที "
        "ระหว่างนั้น GPU ใช้งานแค่ราว 21% ซึ่งเป็นสัญญาณว่าคอขวดไม่ได้อยู่ที่โมเดล",
        "ระบบจึงลอง <code>h264_nvenc</code> → <code>h264_qsv</code> → <code>libx264</code> "
        "ด้วยการ encode จริงสองเฟรมก่อนเริ่มงาน และรายงานเวลาต่อขั้นตอนไว้ในความคืบหน้าเสมอ",
    ], kind="warn"))

    s.append(h2("6.3 แคช ความสดของผล และ capture"))
    s.append(ul([
        "ผลเก็บที่ <code>output/renders/&lt;วิดีโอ&gt;__&lt;โมเดล&gt;__c0.45-s640-st1/</code> "
        "เลือกวิดีโอ โมเดล และค่าเดิมซ้ำจะเล่นผลเดิมทันที",
        "<code>meta.json</code> เก็บขนาดและเวลาแก้ไขของวิดีโอกับ <code>.pt</code> "
        "ถ้าไฟล์ใดเปลี่ยน ผลนั้นถือว่าเก่า",
        "ประมวลผลได้ทีละงานเพราะ GPU 4 GB ปิด server ระหว่างทางจะยกเลิกงานและปิดโปรเซสลูก",
        "<b>Capture ไม่ใช้ภาพจากวิดีโอ AI</b> เพราะกรอบทับตัวเลข OSD — server อ่านเฟรมต้นฉบับ"
        "จากไฟล์ ณ เวลาเดียวกัน แล้วเอากรอบของเฟรมนั้นจาก <code>tracks.jsonl</code> "
        "ภาพ capture จึงตรงกับวิดีโอ AI โดยไม่ต้องให้โมเดลอ่านซ้ำ",
    ]))
    return "".join(s)


def _security_and_limits() -> str:
    s = [page_break(), h1("7. ความปลอดภัยและข้อจำกัด")]
    s.append(h2("7.1 การป้องกัน API"))
    s.append(p(
        "server ตรวจสามชั้นก่อนเข้า handler เพราะทุก endpoint มี side effect ที่เกิดขึ้น"
        "<b>ก่อน</b>ที่ CORS จะตัดสินว่าใครอ่าน response ได้"
    ))
    s.append(ul([
        "<code>Host</code> ต้องเป็น loopback หรือค่าที่ส่งผ่าน <code>--host</code> "
        "มิฉะนั้นตอบ 403 — กัน DNS rebinding",
        "request ที่มี <code>Origin</code> แต่ไม่ผ่าน allowlist ถูก<b>ปฏิเสธ</b>ที่ 403 "
        "ไม่ใช่แค่ไม่ใส่ CORS header",
        "ทุก POST/DELETE ใต้ <code>/api/</code> ต้องมี header <code>X-Telemetry-OCR: 1</code>",
    ]))
    s.append(note("ข้อสุดท้ายสำคัญที่สุด", [
        "form และ <code>text/plain</code> เป็น CORS-safelisted จึงไม่เกิด preflight "
        "ถ้าไม่บังคับ header นี้ เว็บใดก็ตามที่ผู้ใช้เปิดอยู่จะสั่ง <code>/api/renders</code> "
        "ให้ GPU ทำงานยาว หรือแก้ <code>/api/models</code> ว่าเครื่องนี้จะโหลด checkpoint "
        "(pickle) ตัวไหน header ที่ไม่อยู่ใน safelist บังคับให้ browser ส่ง preflight ที่ guard ปฏิเสธได้",
        "กลไกนี้ทำงานจริง: log ของ session หนึ่งบันทึกว่ามีเว็บภายนอกพยายามเรียก API "
        "ในเครื่อง 52 ครั้ง และถูกปฏิเสธด้วย 403 ทุกครั้ง",
    ], kind="warn"))
    s.append(p(
        "หน้าเว็บ<b>ไม่รับอัปโหลด</b> <code>.pt</code> เพราะ checkpoint ของ PyTorch บรรจุโค้ด"
        "ที่ทำงานตอนโหลดได้ การเพิ่มโมเดลเลือกได้เฉพาะไฟล์ที่ server เจอเองใน <code>models/</code> "
        "และ <code>runs/*/weights/</code> ส่วนการเอาออกไม่ลบไฟล์ "
        "ควรใส่ในโฟลเดอร์เหล่านั้นเฉพาะไฟล์ที่เชื่อถือได้"
    ))
    s.append(h2("7.2 ความเป็นส่วนตัวของข้อมูล"))
    s.append(p(
        "วิดีโออยู่ในโฟลเดอร์บนเครื่อง server อ่านเองและส่งให้หน้าเว็บผ่าน <code>127.0.0.1</code> "
        "เท่านั้น การประมวลผลทั้งหมดเกิดในเครื่อง ภาพออกนอกเครื่องเฉพาะเมื่อเปิดการส่งรูปเข้า LINE"
    ))

    s.append(h2("7.3 ข้อจำกัดที่ควรรู้"))
    s.append(ul([
        "<b>ROI ต้องคงที่</b> — วิดีโอที่หมุน ครอป หรือเปลี่ยน layout ของ OSD ต้อง calibrate ใหม่ทั้งหมด",
        "<b>ถ้าไม่มีสัญญาณเหลืออยู่ preprocessing กู้ไม่ได้</b> — เมื่อตัวอักษรกับพื้นหลัง"
        "เหมือนกันจนไม่มี spatial/color signal เหลือ ต้องใช้ต้นฉบับคุณภาพสูงขึ้น "
        "multi-frame reconstruction หรือ fine-tuning",
        "<b>Exact voting ไม่เหมาะกับค่าที่เปลี่ยนทุก tick</b> — field เหล่านั้นใช้ median "
        "และต้องมีอย่างน้อย 3 observation",
        "<b>tracker แก้ false positive ไม่ได้</b> — ถ้าโมเดลทำนายผิด tracker จะติดตามสิ่งที่ผิดต่อไป",
        "<b>การประมวลผลวิดีโอใช้เวลา</b> — ราว 2 เท่าของความยาววิดีโอบน GTX 1650 และทำได้ทีละงาน "
        "เลือก 'ทุก 2 เฟรม' เพื่อลดเวลาลงราวครึ่งหนึ่ง",
    ]))

    s.append(h2("7.4 เป้าหมายก่อนใช้งานจริง"))
    s.append(p("ก่อนนำไปใช้จริงควรมี labeled set อย่างน้อย 1,000 ROI crops แล้วตรวจว่า:"))
    s.append(table(
        ["ตัวชี้วัด", "เป้าหมาย"],
        [
            ["exact-field accuracy", "อย่างน้อย 97%"],
            ["sustained update rate", "อย่างน้อย 10 Hz (เป้าหมาย 12 Hz)"],
            ["p95 frame-to-result latency", "ไม่เกิน 100 ms"],
            ["<code>dropped_frames</code>", "ต้องไม่เพิ่มต่อเนื่องใน workload ปกติ"],
        ],
        widths=["45%", "55%"],
    ))
    s.append(note("ถ้ายังไม่ถึงเป้าหมาย", [
        "ให้ fine-tune <code>en_PP-OCRv5_mobile_rec</code> ด้วยภาพจริงและภาพสังเคราะห์"
        "อย่างน้อย 5,000 crops โดยคง character dictionary เดิมไว้ก่อน และเพิ่ม augmentation "
        "สำหรับ blur, video compression, low contrast และสีที่กลืนพื้นหลัง",
    ]))
    return "".join(s)


def _measurements() -> str:
    s = [page_break(), h1("8. สรุปตัวเลขที่วัดได้จริงทั้งหมด")]
    s.append(p(
        "ทุกค่าในหน้านี้มาจากการวัดบนเครื่องที่ใช้งานจริง (GTX 1650 4 GB, compute 7.5) "
        "ไม่ใช่ค่าประมาณหรือค่าจากเอกสารภายนอก"
    ))
    s.append(h2("8.1 CPU เทียบกับ CUDA"))
    s.append(table(
        ["งาน", "CPU", "CUDA"],
        [
            ["OCR update rate", "9.84 Hz", "<b>10.10 Hz</b>"],
            ["OCR latency p95", "164.0 ms", "<b>46.6 ms</b>"],
            ["OCR dropped frames", "130", "<b>1</b>"],
            ["YOLO track @416 px", "236.8 ms", "<b>32.8 ms</b>"],
            ["YOLO track @640 px", "507.3 ms", "<b>41.4 ms</b>"],
        ],
        widths=["40%", "30%", "30%"],
    ))
    s.append(p(
        "ข้อสังเกตสำคัญ: บน CUDA ที่ <b>640 px (41.4 ms) เร็วกว่า 416 px บน CPU "
        "(236.8 ms) หลายเท่า</b> จึงไม่มีเหตุผลต้องลดขนาดภาพลงเมื่อมี GPU"
    ))

    s.append(h2("8.2 latency ของ telemetry ที่ถอดออกมา"))
    s.append(p("วัดจากคลิปจริง 60 วินาที (606 เฟรม, 7,272 records, dropped 0):"))
    s.append(table(
        ["ช่วง", "median", "p95"],
        [
            ["OCR pipeline อย่างเดียว (GPU ว่าง)", "<b>30.1 ms</b>", "43.7 ms"],
            ["ผ่าน HTTP — pipeline ที่ server รายงาน", "38.7 ms", "53.6 ms"],
            ["ผ่าน HTTP — round trip ทั้งหมด", "<b>54.4 ms</b>", "69.2 ms"],
        ],
        widths=["50%", "25%", "25%"],
    ))
    s.append(p("บวกเวลาที่ต้องรอจนค่าถูกยืนยัน:"))
    s.append(table(
        ["ประเภท field", "รวมทั้งหมด (เฟรมบนจอ → ตัวเลขที่ยืนยันแล้ว)"],
        [
            ["field ทั่วไป (<code>fast</code>)", "<b>~155 ms</b> = round trip 54 + ยืนยัน 99"],
            ["<code>timestamp</code> (<code>stable</code>)",
             "<b>~254 ms</b> = round trip 54 + ยืนยัน 198"],
        ],
        widths=["35%", "65%"],
    ))

    s.append(h2("8.3 ผลของการแก้ request flood"))
    s.append(table(
        ["ตัวชี้วัด", "ก่อนแก้", "หลังแก้"],
        [
            ["อัตราส่วน poll ต่อ frame", "91 : 1", "<b>0.95 : 1</b>"],
            ["requests สูงสุดต่อวินาที", "717", "<b>16</b>"],
            ["<code>harvest</code> — ผลนอนรอใน server", "64.8 ms", "<b>0.2 ms</b>"],
            ["round trip ของ detection", "124.9 ms", "<b>60.1 ms</b>"],
            ["คำเตือน Ultralytics ต่อ session", "17,306", "<b>0</b>"],
        ],
        widths=["40%", "30%", "30%"],
    ))
    s.append(note("สรุปหลักการที่ได้จากงานนี้", [
        "<b>วัดก่อนแก้เสมอ</b> — ปัญหา latency ที่ไล่กันหลายรอบ กลายเป็นว่าส่วนใหญ่มาจาก "
        "request flood ที่มองไม่เห็นจนกระทั่งนับ log จริง",
        "<b>ตัวเลขที่แสดงต้องแยกเป็นช่วงได้</b> — ค่ารวมค่าเดียวบอกไม่ได้ว่าต้องแก้ตรงไหน "
        "การแยกเป็น cap/post/infer/transport/wait คือสิ่งที่ทำให้หาต้นเหตุเจอ",
        "<b>ยอมไม่แสดงดีกว่าแสดงผิด</b> — ทั้งเกณฑ์ 750 ms ที่ซ่อนกรอบ และการปฏิเสธที่จะเดา"
        "จุดทศนิยม ตั้งอยู่บนหลักการเดียวกัน",
    ], kind="ok"))
    return "".join(s)


def build_html(run_dir: Path) -> str:
    body = "".join([
        _cover(),
        _overview(),
        _ocr_pipeline(),
        _temporal(),
        _detection(run_dir),
        _inference(),
        _realtime(),
        _security_and_limits(),
        _measurements(),
    ])
    return (
        '<!doctype html><html lang="th"><head><meta charset="utf-8">'
        "<title>Telemetry Vision — คู่มือหลักการและทฤษฎี</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def build_handbook(
    paths: ProjectPaths,
    *,
    output: Path | None = None,
    run_name: str | None = None,
) -> Path:
    """สร้างคู่มือฉบับเต็มเป็น PDF"""
    from .report import html_to_pdf

    runs_root = paths.root / "runs"
    run_dir = runs_root / (run_name or FIGURE_RUN)
    if not run_dir.is_dir():
        available = ", ".join(sorted(d.name for d in runs_root.glob("*") if d.is_dir()))
        raise ConfigError(
            f"ไม่พบโฟลเดอร์ผลการเทรน: {run_dir}\nที่มีอยู่: {available or 'ไม่มี'}"
        )
    destination = output or paths.output_dir("pdf") / DEFAULT_PDF_NAME
    html_to_pdf(build_html(run_dir), destination)
    return destination
