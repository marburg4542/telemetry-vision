"""สคริปต์เทรนโมเดล YOLOv8 บน Google Colab

วิธีใช้: copy โค้ดแต่ละ "# %% เซลล์ N" ไปวางเป็น cell แยกใน Colab notebook
ตามลำดับ 1-9 แล้วรันทีละเซลล์ (ไม่ต้อง clone repo เพราะไฟล์นี้ทำงานได้ครบในตัวเอง)

ก่อนรัน ให้ตั้งค่า Colab Secret ชื่อ ROBOFLOW_API_KEY ก่อน:
คลิกไอคอนกุญแจ (Secrets) แถบซ้ายของ Colab > Add new secret
Name: ROBOFLOW_API_KEY, Value: <API key ของคุณจาก Roboflow>
แล้วเปิด toggle "Notebook access" ให้ notebook นี้เข้าถึงได้
"""

# %% เซลล์ 1 — ติดตั้ง dependencies
!pip install -q ultralytics ultralyticsplus roboflow
print("✅ ติดตั้ง Library เรียบร้อย")


# %% เซลล์ 2 — Mount Google Drive (กันผลลัพธ์หายเมื่อ session หลุด)
from google.colab import drive

drive.mount("/content/drive")

DRIVE_SAVE_PATH = "/content/drive/MyDrive/YOLO_Training_Results"
print(f"✅ จะบันทึกผลลัพธ์การเทรนไปที่: {DRIVE_SAVE_PATH}")


# %% เซลล์ 3 — import + ปลดล็อกระบบความปลอดภัยของ PyTorch 2.6+
import os
import sys
from datetime import datetime

import torch
from roboflow import Roboflow
from ultralyticsplus import YOLO

# หมายเหตุ: patch นี้แค่เติม default kwarg ให้ torch.load เฉยๆ
# จึงไม่มีผลกระทบและไม่ error แม้ Colab จะติดตั้ง PyTorch เวอร์ชันที่เก่ากว่า 2.6
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load
print("✅ Patch torch.load เรียบร้อย")


# %% เซลล์ 4 — อ่าน Roboflow API key จาก Colab Secrets
from google.colab import userdata

try:
    roboflow_api_key = userdata.get("ROBOFLOW_API_KEY")
except Exception:
    roboflow_api_key = None

if not roboflow_api_key:
    print("❌ ERROR: ไม่พบ Colab Secret ชื่อ ROBOFLOW_API_KEY")
    print("💡 คลิกไอคอนกุญแจ (Secrets) แถบซ้ายของ Colab > Add new secret")
    print("   Name: ROBOFLOW_API_KEY, Value: <API key ของคุณ> แล้วเปิด Notebook access")
    sys.exit(1)

print("✅ อ่าน ROBOFLOW_API_KEY จาก Colab Secrets เรียบร้อย")


# %% เซลล์ 5 — ตรวจสอบสถานะ GPU
if torch.cuda.is_available():
    device = 0
    print(f"✅ ตรวจพบ GPU ของคุณ: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM ทั้งหมด: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    device = "cpu"
    print("⚠️ ไม่พบ GPU! ไปที่ Runtime > Change runtime type > เลือก GPU ก่อนรัน")


# %% เซลล์ 6 — ดาวน์โหลด Dataset จาก Roboflow
# แก้เลข version ตรงนี้ให้ตรงกับ dataset version ใหม่ที่สร้างตามคำแนะนำ
# (Resize: Fit within 640x640, ปิด Roboflow augmentation, เพิ่มภาพ motorcycle)
DATASET_VERSION = 6

print(f"⏳ กำลังดาวน์โหลด Dataset จาก Roboflow (version {DATASET_VERSION})...")
rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("objects-detection-1orcw").project("car-detection-yntbe")
version = project.version(DATASET_VERSION)
dataset = version.download("yolov8")

dataset_yaml = f"{dataset.location}/data.yaml"
print(f"✅ ดาวน์โหลดเสร็จแล้ว: {dataset_yaml}")


# %% เซลล์ 7 — โหลด Pre-trained Weights
print("⏳ กำลังโหลดโมเดลตั้งต้น...")
model = YOLO("mshamrai/yolov8m-visdrone")


# %% เซลล์ 8 — เริ่มเทรนโมเดล
run_name = f"visdrone_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"🚀 เริ่มต้นการเทรนโมเดล... (run: {run_name})")

results = model.train(
    data=dataset_yaml,
    epochs=100,
    imgsz=640,          # ตรงกับ native resolution ของภาพใน dataset (Fit within 640x640)
    batch=16,            # พิสูจน์แล้วว่าใช้ได้ผลจริงบน Colab GPU
    workers=4,
    device=device,
    amp=True,            # เปิด Mixed Precision ช่วยให้เทรนไวและประหยัด VRAM
    lr0=0.001,
    close_mosaic=10,     # ปิด mosaic ช่วง epoch ท้ายเพื่อ fine-tune localization
    cos_lr=True,         # cosine LR decay ให้ LR ลดนุ่มนวลกว่า linear ช่วงท้าย
    patience=30,
    degrees=180.0,       # ภาพมุมสูงจากโดรน ไม่มีทิศ 'บน' ที่ตายตัว หมุนได้เต็มช่วง
    flipud=0.5,          # flip แนวตั้งยังสมจริงสำหรับภาพมุมสูง

    project=DRIVE_SAVE_PATH,
    name=run_name,
    exist_ok=True,
    pretrained=True,
    verbose=True,
)


# %% เซลล์ 9 — สรุปผล
best_weights_path = f"{DRIVE_SAVE_PATH}/{run_name}/weights/best.pt"
print(f"🎉 เทรนเสร็จสิ้น! ไฟล์ weight อยู่ที่: {best_weights_path}")
