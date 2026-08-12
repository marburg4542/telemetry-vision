import sys

# --- เพิ่มส่วนเช็คการติดตั้ง PyTorch ---
try:
    import torch
    print(f"✅ ตรวจพบ PyTorch เวอร์ชั่น: {torch.__version__}")
except ImportError:
    print("❌ ERROR: ยังไม่ได้ติดตั้ง PyTorch!")
    print("💡 คำแนะนำ: กรุณาติดตั้งโดยเปิด Terminal/CMD แล้วพิมพ์คำสั่ง:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit()  # หยุดการทำงานของสคริปต์ทันทีหากไม่มี PyTorch
# ------------------------------------

# 1. เช็คว่า CUDA เปิดใช้งานได้หรือไม่
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # 2. ดูชื่อการ์ดจอที่ใช้ได้
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device Name: {gpu_name}")
    
    # 3. ดูจำนวน GPU ที่มีในระบบ
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # 4. ดูหน่วยความจำ (VRAM) ที่ใช้ได้
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU Memory: {total_memory:.2f} GB")
else:
    print("WARNING: ไม่พบ GPU/CUDA ระบบจะใช้ CPU ซึ่งจะเทรนช้ามาก!")