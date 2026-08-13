import cv2
from pathlib import Path

# รองรับไฟล์วิดีโอนามสกุลหลัก ๆ
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def extract_frames_from_folder(
    data_dir="data", output_base_dir="dataset_frames", interval_seconds=1.0
):
    """ค้นหาไฟล์วิดีโอทั้งหมดในโฟลเดอร์ที่กำหนด แล้วสกัดเป็นเฟรมภาพแยกตามชื่อวิดีโอ

    :param data_dir: โฟลเดอร์ต้นทางที่เก็บไฟล์วิดีโอ (เช่น './data')
    :param output_base_dir: โฟลเดอร์ปลายทางหลักที่จะเก็บ Dataset
    :param interval_seconds: ระยะเวลา (วินาที) ในการสกัด 1 ภาพ (0.1 = 10
        ภาพ/วินาที, 0.5 = 2 ภาพ/วินาที, 1.0 = 1 ภาพ/วินาที)
    """
    data_path = Path(data_dir)
    output_path = Path(output_base_dir)

    # 1. เช็กว่ามีโฟลเดอร์ data จริงหรือไม่
    if not data_path.exists():
        print(f"[Error] ไม่พบโฟลเดอร์ต้นทาง: {data_dir}")
        return

    # 2. ค้นหาวิดีโอทั้งหมดในโฟลเดอร์ (รวมถึงโฟลเดอร์ย่อยด้วย .rglob)
    video_files = [
        f
        for f in data_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(
            f"[Warning] ไม่พบไฟล์วิดีโอ (MP4/MOV/AVI/MKV) ในโฟลเดอร์: {data_dir}"
        )
        return

    print(f"--> พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ ใน '{data_dir}'\n")

    # 3. วนลูปประมวลผลทีละวิดีโอ
    for index, video_file in enumerate(video_files, start=1):
        # สร้างชื่อโฟลเดอร์ปลายทางตามชื่อไฟล์วิดีโอ (ตัดนามสกุลออก)
        # เช่น data/cat_video.mp4 -> dataset_frames/cat_video/
        save_dir = output_path / video_file.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{index}/{len(video_files)}] กำลังประมวลผล: {video_file.name} ..."
        )

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"   - [Skip] ไม่สามารถเปิดไฟล์ได้: {video_file.name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        # ป้องกันกรณีไฟล์วิดีโอบางไฟล์อ่านค่า FPS ไม่ถูก ให้ค่าเริ่มต้นเป็น 30.0
        fps = fps if fps > 0 else 30.0

        frame_step = max(1, int(fps * interval_seconds))
        current_frame = 0
        saved_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame % frame_step == 0:
                # ตั้งชื่อไฟล์ภาพ: ชื่อวิดีโอ_000001.jpg
                img_name = f"{video_file.stem}_{saved_count:06d}.jpg"
                img_path = save_dir / img_name

                # บันทึกเป็น JPEG คุณภาพ 95%
                cv2.imwrite(
                    str(img_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                saved_count += 1

            current_frame += 1

        cap.release()
        print(
            f"   -> บันทึกแล้ว {saved_count} ภาพ ลงในโฟลเดอร์: {save_dir}\n"
        )

    print("--> ดึงภาพจากทุกวิดีโอเสร็จสิ้นเรียบร้อยแล้ว!")


# --- ตัวอย่างการเรียกใช้งาน ---
if __name__ == "__main__":
    # ระบุโฟลเดอร์ในโปรเจ็กต์ของคุณได้เลย
    extract_frames_from_folder(
        data_dir="data",  # โฟลเดอร์เก็บวิดีโอต้นทาง
        output_base_dir="dataset_frames",  # โฟลเดอร์เก็บ Dataset ที่ตัดแล้ว
        interval_seconds=1.0,  # ดึง 1 ภาพต่อวินาที (ทุกๆ 1 วิ)
    )