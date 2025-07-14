from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
from pathlib import Path

def add_noise(img, stddev=350):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, stddev, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def generate_5_lr_versions(hr_path, output_folder):
    img_hr = Image.open(hr_path)
    hr_w, hr_h = img_hr.size
    filename_stem = hr_path.stem

    # Case 1: Bicubic
    lr1 = img_hr.resize((hr_w // 12, hr_h // 12), resample=Image.BICUBIC)
    lr1.save(output_folder / f"{filename_stem}_bicubic_12x.jpg")

    # Case 2: Lanczos
    lr2 = img_hr.resize((hr_w // 8, hr_h // 8), resample=Image.LANCZOS)
    lr2 = lr2.convert("P", palette=Image.ADAPTIVE, colors=4)
    lr2 = lr2.convert("RGB")
    lr2.save(output_folder / f"{filename_stem}_low_quality.jpg", quality=10)

    # Case 3: Blurred
    blurred = img_hr.filter(ImageFilter.GaussianBlur(radius=10))
    lr3 = blurred.resize((hr_w // 10, hr_h // 10), resample=Image.BICUBIC)
    lr3.save(output_folder / f"{filename_stem}_blurred.jpg")

    # # Case 4: Noisy
    # noisy = add_noise(img_hr, stddev=250)
    # lr4 = noisy.resize((hr_w // 16, hr_h // 16), resample=Image.BICUBIC)
    # lr4.save(output_folder / f"{filename_stem}_noisy.jpg")

    # # Case 5: JPEG artifact
    # temp_path = output_folder / f"{filename_stem}_jpeg_temp.jpg"
    # img_hr.resize((hr_w // 4, hr_h // 4), resample=Image.BICUBIC).save(temp_path, quality=0)
    # Image.open(temp_path).save(output_folder / f"{filename_stem}_jpeg_artifact.jpg")
    # temp_path.unlink()


import os
import random

# 기본 설정
output_hr_folder = Path(__file__).parent.parent / 'data/train_HR_gen'
output_lr_folder = Path(__file__).parent.parent / 'data/train_LR_gen'
font_path = Path(__file__).parent.parent / 'data/fonts/hy-m-yoond1004.ttf'

font_size_digit = 92
font_size_korean = 86
os.makedirs(output_hr_folder, exist_ok=True)
os.makedirs(output_lr_folder, exist_ok=True)

# 랜덤 샘플
front_numbers = random.sample([f"{i:03d}" for i in range(1000)], 10)
korean_chars = random.sample([
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머',
    '버', '서', '어', '저', '고', '노', '도', '로', '모',
    '보', '소', '오', '조', '구', '누', '두', '루', '무',
    '부', '수', '우', '주', '아', '바', '사', '자', '하'
], 5)
back_numbers = random.sample([f"{i:04d}" for i in range(10000)], 20)

# 조합 및 이미지 생성
for front in front_numbers:
    for kor in korean_chars:
        for back in back_numbers:
            text = f"{front}{kor} {back}"
            filename = f"{front}{kor}{back}.jpg"

            # 고해상도 이미지 생성
            img_hr = Image.new('RGB', (600, 165), color='white')
            draw = ImageDraw.Draw(img_hr)
            x, y = 10, 30

            for ch in text:
                font_size = font_size_digit if ch.isdigit() else font_size_korean
                font = ImageFont.truetype(str(font_path), font_size)
                draw.text((x, y), ch, font=font, fill='black')
                bbox = draw.textbbox((x, y), ch, font=font)
                x += bbox[2] - bbox[0]

            # HR 저장
            hr_path = output_hr_folder / filename
            img_hr.save(hr_path)

            # LR 이미지 생성 (예: 1/4 축소)
            img_lr = img_hr.resize((150, 41), resample=Image.BICUBIC)
            lr_path = output_lr_folder / filename
            img_lr.save(lr_path)

print("✅ HR + LR 번호판 이미지 생성 완료.")


from pathlib import Path

hr_folder = Path(__file__).parent.parent / 'data/train_HR_gen'
lr_output_folder = Path(__file__).parent.parent / 'data/train_LR_gen'
lr_output_folder.mkdir(parents=True, exist_ok=True)

# 모든 .jpg 파일 대상
hr_images = list(hr_folder.glob("*.jpg"))

print(f"🔍 총 {len(hr_images)}개의 HR 이미지 처리 시작")

for hr_img_path in hr_images:
    generate_5_lr_versions(hr_img_path, lr_output_folder)

print("✅ 모든 HR 이미지에 대해 5개 LR 버전 생성 완료")
