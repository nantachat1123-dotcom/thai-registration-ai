import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import cv2
import io, re, json
from datetime import datetime

st.set_page_config(page_title="Thai Registration AI", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1.8rem;}
.hero {padding: 1.4rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#142b46,#24547b); color:white; margin-bottom:1rem;}
.hero p {color:#d5e4f2; margin-bottom:0;}
.small-note {font-size:.85rem; color:#64748b;}
div[data-testid="stFileUploader"] {border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🔎 THAI REGISTRATION AI</h1>
  <p>ระบบช่วยคัดกรองภาพเล่มทะเบียนรถยนต์ไทย</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "ระบบนี้เป็นเครื่องมือช่วยคัดกรองภาพ ไม่ใช่เครื่องมือรับรองเอกสารแท้/ปลอม "
    "ผลวิเคราะห์จากภาพเพียงอย่างเดียวไม่สามารถยืนยันสถานะทางทะเบียนได้"
)

with st.sidebar:
    st.header("ตั้งค่าการตรวจ")
    use_ocr = st.checkbox("อ่านข้อความ OCR ภาษาไทย/อังกฤษ", value=True)
    st.caption("OCR ต้องติดตั้ง Tesseract พร้อมชุดภาษา tha และ eng บนเซิร์ฟเวอร์")
    st.divider()
    st.caption("รุ่นต้นแบบนี้ไม่มีการเชื่อมต่อฐานข้อมูลกรมการขนส่งทางบก")

st.subheader("1. อัปโหลดภาพเอกสาร")
files = st.file_uploader(
    "เลือกภาพหน้าปกและหน้าภายในเล่มทะเบียน",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="แนะนำภาพต้นฉบับที่ชัดเจน เห็นขอบเอกสารครบ ไม่มีแสงสะท้อน"
)

def analyze(pil):
    rgb = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    contrast = float(gray.std())
    h, w = gray.shape
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)
    notes = []
    if min(w, h) < 1000:
        notes.append(("ควรถ่ายใหม่", "ภาพมีความละเอียดต่ำ รายละเอียดขนาดเล็กอาจอ่านไม่ชัด"))
    if sharp < 45:
        notes.append(("ควรถ่ายใหม่", "ภาพอาจเบลอหรือสั่น"))
    if bright < 45:
        notes.append(("ควรถ่ายใหม่", "ภาพมืด อาจมองไม่เห็นรายละเอียดบางส่วน"))
    elif bright > 220:
        notes.append(("ควรถ่ายใหม่", "ภาพสว่างหรือสะท้อนมากเกินไป"))
    if contrast < 20:
        notes.append(("ข้อมูลจำกัด", "ความต่างระดับสีต่ำ"))
    if not notes:
        notes.append(("คุณภาพภาพเบื้องต้น", "ไม่พบปัญหาคุณภาพภาพเด่นชัดจากตัวชี้วัดพื้นฐาน"))
    return {"width":w, "height":h, "sharpness":sharp, "brightness":bright,
            "contrast":contrast, "edge_ratio":edge_ratio, "notes":notes}

def exif_data(img):
    try:
        result = {}
        for key, val in img.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if name in ("Make", "Model", "DateTime", "Software", "Orientation"):
                result[name] = str(val)
        return result
    except Exception:
        return {}

def ocr_text(img):
    try:
        import pytesseract
        return pytesseract.image_to_string(img, lang="tha+eng", config="--psm 6"), None
    except Exception as e:
        return None, str(e)

if files:
    st.subheader("2. วิเคราะห์ภาพ")
    texts, report = [], [
        "รายงานคัดกรองภาพเล่มทะเบียนรถยนต์ไทย",
        f"วันที่ตรวจ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "คำเตือน: ไม่ใช่การรับรองความแท้หรือปลอม"
    ]
    for i, file in enumerate(files):
        try:
            img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        except Exception:
            st.error(f"อ่านไฟล์ {file.name} ไม่ได้")
            continue
        result = analyze(img)
        with st.expander(f"ภาพที่ {i+1}: {file.name}", expanded=(i == 0)):
            col1, col2 = st.columns([1.15, 1])
            with col1:
                st.image(img, use_container_width=True)
            with col2:
                st.metric("ความละเอียด", f'{result["width"]} × {result["height"]} px')
                st.metric("ความคมชัด", f'{result["sharpness"]:.1f}')
                st.metric("ความสว่าง", f'{result["brightness"]:.1f} / 255')
                st.metric("Contrast", f'{result["contrast"]:.1f}')
                st.caption("ตัวเลขเหล่านี้ประเมินคุณภาพภาพ ไม่ใช่คะแนนความแท้")
            st.markdown("**ข้อสังเกต**")
            for label, note in result["notes"]:
                st.write(f"- **{label}:** {note}")
            with st.expander("ข้อมูลไฟล์ภาพ"):
                st.json(exif_data(img) or {"ข้อมูล":"ไม่พบ EXIF หรือ metadata ถูกลบ"})
            text_value = ""
            if use_ocr:
                with st.expander("ผลอ่านข้อความ OCR", expanded=True):
                    value, error = ocr_text(img)
                    if error:
                        st.info("OCR ยังไม่พร้อมใช้งานบนเซิร์ฟเวอร์นี้ กรุณาติดตั้ง Tesseract และภาษา tha/eng")
                    else:
                        text_value = value.strip()
                        st.text_area("ข้อความที่อ่านได้ (ควรตรวจทานกับภาพ)", text_value, height=160, key=f"ocr_{i}")
                        texts.append((file.name, text_value))
            report.extend([
                "", f"ไฟล์: {file.name}",
                f"ความละเอียด: {result['width']}x{result['height']}",
                f"ความคมชัด: {result['sharpness']:.2f}",
                f"ความสว่าง: {result['brightness']:.2f}",
                *[f"- {a}: {b}" for a,b in result["notes"]]
            ])
            if text_value:
                report.extend(["ข้อความ OCR:", text_value])

    if len(texts) > 1:
        st.subheader("3. เปรียบเทียบข้อความที่อ่านได้")
        st.caption("เป็นการเทียบผล OCR ระหว่างภาพเท่านั้น อาจมีความผิดพลาดจากการอ่านตัวอักษร")
        patterns = {
            "เลขทะเบียน": r"(?:ทะเบียนรถ|เลขทะเบียน|หมายเลขทะเบียน)\s*[:：]?\s*([ก-ฮ]{1,2}\s*\d{1,4})",
            "เลขตัวถัง": r"(?:เลขตัวถัง|หมายเลขตัวถัง|CHASSIS)\s*[:：]?\s*([A-Z0-9\-]{6,})",
            "เลขเครื่องยนต์": r"(?:เลขเครื่องยนต์|หมายเลขเครื่องยนต์|ENGINE)\s*[:：]?\s*([A-Z0-9\-]{4,})"
        }
        for label, pattern in patterns.items():
            found = [(name, m.group(1).strip()) for name, txt in texts
                     if (m := re.search(pattern, txt, re.I))]
            if found:
                st.markdown(f"**{label}**")
                for name, value in found:
                    st.write(f"- {name}: `{value}`")
                normalized = {v.upper().replace(" ", "") for _,v in found}
                if len(normalized) > 1:
                    st.error("พบค่าที่ OCR อ่านได้ไม่ตรงกัน โปรดตรวจสอบต้นฉบับ")
                elif len(found) > 1:
                    st.success("ค่าที่ OCR อ่านได้ตรงกันในภาพที่มีข้อมูล")

    st.download_button("⬇️ ดาวน์โหลดรายงาน TXT", "\n".join(report).encode("utf-8"),
                       "thai_registration_report.txt", "text/plain")
else:
    st.info("ยังไม่มีภาพ กรุณาอัปโหลดภาพเล่มทะเบียนเพื่อเริ่มตรวจ")
    st.markdown("""
    **คำแนะนำก่อนอัปโหลด**
    - ถ่ายหน้าปกและหน้าข้อมูลที่ต้องการตรวจให้เห็นเต็มหน้า
    - ใช้แสงสม่ำเสมอ ไม่ให้เกิดเงาหรือแสงสะท้อน
    - ถ่ายภาพตรง ไม่เอียง และไม่ตัดขอบกระดาษ
    - ปกปิดชื่อ ที่อยู่ เลขประจำตัวประชาชน และข้อมูลส่วนบุคคลที่ไม่จำเป็น
    """)

st.divider()
st.caption("Thai Registration AI • Prototype • ผลการคัดกรองไม่ใช่การยืนยันทางราชการ")
