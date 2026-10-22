import os
import re
import xml.sax.saxutils as saxutils

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAMIL_FONT_PATH = os.path.join(BASE_DIR, "static", "fonts", "Lohit-Tamil.ttf")

# Register the Tamil font once (safe to call multiple times per process)
if "LohitTamil" not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont("LohitTamil", TAMIL_FONT_PATH))

# The bundled Tamil TTF has no Latin glyphs, so any English word or number
# inside a Tamil sentence is wrapped in Helvetica (built into reportlab)
# to avoid empty "tofu" boxes in the PDF.
_ASCII_RUN = re.compile(r"[\x00-\x7F]+")


def tf(text: str) -> str:
    """Escape + wrap ASCII runs in a Latin-capable font for mixed Tamil/English text."""
    escaped = saxutils.escape(str(text))

    def _wrap(match):
        return f'<font face="Helvetica">{match.group(0)}</font>'

    return _ASCII_RUN.sub(_wrap, escaped)


def build_pdf_report(output_path: str, image_path: str, result: dict, farmer_name: str):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontName="LohitTamil",
        textColor=colors.HexColor("#2D5F2D"), fontSize=19, leading=25,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontName="LohitTamil",
        textColor=colors.HexColor("#3E7C3E"), fontSize=14, leading=20,
    )
    normal_style = ParagraphStyle(
        "NormalTamil", parent=styles["Normal"], fontName="LohitTamil",
        fontSize=11, leading=17,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=normal_style, fontSize=8, leading=12, textColor=colors.grey,
    )
    cell_style = ParagraphStyle(
        "CellTamil", parent=normal_style, fontSize=10.5, leading=14,
    )

    story = []
    story.append(Paragraph(tf("LeafCare — இலை நோய் கண்டறியும் அறிக்கை"), title_style))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(tf(f"விவசாயி பெயர்: {farmer_name}"), normal_style))
    story.append(Paragraph(tf(f"தேதி: {result.get('timestamp', '')}"), normal_style))
    story.append(Spacer(1, 8 * mm))

    # Show original + highlighted (diseased-area marked) images side by side
    img_dir = os.path.dirname(image_path)
    highlighted_path = os.path.join(img_dir, result.get("highlighted_file", ""))

    try:
        img_row_data = [[
            Image(image_path, width=75 * mm, height=75 * mm),
            Image(highlighted_path, width=75 * mm, height=75 * mm)
            if os.path.exists(highlighted_path) else "",
        ]]
        img_table = Table(img_row_data, colWidths=[80 * mm, 80 * mm])
        story.append(img_table)
        story.append(Spacer(1, 3 * mm))
        cap_data = [[
            Paragraph(tf("மூலப் படம்"), cell_style),
            Paragraph(tf("நோய் பகுதி குறிக்கப்பட்ட படம்"), cell_style),
        ]]
        story.append(Table(cap_data, colWidths=[80 * mm, 80 * mm]))
        story.append(Spacer(1, 8 * mm))
    except Exception:
        pass

    story.append(Paragraph(tf("கண்டறிதல் சுருக்கம்"), heading_style))
    story.append(Spacer(1, 3 * mm))

    status_tamil = "ஆரோக்கியமானது" if result["status"] == "Healthy" else "நோய் கண்டறியப்பட்டது"
    status_color = "#2D8A2D" if result["status"] == "Healthy" else "#B23B2E"

    data = [
        [Paragraph(tf("நிலை"), cell_style), Paragraph(tf(status_tamil), cell_style)],
        [Paragraph(tf("கண்டறியப்பட்ட நோய்"), cell_style), Paragraph(tf(result["disease_name"]), cell_style)],
        [Paragraph(tf("தீவிரம்"), cell_style), Paragraph(tf(result.get("severity", "-")), cell_style)],
        [Paragraph(tf("நோய்வாய்ப்பட்ட பகுதி"), cell_style), Paragraph(tf(f"{result['diseased_percent']}%"), cell_style)],
        [Paragraph(tf("ஆரோக்கியமான பகுதி"), cell_style), Paragraph(tf(f"{result['healthy_percent']}%"), cell_style)],
    ]
    table = Table(data, colWidths=[60 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF4EA")),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(status_color)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9D6B9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(tf("எப்படி கண்டறியப்பட்டது"), heading_style))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(tf(
        "படம் RGB-லிருந்து Grayscale ஆக மாற்றப்பட்டு, Otsu Thresholding "
        "மூலம் இலை பகுதி பிரிக்கப்பட்டது. பின்னர் HSV வண்ண thresholding "
        "மூலம் ஆரோக்கியமான பச்சை பகுதியும் நோய்வாய்ப்பட்ட "
        "பழுப்பு/மஞ்சள்/கருப்பு பகுதியும் தனித்தனியாக கண்டறியப்பட்டு, "
        "நோய்வாய்ப்பட்ட பகுதியின் சதவீதம் கணக்கிடப்பட்டது."
    ), normal_style))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(tf("தடுப்பு / சிகிச்சை முறை"), heading_style))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(tf(result.get("advice", "-")), normal_style))
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph(tf(
        "குறிப்பு: இது ஒரு தானியங்கி பட பகுப்பாய்வு கருவி மூலம் "
        "உருவாக்கப்பட்ட முடிவு. கடுமையான பாதிப்பு இருந்தால் உங்கள் "
        "அருகிலுள்ள வேளாண் அலுவலரையும் அணுகவும்."
    ), footer_style))

    doc.build(story)
