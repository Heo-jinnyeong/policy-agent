from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'docs/Policy_Agent_발표대본.md'
OUTPUT = ROOT / 'output/pdf/Policy_Agent_발표대본.pdf'
FONT = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'

pdfmetrics.registerFont(TTFont('AppleGothic', FONT))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

ink = colors.HexColor('#213137')
teal = colors.HexColor('#367c78')
muted = colors.HexColor('#657275')
rule = colors.HexColor('#d0d9d7')

title_style = ParagraphStyle(
    'title', fontName='AppleGothic', fontSize=20, leading=30,
    textColor=ink, spaceAfter=4 * mm, alignment=TA_LEFT,
)
intro_style = ParagraphStyle(
    'intro', fontName='AppleGothic', fontSize=9, leading=15,
    textColor=muted, spaceAfter=5 * mm,
)
heading_style = ParagraphStyle(
    'heading', fontName='AppleGothic', fontSize=13.1, leading=19,
    textColor=teal, spaceAfter=2 * mm,
)
body_style = ParagraphStyle(
    'body', fontName='AppleGothic', fontSize=10.2, leading=17.2,
    textColor=ink, spaceAfter=0,
)


def markup(raw: str) -> str:
    text = html.escape(raw, quote=False)
    text = re.sub(
        r'(\[화면:[^\]]+\])',
        lambda m: f'<font color="#367c78">{m.group(1)}</font>',
        text,
    )
    text = re.sub(
        r'`([^`]+)`',
        lambda m: f'<font color="#367c78">{m.group(1)}</font>',
        text,
    )
    return text


source = SOURCE.read_text(encoding='utf-8')
matches = list(re.finditer(r'^## (\d+)페이지 — (.+)$', source, re.M))
assert [int(m.group(1)) for m in matches] == list(range(1, 27))

story = [
    Paragraph('Policy Agent 발표 대본', title_style),
    Paragraph('본문 1~20페이지 · 부록 21~26페이지<br/>청록색 [화면] 표시는 발표자가 가리킬 부분이며 읽지 않습니다.', intro_style),
    HRFlowable(width='100%', thickness=1, color=teal),
    Spacer(1, 5 * mm),
]

for idx, match in enumerate(matches):
    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
    speech = source[match.end():end].strip()
    assert speech and '\n\n' not in speech, (idx + 1, speech)
    heading = f'{match.group(1)}페이지  {match.group(2)}'
    block = [
        Paragraph(html.escape(heading), heading_style),
        Paragraph(markup(speech), body_style),
        Spacer(1, 4 * mm),
        HRFlowable(width='100%', thickness=0.5, color=rule),
        Spacer(1, 4 * mm),
    ]
    story.append(KeepTogether(block))


def decorate(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(rule)
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, height - 18 * mm, width - 22 * mm, height - 18 * mm)
    canvas.setFont('AppleGothic', 8.5)
    canvas.setFillColor(muted)
    canvas.drawString(22 * mm, height - 14 * mm, 'POLICY AGENT · 발표 대본')
    canvas.line(22 * mm, 17 * mm, width - 22 * mm, 17 * mm)
    canvas.drawRightString(width - 22 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=A4,
    rightMargin=22 * mm, leftMargin=22 * mm,
    topMargin=27 * mm, bottomMargin=25 * mm,
    title='Policy Agent 발표 대본', author='Policy Agent',
)
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(OUTPUT)
