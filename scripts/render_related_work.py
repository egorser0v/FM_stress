"""Export the existing critical synthesis to one readable A4 PDF.

Run with the bundled Python (reportlab + pdfplumber) or install those two standard
packages in a project environment. Input Markdown remains the editable source.
"""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import pdfplumber


def markup(text):
    parts=[];pos=0
    # Convert source links without exposing long URLs in body text.
    for m in re.finditer(r'\[([^]]+)\]\(([^)]+)\)',text):
        parts.append(escape(text[pos:m.start()]))
        parts.append(f'<link href="{escape(m.group(2))}" color="#147A80">{escape(m.group(1))}</link>')
        pos=m.end()
    parts.append(escape(text[pos:]));out=''.join(parts)
    out=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',out)
    return out.replace('→','to').replace('−','-').replace('–','-').replace('×','x')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('docs/related_work.md'))
    p.add_argument('--output',type=Path,default=Path('output/pdf/related-work.pdf'))
    a=p.parse_args();text=a.source.read_text().split('## Implementation provenance')[0].strip()
    paragraphs=re.split(r'\n\s*\n',text)
    title=paragraphs.pop(0).lstrip('# ').strip()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    font='Helvetica';bold='Helvetica-Bold'
    font_dirs=list(Path('.venv/lib').glob('python*/site-packages/matplotlib/mpl-data/fonts/ttf'))
    if font_dirs:
        font_dir=font_dirs[0]
        pdfmetrics.registerFont(TTFont('ResearchSans',str(font_dir/'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('ResearchSans-Bold',str(font_dir/'DejaVuSans-Bold.ttf')))
        pdfmetrics.registerFontFamily('ResearchSans',normal='ResearchSans',bold='ResearchSans-Bold',italic='ResearchSans',boldItalic='ResearchSans-Bold')
        font='ResearchSans';bold='ResearchSans-Bold'
    body=ParagraphStyle('Body',fontName=font,fontSize=10.1,leading=13.2,textColor=colors.HexColor('#243640'),spaceAfter=9)
    heading=ParagraphStyle('Title',fontName=bold,fontSize=20,leading=23,textColor=colors.HexColor('#123A4A'),spaceAfter=8)
    kicker=ParagraphStyle('Kicker',fontName=bold,fontSize=9,leading=12,textColor=colors.HexColor('#147A80'),spaceAfter=10)
    story=[Paragraph('RELATED WORK  /  TASK 1',kicker),Paragraph(escape(title),heading),Spacer(1,3)]
    story.extend(Paragraph(markup(s.replace('\n',' ')),body) for s in paragraphs)
    def footer(canvas,doc):
        width,height=A4
        canvas.setStrokeColor(colors.HexColor('#DCE5E8'));canvas.line(40,35,width-40,35)
        canvas.setFillColor(colors.HexColor('#60737C'));canvas.setFont(font,8)
        canvas.drawString(40,22,'Smooth source, unstructured head  |  Critical literature synthesis')
        canvas.drawRightString(width-40,22,str(doc.page))
    doc=SimpleDocTemplate(str(a.output),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=45,
                          title=title,author='Egor Serov; Kirill Frolov; Daniil Koblov; Vasilii Lyamin')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    with pdfplumber.open(a.output) as pdf:
        if len(pdf.pages)!=1:
            raise RuntimeError(f'Related-work note must be one page; got {len(pdf.pages)}. Revise source or layout.')
        if 'TSFlow' not in pdf.pages[0].extract_text():raise RuntimeError('PDF text integrity check failed')
    print(a.output)


if __name__=='__main__':main()
