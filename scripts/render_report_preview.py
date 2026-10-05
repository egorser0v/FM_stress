"""Render an existing PDF to per-page PNGs and a numbered contact sheet.

Read-only with respect to the PDF. Uses pypdfium2 (also a pdfplumber dependency),
avoiding external fontconfig / platform-specific renderer setup.
"""
import argparse
import json
import math
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdf',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--dpi',type=int,default=150)
    p.add_argument('--columns',type=int,default=3)
    p.add_argument('--thumbnail-width',type=int,default=330)
    p.add_argument('--expected-pages',type=int)
    a=p.parse_args()
    if a.dpi<36 or a.columns<1:raise SystemExit('dpi must be >=36 and columns >=1')
    out=a.output or a.pdf.parent/(a.pdf.stem+'-preview')
    out.mkdir(parents=True,exist_ok=True)
    doc=pdfium.PdfDocument(a.pdf)
    if a.expected_pages is not None and len(doc)!=a.expected_pages:
        raise SystemExit(f'Expected {a.expected_pages} pages, found {len(doc)}')
    thumbs=[];records=[]
    for i in range(len(doc)):
        page=doc[i]
        bitmap=page.render(scale=a.dpi/72)
        image=bitmap.to_pil().convert('RGB')
        path=out/f'page-{i+1:02d}.png'
        image.save(path)
        thumb=image.copy();thumb.thumbnail((a.thumbnail_width,10000),Image.Resampling.LANCZOS)
        textpage=page.get_textpage()
        records.append({'page':i+1,'width_pt':page.get_width(),'height_pt':page.get_height(),
                        'text_characters':textpage.count_chars(),'png':str(path)})
        thumbs.append(thumb)
        textpage.close();bitmap.close();page.close()
    doc.close()
    margin=16;label_height=28
    cellw=a.thumbnail_width+2*margin;cellh=max(im.height for im in thumbs)+2*margin+label_height
    sheet=Image.new('RGB',(a.columns*cellw,math.ceil(len(thumbs)/a.columns)*cellh),'#E5ECEF')
    draw=ImageDraw.Draw(sheet)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    for i,thumb in enumerate(thumbs):
        x=(i%a.columns)*cellw+margin;y=(i//a.columns)*cellh+margin
        draw.text((x,y),f'PAGE {i+1:02d}',fill='#21414F',font=font)
        sheet.paste(thumb,(x,y+label_height))
    sheet.save(out/'contact-sheet.png')
    (out/'render-manifest.json').write_text(json.dumps({'source':str(a.pdf),'dpi':a.dpi,'pages':records},indent=2)+'\n')
    print(out/'contact-sheet.png')


if __name__=='__main__':main()
