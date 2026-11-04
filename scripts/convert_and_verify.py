import os
import win32com.client
import pdfplumber

def convert_docx_to_pdf(docx_path, pdf_path):
    abs_docx = os.path.abspath(docx_path)
    abs_pdf = os.path.abspath(pdf_path)
    
    print(f"Converting '{abs_docx}' -> '{abs_pdf}' via Word COM...")
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(abs_docx)
        doc.SaveAs(abs_pdf, FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
    finally:
        word.Quit()
    print("Conversion complete!")

def verify_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        print(f"\nVerifying '{pdf_path}':")
        print(f"Total pages: {len(pdf.pages)}")
        for i, p in enumerate(pdf.pages):
            text = (p.extract_text() or '').strip()
            first_line = text.splitlines()[0] if text else '[EMPTY]'
            print(f"Page {i+1:02d}: lines={len(text.splitlines())} | Header/Start: {first_line[:60]}")

if __name__ == "__main__":
    convert_docx_to_pdf("docs/updated_report.docx", "docs/report_pdf_updated.pdf")
    verify_pdf("docs/report_pdf_updated.pdf")

