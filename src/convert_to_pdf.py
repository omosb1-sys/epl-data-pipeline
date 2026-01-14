from fpdf import FPDF
import os

class StudyGuidePDF(FPDF):
    def header(self):
        # We'll set the font for the header later in the main script
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUnicode", size=8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_pdf(input_md, output_pdf, font_path):
    pdf = StudyGuidePDF()
    pdf.add_page()
    
    # Register Korean font
    pdf.add_font("ArialUnicode", "", font_path)
    pdf.add_font("ArialUnicode", "B", font_path) # Simplified: using same font for bold if needed, or we can just shift size
    
    with open(input_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf.set_font("ArialUnicode", size=11)
    
    for line in lines:
        line = line.strip()
        
        if not line:
            pdf.ln(5)
            continue
            
        # Very simple Markdown handling
        if line.startswith("# "):
            pdf.set_font("ArialUnicode", style="B", size=18)
            pdf.multi_cell(190, 10, line[2:])
            pdf.ln(5)
            pdf.set_font("ArialUnicode", size=11)
        elif line.startswith("## "):
            pdf.set_font("ArialUnicode", style="B", size=14)
            pdf.multi_cell(190, 10, line[3:])
            pdf.ln(3)
            pdf.set_font("ArialUnicode", size=11)
        elif line.startswith("### "):
            pdf.set_font("ArialUnicode", style="B", size=12)
            pdf.multi_cell(190, 8, line[4:])
            pdf.ln(2)
            pdf.set_font("ArialUnicode", size=11)
        elif line.startswith("```"):
            # Code block starts or ends
            pdf.set_font("ArialUnicode", size=9) # Stay with unicode for comments
            pass
        elif line.startswith("|"):
            # Table lines (just print as is for now)
            pdf.set_font("ArialUnicode", size=8)
            pdf.multi_cell(190, 6, line)
            pdf.set_font("ArialUnicode", size=11)
        else:
            # Handle bold/italic markers crudely by removing them or keeping them
            clean_line = line.replace("**", "").replace("`", "")
            if clean_line.strip():
                pdf.multi_cell(190, 7, clean_line)

    pdf.output(output_pdf)

if __name__ == "__main__":
    input_path = "/Users/sebokoh/.gemini/antigravity/brain/dfbbbf53-e34c-4ec9-a4fd-bfc660d0a986/python_study_examples.md"
    output_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/파이썬_실전_가이드.pdf"
    # Found earlier: /Library/Fonts/Arial Unicode.ttf
    font = "/Library/Fonts/Arial Unicode.ttf"
    
    create_pdf(input_path, output_path, font)
    print(f"PDF created at: {output_path}")
