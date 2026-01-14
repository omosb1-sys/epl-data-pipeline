from fpdf import FPDF
import os

class BeautifulPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        # Register Font
        font_path = "/Library/Fonts/Arial Unicode.ttf"
        self.add_font("ArialUnicode", "", font_path)
        self.add_font("ArialUnicode", "B", font_path)
        
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUnicode", size=8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, label):
        self.set_fill_color(0, 123, 255) # Bootstrap blue
        self.set_text_color(255, 255, 255)
        self.set_font("ArialUnicode", "B", 18)
        self.cell(0, 15, label, ln=1, fill=True, align="C")
        self.ln(5)

    def section_title(self, label):
        self.set_font("ArialUnicode", "B", 14)
        self.set_text_color(0, 86, 179)
        self.multi_cell(0, 10, label)
        self.ln(2)

    def sub_section_title(self, label):
        self.set_font("ArialUnicode", "B", 12)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 8, label)
        self.ln(1)

    def body_text(self, text):
        self.set_font("ArialUnicode", size=10.5)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 6.5, text)
        self.ln(2)

    def code_block(self, code_lines):
        # Improved code block with padding and border
        self.set_font("Courier", size=9)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(220, 220, 220)
        self.set_text_color(30, 30, 30)
        
        # Calculate height
        line_height = 5
        height = len(code_lines) * line_height + 6
        
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()
            
        current_y = self.get_y()
        self.rect(self.l_margin, current_y, 190, height, "FD")
        self.set_y(current_y + 3)
        
        for line in code_lines:
            self.set_x(self.l_margin + 5)
            if any(ord(c) > 127 for c in line):
                self.set_font("ArialUnicode", size=8.5)
            else:
                self.set_font("Courier", size=8.5)
            self.multi_cell(180, line_height, line)
        
        self.set_y(current_y + height + 4)
        self.set_font("ArialUnicode", size=11)

    def table_draw(self, table_data):
        self.set_font("ArialUnicode", "B", 9)
        self.set_fill_color(240, 244, 248)
        self.set_draw_color(200, 200, 200)
        
        # Custom widths for this specific table
        # Situations, Rec, Code
        widths = [45, 50, 95] # Total 190
        
        # Header
        for i, header in enumerate(table_data[0]):
            self.cell(widths[i], 10, header, border=1, align="C", fill=True)
        self.ln()
        
        # Rows
        self.set_font("ArialUnicode", size=8.5)
        self.set_text_color(50, 50, 50)
        for row in table_data[1:]:
            # Simple row rendering (assuming single line for now to avoid complexity)
            # Find max height
            for i, item in enumerate(row):
                if i < len(widths):
                    self.cell(widths[i], 8, item, border=1, align="L")
            self.ln()
        self.ln(5)

def parse_and_create():
    md_path = "/Users/sebokoh/.gemini/antigravity/brain/dfbbbf53-e34c-4ec9-a4fd-bfc660d0a986/python_study_examples.md"
    pdf_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/파이썬_실전_가이드_v2.pdf"
    
    pdf = BeautifulPDF()
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    in_code_block = False
    code_cache = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith("```"):
            if in_code_block:
                # End of block
                pdf.code_block(code_cache)
                code_cache = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
            
        if in_code_block:
            code_cache.append(line)
            i += 1
            continue
            
        if line.startswith("# "):
            pdf.chapter_title(line[2:].replace("**", ""))
        elif line.startswith("## "):
            pdf.section_title(line[3:].replace("**", ""))
        elif line.startswith("### "):
            pdf.sub_section_title(line[4:].replace("**", ""))
        elif line.startswith("|"):
            # Table start
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                if "---" not in lines[i]: # Skip separator
                    table_lines.append(lines[i])
                i += 1
            
            if table_lines:
                data = []
                for tl in table_lines:
                    row = [cell.strip() for cell in tl.split("|") if cell.strip()]
                    if row:
                        data.append(row)
                if data:
                    pdf.table_draw(data)
            continue # Already incremented i
        elif line.strip() == "":
            pdf.ln(2)
        else:
            # Normal text
            clean_line = line.replace("**", "").replace("`", "")
            if clean_line.startswith("> "):
                pdf.set_font("ArialUnicode", size=10)
                pdf.set_text_color(100, 100, 100)
                pdf.multi_cell(0, 7, clean_line[2:])
                pdf.set_font("ArialUnicode", size=11)
                pdf.set_text_color(33, 37, 41)
            else:
                pdf.body_text(clean_line)
        
        i += 1
        
    pdf.output(pdf_path)
    print(f"Improved PDF created at: {pdf_path}")

if __name__ == "__main__":
    parse_and_create()
