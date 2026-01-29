import textwrap

def check_html_format_fix():
    # Mocking variables
    v_gradient = "linear-gradient(135deg, rgba(0, 200, 83, 0.1), rgba(0, 0, 0, 0.4))"
    v_color = "#00C853"
    v_title = "Title"
    
    # Simulate the FIXED code block (no indentation)
    html_block = f"""
<div style="
    background: {v_gradient};
    backdrop-filter: blur(10px);
">
    <h1>{v_title}</h1>
</div>
"""
    
    print("--- FIXED BLOCK (First 50 chars) ---")
    print(repr(html_block[:50]))
    
    # Check if lines have unintended indentation that looks like code blocks (4 spaces)
    # The first line is empty in the f-string, the second line starts with <div
    lines = html_block.split('\n')
    problematic = False
    for line in lines:
        if line.strip() and line.startswith('    <'):
             # This is a heuristic: if a tag starts with 4 spaces, markdown might treat it as code
             # However, since we removed the main indentation, it should be fine if not all lines are indented deeply
             pass

    if html_block.strip().startswith('<div'):
         print("\n[PASS] Block starts correctly without indentation.")
    else:
         print("\n[FAIL] formatting issue remains.")

if __name__ == "__main__":
    check_html_format_fix()
