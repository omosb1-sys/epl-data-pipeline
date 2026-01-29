import textwrap

def check_html_format():
    # Mocking variables
    v_gradient = "linear-gradient(135deg, rgba(0, 200, 83, 0.1), rgba(0, 0, 0, 0.4))"
    v_color = "#00C853"
    v_title = "Title"
    v_causal = "Causal text"
    v_trend = "Trend text"
    
    # Simulate the potentially problematic code block
    html_block = f"""
            <div style="
                background: {v_gradient};
                backdrop-filter: blur(10px);
            ">
                <h1>{v_title}</h1>
            </div>
            """
    
    dedented_block = textwrap.dedent(html_block)
    
    print("--- ORIGINAL BLOCK (First 50 chars) ---")
    print(repr(html_block[:50]))
    print("\n--- DEDENTED BLOCK (First 50 chars) ---")
    print(repr(dedented_block[:50]))
    
    if html_block.startswith('\n            <div'):
        print("\n[FAIL] The original block has indentation that Streamlit st.markdown interprets as code!")
    else:
         print("\n[PASS] Block seems OK.")

if __name__ == "__main__":
    check_html_format()
