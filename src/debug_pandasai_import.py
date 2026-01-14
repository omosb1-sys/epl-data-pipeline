
try:
    from pandasai.llm import OpenAI
    print("Success: from pandasai.llm import OpenAI")
except ImportError as e:
    print(f"Failed: {e}")
    try:
        from pandasai.llm.openai import OpenAI
        print("Success: from pandasai.llm.openai import OpenAI")
    except ImportError as e2:
        print(f"Failed explicit import: {e2}")
