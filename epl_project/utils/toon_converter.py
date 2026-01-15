import json
import os
from pytoony import json2toon

def convert_epl_data_to_toon():
    """Converts the main project JSON data to TOON format for token optimization."""
    json_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data/latest_epl_data.json"
    toon_path = "/Users/sebokoh/데이터분석연습/데이콘/k리그데이터/리그데이터/epl_project/data/latest_epl_data.toon"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON string to TOON format
        toon_content = json2toon(json.dumps(data))
        
        with open(toon_path, 'w', encoding='utf-8') as f:
            f.write(toon_content)
        
        # Compare Sizes
        json_size = os.path.getsize(json_path)
        toon_size = os.path.getsize(toon_path)
        reduction = (1 - (toon_size / json_size)) * 100
        
        print(f"✅ Conversion Successful!")
        print(f"📄 JSON Size: {json_size} bytes")
        print(f"📄 TOON Size: {toon_size} bytes")
        print(f"📉 Reduction: {reduction:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    convert_epl_data_to_toon()
