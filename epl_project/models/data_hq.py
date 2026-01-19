# import polars as pl (Lazy)
# import duckdb (Lazy)
import os
import json
from datetime import datetime

class EPLDataHQ:
    """
    [EPL Data HQ] Modern Football Data Pipeline
    - Engine: Polars (Fast Processing) & DuckDB (Large Query)
    - Architecture: Zero-Copy Arrow Streams
    """
    def __init__(self, data_path: str = "../data/latest_epl_data.json"):
        import duckdb
        # 경로 보정 (models에서 실행 시와 root에서 실행 시 모두 대응)
        search_paths = [
            data_path,
            "data/latest_epl_data.json",
            "../data/latest_epl_data.json",
            os.path.join(os.path.dirname(__file__), "../data/latest_epl_data.json")
        ]
        
        final_path = data_path
        for p in search_paths:
            if os.path.exists(p):
                final_path = p
                break
            
        self.data_path = final_path
        self._db = duckdb.connect(':memory:')
        
    def load_and_transform(self) -> "pl.DataFrame":
        """JSON 비정형 데이터를 Polars DataFrame으로 변환 및 정규화"""
        import polars as pl
        print(f"🚀 [Data HQ] Loading source: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Source file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 1. 뉴스 데이터 처리
        news_list = raw_data.get('news', [])
        df_news = pl.from_dicts(news_list) if news_list else pl.DataFrame()
        if not df_news.is_empty():
            df_news = df_news.with_columns([
                pl.col("title").alias("title"),
                pl.col("title").str.to_lowercase().alias("title_low"),
                pl.lit("news").alias("data_type")
            ])
            
        # 2. 이적 데이터 처리
        trans_list = raw_data.get('transfers', [])
        df_trans = pl.from_dicts(trans_list) if trans_list else pl.DataFrame()
        if not df_trans.is_empty():
            # 이적 데이터용 필드 정규화 (필요시)
            df_trans = df_trans.with_columns([
                pl.col("player").str.to_lowercase().alias("title_low"), # 쿼리 호환성용
                pl.lit("transfer").alias("data_type")
            ])
            
        # 3. 데이터 결합 (결과 우선순위: 뉴스 > 이적)
        if not df_news.is_empty() and not df_trans.is_empty():
            # URL 필드 보존을 위해 공통 필드로 선택 및 결합
            return pl.concat([
                df_news.select(["title", "title_low", "url", "data_type"]), 
                df_trans.select([
                    pl.col("player").alias("title"),
                    pl.col("title_low"),
                    pl.lit("#").alias("url"), # 이적 공식 정보는 URL이 없을 수 있음
                    pl.lit("transfer").alias("data_type")
                ])
            ], how="diagonal")
        
        if not df_news.is_empty():
            return df_news.select(["title", "title_low", "url", "data_type"])
        return df_trans

    def query_with_duckdb(self, df: "pl.DataFrame", query: str):
        """Polars DF를 DuckDB에서 SQL로 고속 쿼리"""
        import duckdb
        # [SOTA Fix] 명시적으로 'df'라는 이름으로 등록하여 쿼리 안정성 확보
        self._db.register("df", df)
        return self._db.query(query).to_df()

if __name__ == "__main__":
    hq = EPLDataHQ()
    try:
        df = hq.load_and_transform()
        print(f"✅ Data Head:\n{df.head()}")
        
        # DuckDB 쿼리 테스트 (최근 뉴스 제목에 'Arsenal' 포함된 건수)
        res = hq.query_with_duckdb(df, "SELECT count(*) FROM df WHERE title_low LIKE '%arsenal%'")
        print(f"📊 Arsenal News Count: {res.iloc[0,0]}")
    except Exception as e:
        print(f"❌ HQ Pipeline Error: {e}")
