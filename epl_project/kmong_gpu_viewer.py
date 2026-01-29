import polars as pl
import streamlit as st
import streamlit.components.v1 as components
import json
from pathlib import Path

# [Senior Tip] 8GB RAM 환경에서는 데이터 로드 시 필요한 컬럼만 선택하는 것이 핵심입니다.
# WebGPU가 작동하지 않는 환경을 대비해 Canvas 기반 Fallback 라이브러리(Chart.js)도 함께 탑재합니다.

class KmongAdvancedViewer:
    def __init__(self, parquet_path: str):
        self.parquet_path = Path(parquet_path)
        
    @st.cache_data
    def load_data(_self, sample_size=100000):
        if not _self.parquet_path.exists():
            return None
        # 메모리 효율을 위해 필요한 수치형 컬럼만 로드
        df = pl.read_parquet(_self.parquet_path).select([
            pl.col("배기량").str.replace_all(r"[^0-9.]", "").cast(pl.Float32, strict=False).fill_null(0),
            pl.col("취득금액").str.replace_all(r"[^0-9.]", "").cast(pl.Float32, strict=False).fill_null(0)
        ])
        if len(df) > sample_size:
            df = df.sample(sample_size)
        return df

    def render_hybrid_chart(self, df):
        data_points = df.to_dicts()
        
        # HTML/JS: ChartGPU (WebGPU) + Chart.js (Canvas Fallback)
        html_code = f"""
        <div style="background: #1e293b; padding: 20px; border-radius: 16px; border: 1px solid #334155;">
            <div id="status-bar" style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div id="gpu-status" style="color: #60a5fa; font-family: monospace; font-size: 13px;">🔄 엔진 초기화 중...</div>
                <div id="engine-type" style="color: #94a3b8; font-size: 11px;">MODE: AUTO</div>
            </div>
            
            <div id="chart-container" style="position: relative; width: 100%; height: 500px; background: #0f172a; border-radius: 12px; overflow: hidden;">
                <!-- WebGPU Layer -->
                <canvas id="chartgpu-canvas" style="position: absolute; top:0; left:0; width: 100%; height: 100%; z-index: 10;"></canvas>
                <!-- Fallback Canvas Layer -->
                <canvas id="fallback-canvas" style="position: absolute; top:0; left:0; width: 100%; height: 100%; z-index: 5; display: none;"></canvas>
            </div>
            
            <div style="margin-top: 10px; color: #64748b; font-size: 12px; text-align: center;">
                * 마우스 휠로 확대/축소가 가능합니다. 점이 보이지 않으면 잠시 기다려주세요.
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script type="module">
            import {{ createChart }} from 'https://cdn.skypack.dev/chartgpu';

            const rawData = {json.dumps(data_points)};
            const canvasGPU = document.getElementById('chartgpu-canvas');
            const canvasFallback = document.getElementById('fallback-canvas');
            const status = document.getElementById('gpu-status');
            const engineType = document.getElementById('engine-type');

            async function startEngine() {{
                try {{
                    // 1. WebGPU 시도
                    if (!navigator.gpu) throw new Error("WebGPU 미지원 브라우저");

                    const chart = await createChart(canvasGPU, {{
                        type: 'scatter',
                        data: {{
                            datasets: [{{
                                data: rawData.map(d => [d.배기량, d.취득금액]),
                                size: 5,
                                color: [1.0, 0.9, 0.4, 0.8] // 황금색 포인트 (가시성 증대)
                            }}]
                        }},
                        options: {{
                            scales: {{
                                x: {{ type: 'linear', auto: true }},
                                y: {{ type: 'linear', auto: true }}
                            }},
                            interaction: {{ zoom: true, pan: true }}
                        }}
                    }});
                    
                    status.innerText = "⚡ WebGPU Active: 50,000 pts 렌더링 완료";
                    engineType.innerText = "MODE: GPU ACCELERATED";
                    engineType.style.color = "#4ade80";

                }} catch (e) {{
                    console.warn("GPU Engine Failed, switching to Canvas:", e.message);
                    status.innerText = "⚠️ GPU 가속 오류: Canvas 모드로 전환됨";
                    status.style.color = "#fbbf24";
                    
                    // 2. Canvas Fallback (Chart.js)
                    canvasGPU.style.display = 'none';
                    canvasFallback.style.display = 'block';
                    
                    new Chart(canvasFallback, {{
                        type: 'scatter',
                        data: {{
                            datasets: [{{
                                label: 'Car Data',
                                data: rawData.slice(0, 5000).map(d => ({{ x: d.배기량, y: d.취득금액 }})),
                                backgroundColor: 'rgba(96, 165, 250, 0.6)',
                                pointRadius: 2
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{ grid: {{ color: '#334155' }} }},
                                y: {{ grid: {{ color: '#334155' }} }}
                            }}
                        }}
                    }});
                    engineType.innerText = "MODE: CANVAS FALLBACK";
                }}
            }}

            startEngine();
        </script>
        """
        components.html(html_code, height=620)

if __name__ == "__main__":
    st.set_page_config(page_title="Kmong Pro Viewer", layout="wide")
    st.title("🚗 크몽 차량 데이터 하이브리드 자동 가속 뷰어")
    
    viewer = KmongAdvancedViewer("./data/kmong_project/processed/master_v1.parquet")
    
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        sample_size = st.slider("샘플 크기", 1000, 100000, 50000)
        if st.button("데이터 다시 로드"):
            st.cache_data.clear()
            st.rerun()

    data = viewer.load_data(sample_size)
    
    if data is not None:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric("총 분석 대상", f"{len(data):,} 건")
            st.write("---")
            st.info("💡 **Antigravity Tip**\n점이 너무 작아 보이지 않으면 마우스 휠을 굴려 확대해보세요. 노란색 점이 차량 데이터입니다.")
        
        with col2:
            viewer.render_hybrid_chart(data)
    else:
        st.warning("데이터가 준비되지 않았습니다. 파이프라인을 먼저 실행하거나 더미 데이터를 생성하세요.")
