import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import queue
from datetime import datetime, timedelta
import traceback

# 사용자 정의 모듈 가져오기
from 유클리디안 import EuclideanPatternAnalyzer
from 상관계수 import CorrelationPatternAnalyzer
from 행렬프로파일 import StumpyPatternAnalyzer
from 형태기반거리 import ShapeBasedPatternAnalyzer


class AnalysisThread(threading.Thread):
    """비동기 분석 작업을 처리하는 워커 스레드"""

    def __init__(self, analyzer, symbol, timeframe, start_date, end_date, queue):
        threading.Thread.__init__(self)
        self.analyzer = analyzer
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.queue = queue
        self.daemon = True  # 메인 스레드가 종료되면 같이 종료

    def run(self):
        """스레드 실행 메서드 - 전체 분석 과정 수행"""
        try:
            # 데이터 로드
            self.queue.put(("log", f"데이터 로드 중... (시간프레임: {self.timeframe})"))
            df = self.analyzer.load_data(
                self.symbol, self.timeframe, self.start_date, self.end_date
            )

            # 고급 알고리즘 사용 시 추가 안내
            if isinstance(self.analyzer, StumpyPatternAnalyzer) or isinstance(
                self.analyzer, ShapeBasedPatternAnalyzer
            ):
                self.queue.put(
                    ("log", "고급 패턴 매칭 알고리즘은 계산량이 많아 시간이 오래 걸릴 수 있습니다.")
                )

                # 데이터가 너무 많으면 경고
                if len(df) > 10000:
                    self.queue.put(
                        (
                            "log",
                            f"데이터가 많습니다 ({len(df)} 행). 처리에 시간이 오래 걸릴 수 있습니다.",
                        )
                    )

            if df is None or df.empty:
                self.queue.put(("error", "데이터 로드 실패"))
                return

            self.queue.put(("log", f"패턴 추출 중..."))
            patterns, timestamps = self.analyzer.extract_patterns(df)

            if len(patterns) < 2:
                self.queue.put(("error", "패턴이 충분하지 않습니다"))
                return

            # 현재 패턴 (가장 최근 패턴)
            current_pattern = patterns[-1]

            # 유사 패턴 찾기
            self.queue.put(("log", "유사 패턴 찾는 중..."))
            most_similar = self.analyzer.find_most_similar_historical_pattern(
                current_pattern, patterns[:-1], timestamps[:-1], top_n=10
            )

            self.queue.put(("finished", (df, current_pattern, most_similar)))

        except Exception as e:
            self.queue.put(("error", f"분석 오류: {str(e)}"))
            traceback.print_exc()


class PatternAnalyzerApp:
    """비트코인 패턴 분석기 메인 애플리케이션 클래스"""

    def __init__(self, root):
        """애플리케이션 초기화"""
        self.root = root
        self.root.title("비트코인 패턴 분석기")
        self.root.geometry("1200x800")

        # GPU 가속 사용 여부 설정
        self.use_gpu = True  # 기본값

        # 분석기 초기화 (모두 GPU 가속 지원)
        self.euclidean_analyzer = EuclideanPatternAnalyzer(window_size=60, use_gpu=self.use_gpu)
        self.correlation_analyzer = CorrelationPatternAnalyzer(window_size=60, use_gpu=self.use_gpu)
        self.stumpy_analyzer = StumpyPatternAnalyzer(window_size=60, use_gpu=self.use_gpu)
        self.shape_analyzer = ShapeBasedPatternAnalyzer(window_size=60, use_gpu=self.use_gpu)

        self.current_analyzer = self.euclidean_analyzer  # 기본값은 유클리디안

        # 통신을 위한 큐 생성
        self.queue = queue.Queue()

        # UI 초기화
        self.init_ui()

        # 큐 처리 시작
        self.process_queue()

    def init_ui(self):
        """사용자 인터페이스 초기화"""
        # 컨트롤 프레임 생성
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        # 분석 대상 심볼 선택
        ttk.Label(control_frame, text="심볼:").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_combo = ttk.Combobox(
            control_frame, values=["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]
        )
        self.symbol_combo.current(0)  # 기본값: BTC/USDT
        self.symbol_combo.grid(row=0, column=1, padx=5, pady=5)

        # 시간프레임 선택
        ttk.Label(control_frame, text="시간프레임:").grid(row=0, column=2, padx=5, pady=5)
        self.timeframe_combo = ttk.Combobox(control_frame, values=["15m", "1h", "4h", "1d"])
        self.timeframe_combo.current(3)  # 기본값: 일봉
        self.timeframe_combo.grid(row=0, column=3, padx=5, pady=5)

        # 유사도 측정 방법 선택
        ttk.Label(control_frame, text="유사도 측정:").grid(row=0, column=4, padx=5, pady=5)
        self.similarity_combo = ttk.Combobox(
            control_frame,
            values=[
                "유클리디안 거리",
                "상관계수",
                "Matrix Profile (STUMPY)",
                "형태 기반 거리",
            ],
        )
        self.similarity_combo.current(0)  # 기본값: 유클리디안 거리
        self.similarity_combo.bind("<<ComboboxSelected>>", self.update_similarity_method)
        self.similarity_combo.grid(row=0, column=5, padx=5, pady=5)

        # 고정 기간 표시 레이블
        ttk.Label(control_frame, text="분석 기간: 2017년 11월 1일 ~ 현재").grid(
            row=0, column=6, padx=5, pady=5
        )

        # 분석 버튼
        self.analyze_button = ttk.Button(
            control_frame, text="분석 시작", command=self.start_analysis
        )
        self.analyze_button.grid(row=0, column=7, padx=5, pady=5)

        # GPU 가속 체크박스
        self.gpu_var = tk.BooleanVar(value=self.use_gpu)
        gpu_check = ttk.Checkbutton(
            control_frame, text="GPU 가속 사용", variable=self.gpu_var, command=self.toggle_gpu
        )
        gpu_check.grid(row=0, column=8, padx=5, pady=5)

        # 프로그레스 바
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # 패널 분리 (로그와 그래프 분리)
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 로그 출력 영역
        log_frame = ttk.LabelFrame(paned, text="로그")
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)  # 읽기 전용

        # 그래프 영역 설정
        graph_frame = ttk.LabelFrame(paned, text="패턴 분석 결과")

        # Matplotlib 그림 설정
        self.figure = Figure(figsize=(12, 6), dpi=100)

        # 캔버스 생성
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 네비게이션 툴바 추가 (확대/축소, 저장 기능)
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()

        # 패널에 프레임 추가
        paned.add(log_frame, weight=1)
        paned.add(graph_frame, weight=5)

        # 초기 로그 메시지
        self.log("비트코인 패턴 분석기가 시작되었습니다.")
        self.log(f"현재 유사도 측정 방법: {self.current_analyzer.name}")
        self.log("분석 기간: 2017년 11월 1일 ~ 현재")
        if self.use_gpu:
            self.log("GPU 가속이 활성화되어 있습니다.")

    def log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # 읽기 전용 해제
        self.log_text.config(state=tk.NORMAL)

        # 메시지 추가
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")

        # 자동 스크롤
        self.log_text.see(tk.END)

        # 다시 읽기 전용으로 설정
        self.log_text.config(state=tk.DISABLED)

    def update_similarity_method(self, event=None):
        """유사도 측정 방법 변경 처리"""
        method = self.similarity_combo.get()
        if method == "유클리디안 거리":
            self.current_analyzer = self.euclidean_analyzer
            self.log("유클리디안 거리 방식으로 변경")
        elif method == "상관계수":
            self.current_analyzer = self.correlation_analyzer
            self.log("상관계수 방식으로 변경")
        elif method == "Matrix Profile (STUMPY)":
            self.current_analyzer = self.stumpy_analyzer
            self.log("Matrix Profile (STUMPY) 방식으로 변경")
        elif method == "형태 기반 거리":
            self.current_analyzer = self.shape_analyzer
            self.log("형태 기반 거리 방식으로 변경")

    def toggle_gpu(self):
        """GPU 가속 사용 전환"""
        self.use_gpu = self.gpu_var.get()

        # 모든 분석기 GPU 설정 업데이트
        self.euclidean_analyzer.use_gpu = self.use_gpu
        self.correlation_analyzer.use_gpu = self.use_gpu
        self.stumpy_analyzer.use_gpu = self.use_gpu
        self.shape_analyzer.use_gpu = self.use_gpu

        if self.use_gpu:
            self.log("GPU 가속이 활성화되었습니다")
        else:
            self.log("GPU 가속이 비활성화되었습니다 (CPU 모드)")

    def start_analysis(self):
        """데이터 로드 및 분석 시작"""
        try:
            # 버튼 비활성화
            self.analyze_button.config(state=tk.DISABLED)

            # 프로그레스 바 시작
            self.progress_bar.start()

            # 설정 가져오기
            symbol = self.symbol_combo.get()
            timeframe = self.timeframe_combo.get()

            # 기간 고정: 2017년 11월 1일부터 현재까지
            end_date = datetime.now()
            start_date = datetime(2017, 11, 1)

            # 날짜 문자열로 변환
            end_date_str = end_date.strftime("%Y-%m-%d")
            start_date_str = start_date.strftime("%Y-%m-%d")

            self.log(f"분석 시작: {symbol}, {timeframe}, {start_date_str} ~ {end_date_str}")
            self.log("캐시된 데이터가 있으면 사용하고, 없으면 새로 다운로드합니다.")

            # 분석 스레드 생성 및 시작
            thread = AnalysisThread(
                self.current_analyzer, symbol, timeframe, start_date_str, end_date_str, self.queue
            )
            thread.start()

        except Exception as e:
            self.log(f"오류 발생: {str(e)}")
            self.analyze_button.config(state=tk.NORMAL)
            self.progress_bar.stop()
            traceback.print_exc()

    def process_queue(self):
        """큐 처리 - 워커 스레드로부터 메시지 받기"""
        try:
            while not self.queue.empty():
                message = self.queue.get(0)
                message_type = message[0]

                if message_type == "log":
                    self.log(message[1])
                elif message_type == "error":
                    self.handle_error(message[1])
                elif message_type == "finished":
                    self.analysis_finished(*message[1])
        except Exception as e:
            print(f"큐 처리 오류: {e}")

        # 100ms 후 다시 확인
        self.root.after(100, self.process_queue)

    def handle_error(self, error_message):
        """분석 오류 처리"""
        self.log(f"오류: {error_message}")
        self.analyze_button.config(state=tk.NORMAL)
        self.progress_bar.stop()

        messagebox.showerror("분석 오류", error_message)

    def analysis_finished(self, df, current_pattern, similar_patterns):
        """분석 완료 및 결과 표시"""
        try:
            self.log("분석 완료, 결과 시각화 중...")

            # 그래프 초기화
            self.figure.clear()

            # 결과 플로팅
            if similar_patterns:
                # Matplotlib 그래프를 직접 생성
                self.plot_similar_patterns(
                    df,
                    current_pattern,
                    similar_patterns,
                    self.current_analyzer.window_size,
                    plot_count=5,
                    future_days=30,
                )

                # 캔버스 업데이트
                self.canvas.draw()

                # 결과 로깅 (10개 모두 표시)
                self.log(f"{len(similar_patterns)}개의 유사 패턴을 찾았습니다.")
                self.log("\n현재 패턴과 가장 유사한 과거 패턴들 (상위 10개):")

                # 시간프레임에 따라 날짜 포맷 조정
                timeframe = self.timeframe_combo.get()
                if timeframe == "1d":
                    date_format = "%Y-%m-%d"
                else:
                    date_format = "%Y-%m-%d %H:%M"

                for i, (distance, timestamp, _) in enumerate(similar_patterns):
                    self.log(
                        f"패턴 {i+1}: {timestamp.strftime(date_format)}, 유사도: {distance:.4f}"
                    )
            else:
                self.log("유사한 패턴을 찾지 못했습니다.")

        except Exception as e:
            self.log(f"결과 시각화 오류: {str(e)}")
            traceback.print_exc()

        finally:
            # UI 상태 복원
            self.analyze_button.config(state=tk.NORMAL)
            self.progress_bar.stop()

    def plot_similar_patterns(
        self, df, current_pattern, similar_patterns, window_size, plot_count=5, future_days=30
    ):
        """유사 패턴들을 시각화 (Tkinter 버전)"""
        if not similar_patterns:
            self.log("시각화할 유사 패턴이 없습니다.")
            return

        plot_count = min(plot_count, len(similar_patterns))
        if plot_count == 0:
            self.log("시각화할 충분한 패턴이 없습니다.")
            return

        # Tkinter에서 사용할 수 있는 matplotlib 그래프 생성
        ax = self.figure.add_subplot(111)

        # 현재 시점 구하기 (최근 패턴의 종료 시점)
        latest_date = df.index[-1]
        start_of_current_pattern = df.index[-(window_size):]

        # 현재 패턴의 실제 가격 변화율 계산
        current_actual_prices = df["Close"].iloc[-(window_size):]
        current_normalized = (
            (current_actual_prices - current_actual_prices.iloc[0])
            / current_actual_prices.iloc[0]
            * 100
        )

        # 현재 패턴 플롯 (실제 가격 변화율)
        ax.plot(
            range(window_size),
            current_normalized,
            "b-",
            label=f'현재 패턴 ({start_of_current_pattern[0].strftime("%Y-%m-%d")} ~ {latest_date.strftime("%Y-%m-%d")})',
            linewidth=3,
        )

        # 과거 유사 패턴들 플롯
        colors = ["r", "g", "c", "m", "y"]
        for i, ((similarity, timestamp, _), color) in enumerate(
            zip(similar_patterns[:plot_count], colors)
        ):
            if timestamp in df.index:
                pattern_start_idx = df.index.get_loc(timestamp)
                pattern_end_idx = pattern_start_idx + window_size

                # 패턴 이후 30일까지 데이터가 있는지 확인
                if pattern_end_idx + future_days <= len(df):
                    # 패턴 시작부터 미래까지의 가격 데이터
                    prices = df["Close"].iloc[pattern_start_idx : pattern_end_idx + future_days]
                    normalized_prices = (prices - prices.iloc[0]) / prices.iloc[0] * 100

                    # 패턴 부분 (점선)
                    ax.plot(
                        range(window_size),
                        normalized_prices[:window_size],
                        f"{color}--",
                        label=f'패턴 {i + 1}: {timestamp.strftime("%Y-%m-%d")} (유사도: {similarity:.4f})',
                        alpha=0.7,
                    )

                    # 미래 부분 (실선으로 변경)
                    ax.plot(
                        range(window_size - 1, window_size + future_days),
                        normalized_prices[window_size - 1 :],
                        f"{color}-",  # 같은 색상의 실선
                        alpha=0.7,
                    )

        # 타이틀에 현재 방법론 표시
        method_name = self.current_analyzer.name  # 각 분석기 클래스의 name 속성 사용

        # 예전 코드로 폴백 (만약 name 속성이 없는 경우를 대비)
        if not hasattr(self.current_analyzer, "name") or not self.current_analyzer.name:
            if self.current_analyzer.__class__.__name__ == "EuclideanPatternAnalyzer":
                method_name = "유클리디안 거리"
            elif self.current_analyzer.__class__.__name__ == "CorrelationPatternAnalyzer":
                method_name = "상관계수"
            elif self.current_analyzer.__class__.__name__ == "StumpyPatternAnalyzer":
                method_name = "Matrix Profile (STUMPY)"
            elif self.current_analyzer.__class__.__name__ == "ShapeBasedPatternAnalyzer":
                method_name = "형태 기반 거리"
            else:
                method_name = "알 수 없는 방법"

        ax.set_title(
            f"비트코인: 현재 패턴과 유사 패턴들의 실제 가격 움직임 (%) - {method_name} 기반",
            fontsize=14,
        )
        ax.set_xlabel("패턴 시작으로부터 경과 일수", fontsize=12)
        ax.set_ylabel("가격 변화율 (%)", fontsize=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True)
        ax.axvline(x=window_size, color="k", linestyle="--")

        # 미래 영역에 음영 표시
        ax.axvspan(window_size, window_size + future_days, alpha=0.1, color="gray")

        # 미래 영역 텍스트
        y_min, y_max = ax.get_ylim()
        y_text_pos = y_min + (y_max - y_min) * 0.05
        ax.text(window_size + future_days / 2, y_text_pos, "미래 영역", ha="center", va="bottom")

        self.figure.tight_layout()


if __name__ == "__main__":
    """애플리케이션 실행 진입점"""
    root = tk.Tk()
    app = PatternAnalyzerApp(root)
    root.mainloop()
