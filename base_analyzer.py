import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import ccxt
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class BasePatternAnalyzer(ABC):
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()

        # 한글 폰트 설정
        mpl.rcParams["font.family"] = "Malgun Gothic"
        mpl.rcParams["axes.unicode_minus"] = False

    def load_data(self, symbol="BTC/USDT", timeframe="1d", start_date=None, end_date=None):
        """CCXT를 사용하여 바이낸스에서 비트코인 데이터 로드 (캐싱 기능 추가)"""
        import os

        # 날짜 변환
        if start_date:
            if isinstance(start_date, str):
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_date_obj = start_date
        else:
            start_date_obj = datetime.now() - timedelta(days=365)

        if end_date:
            if isinstance(end_date, str):
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_date_obj = end_date
        else:
            end_date_obj = datetime.now()

        # 유효한 시작일인지 확인
        if start_date_obj > end_date_obj:
            raise ValueError("시작일이 종료일보다 늦을 수 없습니다.")

        # 캐시 디렉터리 생성
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # 파일명 생성 (심볼과 시간프레임 기반)
        symbol_clean = symbol.replace("/", "_")
        cache_filename = (
            f"{cache_dir}/{symbol_clean}_{timeframe}_{start_date_obj.strftime('%Y_%m_%d')}.csv"
        )

        print(f"캐시 파일: {cache_filename}")

        # 현재 시간 구하기 (최신 데이터 갱신 기준)
        now = datetime.now()

        # 시간프레임에 따라 최신 데이터 기준점 설정
        if timeframe == "15m":
            # 현재 시간 기준 하루 이내의 데이터는 갱신
            refresh_from = now - timedelta(days=1)
        elif timeframe == "1h":
            # 현재 시간 기준 하루 이내의 데이터는 갱신
            refresh_from = now - timedelta(days=1)
        elif timeframe == "4h":
            # 현재 시간 기준 이틀 이내의 데이터는 갱신
            refresh_from = now - timedelta(days=2)
        else:  # 1d
            # 현재 시간 기준 최근 3일의 데이터는 갱신 (주말 포함 고려)
            refresh_from = now - timedelta(days=3)

        # 캐시된 데이터 읽기
        cached_df = None
        last_timestamp = None
        refresh_timestamp = int(refresh_from.timestamp() * 1000)

        if os.path.exists(cache_filename):
            try:
                print(f"캐시된 데이터 파일을 읽는 중...")
                cached_df = pd.read_csv(cache_filename)

                # 타임스탬프 열을 datetime으로 변환
                cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"])
                cached_df.set_index("timestamp", inplace=True)

                # 마지막 타임스탬프 확인
                last_timestamp = cached_df.index[-1]
                last_timestamp_ms = int(last_timestamp.timestamp() * 1000)

                print(f"캐시된 데이터의 마지막 날짜: {last_timestamp}")

                # 최신 데이터 갱신을 위해 마지막 시점 이전 데이터만 사용
                # refresh_from 시점 이후 데이터는 항상 새로 가져옴
                if cached_df is not None:
                    # 갱신 시점 이전의 데이터만 유지
                    cached_df = cached_df[cached_df.index < refresh_from]

                    if len(cached_df) > 0:
                        # 남은 데이터의 마지막 타임스탬프 업데이트
                        last_timestamp = cached_df.index[-1]
                        last_timestamp_ms = int(last_timestamp.timestamp() * 1000)
                        print(
                            f"최신 데이터 갱신을 위해 {refresh_from} 이후 데이터는 제외, 마지막 유지 타임스탬프: {last_timestamp}"
                        )
                    else:
                        # 모든 데이터가 갱신 대상이면 처음부터 다시 가져옴
                        cached_df = None
                        last_timestamp = None
                        print("캐시된 모든 데이터가 갱신 대상입니다. 처음부터 다시 가져옵니다.")

            except Exception as e:
                print(f"캐시된 데이터 읽기 실패: {e}. 새로 데이터를 가져옵니다.")
                cached_df = None

        try:
            # 바이낸스 인스턴스 생성
            exchange = ccxt.binance()

            # 새로 가져올 데이터 범위 결정
            if cached_df is not None and last_timestamp is not None:
                print(f"마지막 캐시 이후의 새 데이터를 가져옵니다...")
                current = last_timestamp_ms + 1  # 마지막 타임스탬프 다음부터 시작
            else:
                print(f"전체 기간의 데이터를 가져옵니다...")
                current = int(start_date_obj.timestamp() * 1000)

            end_timestamp = int(end_date_obj.timestamp() * 1000)

            # 새 데이터 가져오기
            ohlcv = []
            while current < end_timestamp:
                print(
                    f"데이터 청크 가져오는 중... {datetime.fromtimestamp(current / 1000).strftime('%Y-%m-%d %H:%M')}"
                )
                chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current, limit=1000)

                if not chunk:
                    break

                ohlcv.extend(chunk)

                # 시간프레임에 따라 다음 청크의 시작점 계산
                if timeframe == "15m":
                    time_increment = 15 * 60 * 1000 * 1000  # 15분 * 1000개
                elif timeframe == "1h":
                    time_increment = 60 * 60 * 1000 * 1000  # 1시간 * 1000개
                elif timeframe == "4h":
                    time_increment = 4 * 60 * 60 * 1000 * 1000  # 4시간 * 1000개
                else:  # 기본값: 1d
                    time_increment = 24 * 60 * 60 * 1000 * 1000  # 1일 * 1000개

                current = chunk[-1][0] + (time_increment // 1000)  # 다음 청크 시작점

                # API 속도 제한 방지
                time.sleep(1)

            # 새로 가져온 데이터가 없으면 캐시된 데이터만 반환
            if not ohlcv and cached_df is not None:
                print("새로운 데이터가 없습니다. 캐시된 데이터를 사용합니다.")
                # 마지막으로 일부분 필터링 및 수익률 계산
                filtered_df = cached_df[
                    (cached_df.index >= pd.to_datetime(start_date_obj))
                    & (cached_df.index <= pd.to_datetime(end_date_obj))
                ]

                if "Returns" not in filtered_df.columns:
                    filtered_df.loc[:, "Returns"] = filtered_df["Close"].pct_change()
                    filtered_df = filtered_df.dropna()

                return filtered_df

            # 새 데이터를 DataFrame으로 변환
            if ohlcv:
                new_df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
                )

                # 타임스탬프를 날짜 인덱스로 변환
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
                new_df.set_index("timestamp", inplace=True)

                # 중복 제거 및 정렬
                new_df = new_df[~new_df.index.duplicated(keep="first")]
                new_df.sort_index(inplace=True)

                # 캐시된 데이터와 합치기
                if cached_df is not None:
                    df = pd.concat([cached_df, new_df])
                    # 중복 제거
                    df = df[~df.index.duplicated(keep="last")]
                    df.sort_index(inplace=True)
                else:
                    df = new_df

                # 전체 데이터 캐시에 저장
                print(f"최신 데이터를 캐시 파일에 저장 중...")
                df_to_save = df.copy()
                df_to_save.reset_index(inplace=True)  # 인덱스를 컬럼으로 변환하여 저장
                df_to_save.to_csv(cache_filename, index=False)
                print(f"캐시 파일 저장 완료: {cache_filename}")
            else:
                df = cached_df

            # 날짜 범위 필터링
            if start_date_obj and end_date_obj:
                df = df[
                    (df.index >= pd.to_datetime(start_date_obj))
                    & (df.index <= pd.to_datetime(end_date_obj))
                ]

            # 수익률 계산
            if "Returns" not in df.columns:
                df.loc[:, "Returns"] = df["Close"].pct_change()
                df = df.dropna()

            print(f"데이터 로드 완료: {len(df)} 행의 데이터")
            return df

        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            traceback.print_exc()

            # 오류 발생 시 캐시된 데이터라도 반환 시도
            if cached_df is not None:
                print("오류 발생으로 캐시된 데이터만 반환합니다.")
                # 날짜 범위 필터링
                filtered_df = cached_df[
                    (cached_df.index >= pd.to_datetime(start_date_obj))
                    & (cached_df.index <= pd.to_datetime(end_date_obj))
                ]

                if "Returns" not in filtered_df.columns:
                    filtered_df.loc[:, "Returns"] = filtered_df["Close"].pct_change()
                    filtered_df = filtered_df.dropna()

                return filtered_df

            raise

    def normalize_window(self, window):
        """각 윈도우 구간 정규화"""
        if len(window) == 0:
            return np.array([])
        return self.scaler.fit_transform(window.reshape(-1, 1)).flatten()

    def extract_patterns(self, df, target_col="Close"):
        """주어진 기간에서 패턴 추출"""
        if df.empty or len(df) < self.window_size:
            print(
                f"경고: 데이터가 부족하여 패턴을 추출할 수 없습니다 (필요: {self.window_size}, 실제: {len(df)})"
            )
            return np.array([]), []

        patterns = []
        timestamps = []

        for i in range(len(df) - self.window_size + 1):
            window = df[target_col].iloc[i : i + self.window_size].values
            normalized_window = self.normalize_window(window)
            patterns.append(normalized_window)
            timestamps.append(df.index[i])

        print(f"추출된 패턴 수: {len(patterns)}")
        return np.array(patterns), timestamps

    @abstractmethod
    def calculate_similarity(self, pattern1, pattern2):
        """두 패턴 간의 유사도 계산 - 하위 클래스에서 구현"""
        pass

    @abstractmethod
    def find_similar_patterns(self, patterns, timestamps, threshold):
        """유사한 패턴 클러스터링 - 하위 클래스에서 구현"""
        pass

    def analyze_future_movements(self, df, pattern_groups):
        """각 패턴 그룹 이후의 가격 움직임 분석"""
        future_movements = {}

        if not pattern_groups:
            print("분석할 패턴 그룹이 없습니다.")
            return future_movements

        for cluster, patterns in pattern_groups.items():
            movements = []
            dates = []
            for timestamp, _ in patterns:
                try:
                    if timestamp in df.index:
                        start_idx = df.index.get_loc(timestamp) + self.window_size
                        if start_idx < len(df) and start_idx + 30 < len(df):
                            future_return = (
                                df["Close"].iloc[start_idx + 30] / df["Close"].iloc[start_idx] - 1
                            ) * 100
                            movements.append(future_return)
                            dates.append(timestamp)
                    else:
                        print(f"타임스탬프 {timestamp}가 데이터프레임 인덱스에 없습니다.")
                except Exception as e:
                    print(f"Warning: Error processing timestamp {timestamp}: {e}")
                    continue

            if movements:
                future_movements[cluster] = {
                    "mean_return": np.mean(movements),
                    "std_return": np.std(movements),
                    "positive_prob": np.mean(np.array(movements) > 0),
                    "count": len(movements),
                    "dates": dates,
                    "returns": movements,
                }
            else:
                print(f"클러스터 {cluster}에서 미래 움직임을 분석할 수 없습니다.")

        return future_movements

    @abstractmethod
    def find_most_similar_historical_pattern(self, current_pattern, patterns, timestamps, top_n=10):
        """현재 패턴과 가장 유사한 과거 패턴 찾기 - 하위 클래스에서 구현"""
        pass

    def plot_similar_patterns(
        self, df, current_pattern, similar_patterns, window_size, plot_count=5, future_days=30
    ):
        """유사 패턴들을 시각화 (두 번째 차트만 표시, 한글 폰트 지원)"""
        if not similar_patterns:
            print("시각화할 유사 패턴이 없습니다.")
            return

        plot_count = min(plot_count, len(similar_patterns))
        if plot_count == 0:
            print("시각화할 충분한 패턴이 없습니다.")
            return

        # 두 번째 차트만 표시
        plt.figure(figsize=(15, 8))

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
        plt.plot(
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
                    plt.plot(
                        range(window_size),
                        normalized_prices[:window_size],
                        f"{color}--",
                        label=f'패턴 {i + 1}: {timestamp.strftime("%Y-%m-%d")} (유사도: {similarity:.4f})',
                        alpha=0.7,
                    )

                    # 미래 부분 (실선으로 변경)
                    plt.plot(
                        range(window_size - 1, window_size + future_days),
                        normalized_prices[window_size - 1 :],
                        f"{color}-",  # 같은 색상의 실선
                        alpha=0.7,
                    )

        # 타이틀에 현재 방법론 표시
        method_name = (
            "유클리디안 거리"
            if self.__class__.__name__ == "EuclideanPatternAnalyzer"
            else "상관계수"
        )
        plt.title(
            f"비트코인: 현재 패턴과 유사 패턴들의 실제 가격 움직임 (%) - {method_name} 기반",
            fontsize=14,
        )
        plt.xlabel("패턴 시작으로부터 경과 일수", fontsize=12)
        plt.ylabel("가격 변화율 (%)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.axvline(x=window_size, color="k", linestyle="--")

        # 미래 영역에 음영 표시
        plt.axvspan(window_size, window_size + future_days, alpha=0.1, color="gray")

        # 미래 영역 텍스트
        y_min, y_max = plt.ylim()
        y_text_pos = y_min + (y_max - y_min) * 0.05
        plt.text(window_size + future_days / 2, y_text_pos, "미래 영역", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
