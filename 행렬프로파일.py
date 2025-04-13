import numpy as np
import pandas as pd
from base_analyzer import BasePatternAnalyzer
from sklearn.cluster import DBSCAN
import traceback
import time

# STUMPY 라이브러리 임포트
import stumpy


class StumpyPatternAnalyzer(BasePatternAnalyzer):
    """Matrix Profile 기반의 패턴 분석기 (CUDA 가속 지원)"""

    def __init__(self, window_size=60, use_gpu=True):
        super().__init__(window_size)
        self.name = "Matrix Profile (STUMPY)"
        self.matrix_profile = None
        self.mp_indices = None
        self.use_gpu = use_gpu

        # GPU 사용 가능 여부 확인
        if self.use_gpu:
            try:
                import cupy

                self.gpu_available = True
                print("CUDA GPU 가속이 활성화되었습니다 (STUMPY)")
            except ImportError:
                self.gpu_available = False
                print("cupy 라이브러리가 설치되지 않아 GPU 가속을 사용할 수 없습니다.")
                print(
                    "GPU 가속을 위해 'pip install cupy-cuda11x'를 실행하세요. (CUDA 버전에 맞게 설치)"
                )

    def extract_patterns(self, df, target_col="Close"):
        """패턴 추출 및 Matrix Profile 계산 (GPU 가속 지원)"""
        if df.empty or len(df) < self.window_size:
            print(
                f"경고: 데이터가 부족하여 패턴을 추출할 수 없습니다 (필요: {self.window_size}, 실제: {len(df)})"
            )
            return np.array([]), []

        # 원본 데이터 저장
        self.original_series = df[target_col].values

        # 패턴 및 타임스탬프 추출
        patterns = []
        timestamps = []

        for i in range(len(df) - self.window_size + 1):
            window = df[target_col].iloc[i : i + self.window_size].values
            normalized_window = self.normalize_window(window)
            patterns.append(normalized_window)
            timestamps.append(df.index[i])

        # Matrix Profile 계산 (GPU 또는 CPU)
        try:
            print("Matrix Profile 계산 중...")
            start_time = time.time()

            if self.use_gpu and self.gpu_available:
                # GPU 가속 사용
                print("CUDA GPU 가속으로 Matrix Profile 계산 중...")
                self.matrix_profile = stumpy.gpu_stump(self.original_series, self.window_size)
            else:
                # CPU 계산
                print("CPU로 Matrix Profile 계산 중...")
                self.matrix_profile = stumpy.stump(self.original_series, self.window_size)

            self.mp_indices = self.matrix_profile[:, 1].astype(int)  # 가장 가까운 이웃의 인덱스

            elapsed = time.time() - start_time
            print(
                f"Matrix Profile 계산 완료. 소요 시간: {elapsed:.2f}초, 길이: {len(self.matrix_profile)}"
            )
        except Exception as e:
            print(f"Matrix Profile 계산 중 오류 발생: {e}")
            traceback.print_exc()

        print(f"추출된 패턴 수: {len(patterns)}")
        return np.array(patterns), timestamps

    def calculate_similarity(self, pattern1, pattern2):
        """Matrix Profile 거리를 기준으로 두 패턴 간의 유사도 계산"""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return float("inf")  # 비교할 수 없는 경우 무한대 거리 반환

        try:
            # GPU 가속을 사용할 수 있으면 cupy로 계산
            if self.use_gpu and self.gpu_available:
                try:
                    import cupy as cp

                    p1 = cp.array(pattern1)
                    p2 = cp.array(pattern2)
                    distance = cp.sqrt(cp.sum((p1 - p2) ** 2))
                    return float(cp.asnumpy(distance))
                except:
                    # GPU 계산 실패 시 CPU로 대체
                    distance = np.sqrt(np.sum((pattern1 - pattern2) ** 2))
                    return distance
            else:
                # CPU 계산
                distance = np.sqrt(np.sum((pattern1 - pattern2) ** 2))
                return distance
        except Exception as e:
            print(f"유사도 계산 중 오류 발생: {e}")
            return float("inf")

    def find_similar_patterns(self, patterns, timestamps, threshold=2.0):
        """Matrix Profile을 활용한 유사한 패턴 클러스터링 (GPU 가속 지원)"""
        n_patterns = len(patterns)
        print(f"전체 패턴 수: {n_patterns}")

        # 패턴이 없는 경우 빈 결과 반환
        if n_patterns == 0 or self.matrix_profile is None:
            print("경고: 패턴이 없거나 Matrix Profile이 계산되지 않았습니다.")
            return {}, np.array([])

        # 패턴이 1개인 경우 특수 처리
        if n_patterns == 1:
            print("경고: 패턴이 하나밖에 없어 클러스터링을 건너뜁니다.")
            pattern_groups = {0: [(timestamps[0], patterns[0])]}
            return pattern_groups, np.array([[0]])

        # Matrix Profile 거리를 사용하여 거리 행렬 생성
        distances = np.zeros((n_patterns, n_patterns))

        print("STUMPY 거리 행렬 계산 중...")
        start_time = time.time()

        # GPU 가속 사용 가능 확인
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                # 청크 단위로 처리하여 메모리 효율화
                chunk_size = min(500, n_patterns)

                for i in range(0, n_patterns, chunk_size):
                    end_i = min(i + chunk_size, n_patterns)
                    patterns_i = cp.array(patterns[i:end_i])

                    for j in range(0, n_patterns, chunk_size):
                        end_j = min(j + chunk_size, n_patterns)
                        patterns_j = cp.array(patterns[j:end_j])

                        # GPU에서 거리 계산
                        for ii in range(end_i - i):
                            for jj in range(end_j - j):
                                if i + ii <= j + jj:  # 대칭성 활용
                                    dist = cp.sqrt(cp.sum((patterns_i[ii] - patterns_j[jj]) ** 2))
                                    distances[i + ii, j + jj] = float(cp.asnumpy(dist))
                                    distances[j + jj, i + ii] = distances[i + ii, j + jj]  # 대칭성

                    # 진행 상황 보고
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (GPU): {end_i}/{n_patterns} 패턴 처리됨 ({end_i/n_patterns*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

                    # GPU 메모리 관리
                    del patterns_i
                    cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                for i in range(n_patterns):
                    for j in range(i, n_patterns):  # 대칭성 활용
                        distances[i, j] = self.calculate_similarity(patterns[i], patterns[j])
                        distances[j, i] = distances[i, j]  # 대칭 값 복사

                    if i % 100 == 0 or i == n_patterns - 1:
                        elapsed = time.time() - start_time
                        print(
                            f"진행 중 (CPU): {i+1}/{n_patterns} 패턴 처리됨 ({(i+1)/n_patterns*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                        )
        else:
            # CPU 버전
            for i in range(n_patterns):
                for j in range(i, n_patterns):  # 대칭성 활용
                    distances[i, j] = self.calculate_similarity(patterns[i], patterns[j])
                    distances[j, i] = distances[i, j]  # 대칭 값 복사

                if i % 100 == 0 or i == n_patterns - 1:
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (CPU): {i+1}/{n_patterns} 패턴 처리됨 ({(i+1)/n_patterns*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

        # DBSCAN 클러스터링으로 유사 패턴 그룹화
        print("클러스터링 시작...")
        try:
            clustering = DBSCAN(eps=threshold, min_samples=2, metric="precomputed")
            clusters = clustering.fit_predict(distances)

            # 클러스터별로 패턴 그룹화
            pattern_groups = {}
            for i, cluster in enumerate(clusters):
                if cluster != -1:  # 노이즈 제외
                    if cluster not in pattern_groups:
                        pattern_groups[cluster] = []
                    pattern_groups[cluster].append((timestamps[i], patterns[i]))

            print(f"발견된 클러스터 수: {len(pattern_groups)}")
            return pattern_groups, distances

        except Exception as e:
            print(f"클러스터링 오류: {e}")
            traceback.print_exc()
            return {}, distances

    def find_most_similar_historical_pattern(self, current_pattern, patterns, timestamps, top_n=10):
        """Matrix Profile 기반으로 현재 패턴과 가장 유사한 과거 패턴 찾기 (GPU 가속 지원)"""
        if len(patterns) == 0:
            print("비교할 과거 패턴이 없습니다.")
            return []

        if len(current_pattern) == 0:
            print("현재 패턴이 유효하지 않습니다.")
            return []

        # Matrix Profile 접근법: 현재 패턴과 모든 과거 패턴 간의 거리 계산
        print("STUMPY 방식으로 가장 유사한 패턴 검색 중...")
        start_time = time.time()

        # GPU 가속 사용 가능 확인
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                # 현재 패턴을 GPU로 복사
                cp_current = cp.array(current_pattern)

                # 청크 단위로 처리
                chunk_size = 500
                distances = []

                for i in range(0, len(patterns), chunk_size):
                    end_i = min(i + chunk_size, len(patterns))
                    # 패턴 청크를 GPU로 복사
                    cp_patterns = cp.array(patterns[i:end_i])

                    # GPU에서 유사도 계산
                    for j in range(end_i - i):
                        dist = cp.sqrt(cp.sum((cp_current - cp_patterns[j]) ** 2))
                        distances.append(
                            (float(cp.asnumpy(dist)), timestamps[i + j], patterns[i + j])
                        )

                    # GPU 메모리 해제
                    del cp_patterns
                    cp.get_default_memory_pool().free_all_blocks()

                    # 진행 상황 보고
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (GPU): {end_i}/{len(patterns)} 패턴 처리됨 ({end_i/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                distances = []
                for i, pattern in enumerate(patterns):
                    # 각 패턴과 현재 패턴 간의 거리 계산
                    distance = self.calculate_similarity(current_pattern, pattern)
                    distances.append((distance, timestamps[i], pattern))

                    if i % 500 == 0 or i == len(patterns) - 1:
                        elapsed = time.time() - start_time
                        print(
                            f"진행 중 (CPU): {i+1}/{len(patterns)} 패턴 처리됨 ({(i+1)/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                        )
        else:
            # CPU 버전
            distances = []
            for i, pattern in enumerate(patterns):
                # 각 패턴과 현재 패턴 간의 거리 계산
                distance = self.calculate_similarity(current_pattern, pattern)
                distances.append((distance, timestamps[i], pattern))

                if i % 500 == 0 or i == len(patterns) - 1:
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (CPU): {i+1}/{len(patterns)} 패턴 처리됨 ({(i+1)/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

        # 거리 기준으로 정렬
        sorted_distances = sorted(distances, key=lambda x: x[0])

        # 월별 중복 제거
        unique_months = {}
        for distance, timestamp, pattern in sorted_distances:
            month_key = timestamp.strftime("%Y-%m")  # 년-월 형식으로 키 생성
            if month_key not in unique_months:
                unique_months[month_key] = (distance, timestamp, pattern)

        # 중복이 제거된 결과를 다시 거리순으로 정렬
        filtered_results = sorted(unique_months.values(), key=lambda x: x[0])

        # top_n 값이 필터링된 결과보다 크면 조정
        return_count = min(top_n, len(filtered_results))
        print(f"{return_count}개의 유사 패턴을 찾았습니다.")
        return filtered_results[:return_count]
