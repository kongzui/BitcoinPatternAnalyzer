from base_analyzer import BasePatternAnalyzer
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import traceback
import time
from scipy.fftpack import fft, ifft


class ShapeBasedPatternAnalyzer(BasePatternAnalyzer):
    """형태 기반 거리(Shape-Based Distance) 패턴 분석기 (GPU 가속 지원)"""

    def __init__(self, window_size=60, use_gpu=True):
        super().__init__(window_size)
        self.name = "형태 기반 거리"
        self.use_gpu = use_gpu

        # GPU 사용 가능 여부 확인
        if self.use_gpu:
            try:
                import cupy

                self.gpu_available = True
                print("CUDA GPU 가속이 활성화되었습니다 (형태 기반 거리)")
            except ImportError:
                self.gpu_available = False
                print("cupy 라이브러리가 설치되지 않아 GPU 가속을 사용할 수 없습니다.")
                print(
                    "GPU 가속을 위해 'pip install cupy-cuda11x'를 실행하세요. (CUDA 버전에 맞게 설치)"
                )
        else:
            self.gpu_available = False

    def calculate_similarity(self, pattern1, pattern2):
        """형태 기반 거리를 사용하여 두 패턴 간의 유사도 계산 (GPU 가속 지원)"""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return float("inf")  # 비교할 수 없는 경우 무한대 거리 반환

        # 표준편차 체크 - 변동이 없는 패턴 감지
        if np.std(pattern1) < 1e-8 or np.std(pattern2) < 1e-8:
            return float("inf")  # 변동이 없는 패턴은 유사도 낮음

        try:
            # GPU 가속 사용
            if self.use_gpu and self.gpu_available:
                try:
                    import cupy as cp
                    from cupyx.scipy import fftpack

                    # 패턴을 GPU로 복사
                    p1 = cp.array(pattern1)
                    p2 = cp.array(pattern2)

                    # z-정규화 (평균 0, 표준편차 1)
                    p1_norm = (p1 - cp.mean(p1)) / (cp.std(p1) + 1e-8)
                    p2_norm = (p2 - cp.mean(p2)) / (cp.std(p2) + 1e-8)

                    # FFT를 이용한 크로스 상관관계 계산
                    fft_p1 = fftpack.fft(p1_norm)
                    fft_p2 = fftpack.fft(p2_norm)
                    fft_p2_conj = cp.conj(fft_p2)
                    ccr = cp.abs(fftpack.ifft(fft_p1 * fft_p2_conj))

                    # 최대 상관관계 찾기
                    max_ccr = cp.max(ccr)

                    # 형태 기반 거리 계산 (1 - 최대 상관관계)
                    # 즉, 최대 상관관계가 1일 때 거리는 0
                    distance = float(1.0 - cp.asnumpy(max_ccr))
                    return distance
                except Exception as e:
                    # GPU 계산 실패 시 CPU로 대체
                    return self._cpu_calculate_similarity(pattern1, pattern2)
            else:
                # CPU 계산
                return self._cpu_calculate_similarity(pattern1, pattern2)
        except Exception as e:
            print(f"형태 기반 거리 계산 중 오류 발생: {e}")
            return float("inf")

    def _cpu_calculate_similarity(self, pattern1, pattern2):
        """CPU에서 형태 기반 거리 계산"""
        from scipy.fftpack import fft, ifft

        # z-정규화 (평균 0, 표준편차 1)
        p1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-8)
        p2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-8)

        # FFT를 이용한 크로스 상관관계 계산
        fft_p1 = fft(p1_norm)
        fft_p2 = fft(p2_norm)
        fft_p2_conj = np.conj(fft_p2)
        ccr = np.abs(ifft(fft_p1 * fft_p2_conj))

        # 최대 상관관계 찾기
        max_ccr = np.max(ccr)

        # 형태 기반 거리 계산 (1 - 최대 상관관계)
        distance = 1.0 - max_ccr
        return distance

    def find_similar_patterns(self, patterns, timestamps, threshold=0.3):
        """형태 기반 거리를 사용한 유사한 패턴 클러스터링 (GPU 가속 지원)"""
        n_patterns = len(patterns)
        print(f"전체 패턴 수: {n_patterns}")

        # 패턴이 없는 경우 빈 결과 반환
        if n_patterns == 0:
            print("경고: 패턴이 없어 유사성 분석을 건너뜁니다.")
            return {}, np.array([])

        # 패턴이 1개인 경우 특수 처리
        if n_patterns == 1:
            print("경고: 패턴이 하나밖에 없어 클러스터링을 건너뜁니다.")
            pattern_groups = {0: [(timestamps[0], patterns[0])]}
            return pattern_groups, np.array([[0]])

        # 표준편차가 0인 패턴 필터링
        valid_indices = []
        valid_patterns = []
        valid_timestamps = []

        for i, pattern in enumerate(patterns):
            if np.std(pattern) >= 1e-8:  # 변동이 있는 패턴만 선택
                valid_indices.append(i)
                valid_patterns.append(pattern)
                valid_timestamps.append(timestamps[i])

        if len(valid_patterns) < 2:
            print("경고: 유효한 패턴이 2개 미만입니다. 클러스터링을 건너뜁니다.")
            pattern_groups = {}
            if len(valid_patterns) == 1:
                pattern_groups[0] = [(valid_timestamps[0], valid_patterns[0])]
            return pattern_groups, np.array([])

        print(f"유효한 패턴 수: {len(valid_patterns)} / {n_patterns}")

        # 거리 행렬을 저장할 NumPy 배열
        distances = np.zeros((len(valid_patterns), len(valid_patterns)))

        print("형태 기반 거리 행렬 계산 중...")
        start_time = time.time()

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp
                from cupyx.scipy import fftpack

                # 청크 단위로 처리하여 메모리 효율화
                chunk_size = min(100, len(valid_patterns))

                # z-정규화된 패턴들 미리 계산
                normalized_patterns = []
                for pattern in valid_patterns:
                    norm_pattern = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
                    normalized_patterns.append(norm_pattern)

                for i in range(0, len(valid_patterns), chunk_size):
                    end_i = min(i + chunk_size, len(valid_patterns))

                    # 현재 청크의 패턴들을 GPU로 복사
                    patterns_i = [cp.array(normalized_patterns[idx]) for idx in range(i, end_i)]
                    ffts_i = [fftpack.fft(p) for p in patterns_i]

                    for j in range(i, len(valid_patterns), chunk_size):
                        end_j = min(j + chunk_size, len(valid_patterns))

                        # 비교할 청크의 패턴들을 GPU로 복사
                        patterns_j = [cp.array(normalized_patterns[idx]) for idx in range(j, end_j)]
                        ffts_j = [fftpack.fft(p) for p in patterns_j]

                        # 각 패턴 쌍에 대해 형태 기반 거리 계산
                        for ii in range(len(patterns_i)):
                            for jj in range(len(patterns_j)):
                                if i + ii <= j + jj:  # 대칭성 활용
                                    # 크로스 상관관계 계산
                                    fft_i = ffts_i[ii]
                                    fft_j_conj = cp.conj(ffts_j[jj])
                                    ccr = cp.abs(fftpack.ifft(fft_i * fft_j_conj))

                                    # 최대 상관관계 찾기
                                    max_ccr = cp.max(ccr)

                                    # 형태 기반 거리 계산
                                    distance = float(1.0 - cp.asnumpy(max_ccr))
                                    distances[i + ii, j + jj] = distance
                                    distances[j + jj, i + ii] = distance  # 대칭성

                    # 진행 상황 보고
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (GPU): {end_i}/{len(valid_patterns)} 패턴 처리됨 ({end_i/len(valid_patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                for i in range(len(valid_patterns)):
                    for j in range(i, len(valid_patterns)):  # 대칭성 활용
                        distances[i, j] = self.calculate_similarity(
                            valid_patterns[i], valid_patterns[j]
                        )
                        distances[j, i] = distances[i, j]  # 대칭 값 복사

                    if i % 50 == 0 or i == len(valid_patterns) - 1:
                        elapsed = time.time() - start_time
                        print(
                            f"진행 중 (CPU): {i+1}/{len(valid_patterns)} 패턴 처리됨 ({(i+1)/len(valid_patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                        )
        else:
            # CPU 방식
            for i in range(len(valid_patterns)):
                for j in range(i, len(valid_patterns)):  # 대칭성 활용
                    distances[i, j] = self.calculate_similarity(
                        valid_patterns[i], valid_patterns[j]
                    )
                    distances[j, i] = distances[i, j]  # 대칭 값 복사

                if i % 50 == 0 or i == len(valid_patterns) - 1:
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (CPU): {i+1}/{len(valid_patterns)} 패턴 처리됨 ({(i+1)/len(valid_patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

        # DBSCAN 클러스터링
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
                    pattern_groups[cluster].append((valid_timestamps[i], valid_patterns[i]))

            print(f"발견된 클러스터 수: {len(pattern_groups)}")
            return pattern_groups, distances

        except Exception as e:
            print(f"클러스터링 오류: {e}")
            traceback.print_exc()
            return {}, distances

    def find_most_similar_historical_pattern(self, current_pattern, patterns, timestamps, top_n=10):
        """형태 기반 거리를 사용하여 현재 패턴과 가장 유사한 과거 패턴 찾기 (GPU 가속 지원)"""
        if len(patterns) == 0:
            print("비교할 과거 패턴이 없습니다.")
            return []

        if len(current_pattern) == 0:
            print("현재 패턴이 유효하지 않습니다.")
            return []

        # 현재 패턴의 표준편차 확인
        if np.std(current_pattern) < 1e-8:
            print("현재 패턴의 변동성이 너무 낮아 유사도 계산이 어렵습니다.")
            return []

        print("형태 기반 거리 방식으로 가장 유사한 패턴 검색 중...")
        start_time = time.time()

        # z-정규화된 현재 패턴
        current_norm = (current_pattern - np.mean(current_pattern)) / (
            np.std(current_pattern) + 1e-8
        )

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp
                from cupyx.scipy import fftpack

                # 현재 패턴을 GPU로 복사
                cp_current_norm = cp.array(current_norm)
                fft_current = fftpack.fft(cp_current_norm)

                # 청크 단위로 처리
                chunk_size = 200
                distances = []

                for i in range(0, len(patterns), chunk_size):
                    end_i = min(i + chunk_size, len(patterns))

                    # 패턴들을 z-정규화하고 FFT 계산
                    for j in range(i, end_i):
                        pattern = patterns[j]

                        # 표준편차 체크
                        if np.std(pattern) < 1e-8:
                            distances.append((float("inf"), timestamps[j], pattern))
                            continue

                        # z-정규화 및 FFT
                        pattern_norm = (pattern - np.mean(pattern)) / (np.std(pattern) + 1e-8)
                        cp_pattern_norm = cp.array(pattern_norm)
                        fft_pattern = fftpack.fft(cp_pattern_norm)

                        # 크로스 상관관계 계산
                        fft_pattern_conj = cp.conj(fft_pattern)
                        ccr = cp.abs(fftpack.ifft(fft_current * fft_pattern_conj))

                        # 최대 상관관계 및 거리 계산
                        max_ccr = cp.max(ccr)
                        distance = float(1.0 - cp.asnumpy(max_ccr))

                        distances.append((distance, timestamps[j], pattern))

                    # 진행 상황 보고
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (GPU): {end_i}/{len(patterns)} 패턴 처리됨 ({end_i/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

                    # GPU 메모리 관리
                    cp.get_default_memory_pool().free_all_blocks()

                # GPU 메모리 해제
                del cp_current_norm, fft_current
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                distances = []
                for i, pattern in enumerate(patterns):
                    distance = self.calculate_similarity(current_pattern, pattern)
                    distances.append((distance, timestamps[i], pattern))

                    if i % 200 == 0 or i == len(patterns) - 1:
                        elapsed = time.time() - start_time
                        print(
                            f"진행 중 (CPU): {i+1}/{len(patterns)} 패턴 처리됨 ({(i+1)/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                        )
        else:
            # CPU 방식
            distances = []
            for i, pattern in enumerate(patterns):
                distance = self.calculate_similarity(current_pattern, pattern)
                distances.append((distance, timestamps[i], pattern))

                if i % 200 == 0 or i == len(patterns) - 1:
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (CPU): {i+1}/{len(patterns)} 패턴 처리됨 ({(i+1)/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

        # 거리 기준으로 정렬 (거리가 작을수록 유사)
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

        elapsed = time.time() - start_time
        print(f"유사 패턴 검색 완료: {elapsed:.2f}초")

        return filtered_results[:return_count]
