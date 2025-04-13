from base_analyzer import BasePatternAnalyzer
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import traceback
import time


class CorrelationPatternAnalyzer(BasePatternAnalyzer):
    def __init__(self, window_size=60, use_gpu=True):
        super().__init__(window_size)
        self.name = "상관계수"
        self.use_gpu = use_gpu

        # GPU 사용 가능 여부 확인
        if self.use_gpu:
            try:
                import cupy
                from cupyx.scipy import stats as cupyx_stats

                self.gpu_available = True
                print("CUDA GPU 가속이 활성화되었습니다 (상관계수)")
            except ImportError:
                self.gpu_available = False
                print("cupy 라이브러리가 설치되지 않아 GPU 가속을 사용할 수 없습니다.")
                print(
                    "GPU 가속을 위해 'pip install cupy-cuda11x'를 실행하세요. (CUDA 버전에 맞게 설치)"
                )
        else:
            self.gpu_available = False

    def calculate_similarity(self, pattern1, pattern2):
        """상관계수를 사용하여 두 패턴 간의 유사도 계산 (GPU 가속 지원)"""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0  # 비교할 수 없는 경우 최소 유사도 반환

        # 표준편차 체크 - 변동이 없는 패턴 감지
        if np.std(pattern1) < 1e-8 or np.std(pattern2) < 1e-8:
            return 0.0  # 변동이 없는 패턴은 유사도 0

        try:
            # GPU 가속 사용
            if self.use_gpu and self.gpu_available:
                try:
                    import cupy as cp

                    # 패턴을 GPU로 복사
                    p1 = cp.array(pattern1)
                    p2 = cp.array(pattern2)

                    # 평균 제거
                    p1_centered = p1 - cp.mean(p1)
                    p2_centered = p2 - cp.mean(p2)

                    # 표준편차로 나누기
                    p1_std = cp.std(p1)
                    p2_std = cp.std(p2)

                    if p1_std == 0 or p2_std == 0:
                        return 0.0  # 변동이 없는 패턴

                    p1_normalized = p1_centered / p1_std
                    p2_normalized = p2_centered / p2_std

                    # 상관계수 계산
                    corr = cp.sum(p1_normalized * p2_normalized) / len(p1)

                    # 거리로 변환 (상관계수가 높을수록 거리는 짧음)
                    # 상관계수 범위 [-1, 1]을 거리 범위 [0, 1]로 변환
                    distance = (1.0 - float(cp.asnumpy(corr))) / 2.0
                    return distance
                except Exception as e:
                    # 오류 발생 시 CPU로 대체
                    corr = np.corrcoef(pattern1, pattern2)[0, 1]
                    return (1.0 - corr) / 2.0
            else:
                # CPU 계산
                corr = np.corrcoef(pattern1, pattern2)[0, 1]
                return (1.0 - corr) / 2.0
        except Exception as e:
            # 계산 불가능한 경우 (예: 모든 값이 같을 때)
            print(f"상관계수 계산 중 오류: {e}")
            return 0.5  # 중간값 반환 (0.5는 상관계수 0에 해당)

    def find_similar_patterns(self, patterns, timestamps, threshold=0.3):
        """상관계수를 사용한 유사한 패턴 클러스터링 (GPU 가속 지원)"""
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

        print("상관관계 거리 행렬 계산 중...")
        start_time = time.time()

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                # 모든 패턴을 GPU로 복사
                patterns_gpu = cp.array(valid_patterns)

                # 평균 계산 및 제거
                means = cp.mean(patterns_gpu, axis=1, keepdims=True)
                centered = patterns_gpu - means

                # 표준편차 계산
                stds = cp.sqrt(cp.sum(centered**2, axis=1, keepdims=True))
                # 0으로 나누기 방지
                stds = cp.where(stds == 0, 1, stds)
                normalized = centered / stds

                # 상관계수 행렬 계산
                similarities = cp.dot(normalized, normalized.T)

                # 거리로 변환 (-1~1 범위의 상관계수를 0~1 범위의 거리로 변환)
                distances = cp.asnumpy((1 - similarities) / 2)

                elapsed = time.time() - start_time
                print(f"GPU 상관계수 행렬 계산 완료: {elapsed:.2f}초")

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체 - 효율적인 행렬 연산 사용
                patterns_array = np.array(valid_patterns)

                # 평균 계산 및 제거
                means = np.mean(patterns_array, axis=1, keepdims=True)
                centered = patterns_array - means

                # 표준편차 계산
                stds = np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
                # 0으로 나누기 방지
                stds = np.where(stds == 0, 1, stds)
                normalized = centered / stds

                # 상관계수 행렬 계산
                similarities = np.dot(normalized, normalized.T)

                # 거리로 변환
                distances = (1 - similarities) / 2

                elapsed = time.time() - start_time
                print(f"CPU 상관계수 행렬 계산 완료: {elapsed:.2f}초")
        else:
            # CPU 방식 - 효율적인 행렬 연산 사용
            patterns_array = np.array(valid_patterns)

            # 평균 계산 및 제거
            means = np.mean(patterns_array, axis=1, keepdims=True)
            centered = patterns_array - means

            # 표준편차 계산
            stds = np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
            # 0으로 나누기 방지
            stds = np.where(stds == 0, 1, stds)
            normalized = centered / stds

            # 상관계수 행렬 계산
            similarities = np.dot(normalized, normalized.T)

            # 거리로 변환
            distances = (1 - similarities) / 2

            elapsed = time.time() - start_time
            print(f"CPU 상관계수 행렬 계산 완료: {elapsed:.2f}초")

        # 숫자 범위 확인 및 클리핑
        distances = np.clip(distances, 0, 1)

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
        """상관계수를 사용하여 현재 패턴과 가장 유사한 과거 패턴 찾기 (GPU 가속 지원)"""
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

        print("상관계수 방식으로 가장 유사한 패턴 검색 중...")
        start_time = time.time()

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                # 현재 패턴을 GPU로 복사
                cp_current = cp.array(current_pattern)
                cp_current_mean = cp.mean(cp_current)
                cp_current_std = cp.std(cp_current)

                if cp_current_std == 0:
                    raise ValueError("현재 패턴의 표준편차가 0입니다.")

                cp_current_norm = (cp_current - cp_current_mean) / cp_current_std

                # 패턴 배열 준비
                patterns_array = np.array(patterns)
                cp_patterns = cp.array(patterns_array)

                # 각 패턴의 평균 및 표준편차 계산
                means = cp.mean(cp_patterns, axis=1, keepdims=True)
                centered = cp_patterns - means
                stds = cp.sqrt(cp.sum(centered**2, axis=1))

                # 유효한 패턴만 선택 (표준편차 > 0)
                valid_indices = cp.where(stds > 1e-8)[0]

                if len(valid_indices) == 0:
                    print("유효한 패턴이 없습니다.")
                    return []

                valid_patterns = cp_patterns[valid_indices]
                valid_timestamps = [timestamps[i] for i in cp.asnumpy(valid_indices)]
                valid_stds = stds[valid_indices].reshape(-1, 1)

                # 정규화된 패턴 계산
                normalized_patterns = (
                    valid_patterns - cp.mean(valid_patterns, axis=1, keepdims=True)
                ) / valid_stds

                # 상관계수 계산
                correlations = cp.dot(normalized_patterns, cp_current_norm) / len(cp_current)

                # 상관계수를 CPU로 가져오기
                correlations_cpu = cp.asnumpy(correlations)

                # 결과 정리
                correlation_results = []
                for i, corr in enumerate(correlations_cpu):
                    # 상관계수를 유사도 점수로 변환 (높을수록 더 유사)
                    similarity = (corr + 1) / 2  # -1~1 범위를 0~1 범위로 변환
                    correlation_results.append(
                        (
                            similarity,
                            valid_timestamps[i],
                            patterns_array[cp.asnumpy(valid_indices)[i]],
                        )
                    )

                # GPU 메모리 해제
                del cp_current, cp_patterns, correlations
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                correlation_results = []

                # 현재 패턴 정규화
                current_mean = np.mean(current_pattern)
                current_std = np.std(current_pattern)

                if current_std < 1e-8:
                    print("현재 패턴의 변동성이 너무 낮습니다.")
                    return []

                current_norm = (current_pattern - current_mean) / current_std

                for i, pattern in enumerate(patterns):
                    # 각 패턴의 표준편차 확인
                    pattern_std = np.std(pattern)

                    if pattern_std >= 1e-8:  # 변동이 있는 패턴만 처리
                        try:
                            # np.corrcoef 사용하여 상관계수 계산
                            corr = np.corrcoef(current_pattern, pattern)[0, 1]

                            # NaN 체크
                            if np.isnan(corr):
                                continue

                            # 상관계수를 유사도 점수로 변환 (높을수록 더 유사)
                            similarity = (corr + 1) / 2  # -1~1 범위를 0~1 범위로 변환
                            correlation_results.append((similarity, timestamps[i], pattern))
                        except Exception as e:
                            print(f"패턴 {i} 유사도 계산 실패: {e}")
                            continue

                if len(correlation_results) == 0:
                    print("유효한 유사 패턴을 찾을 수 없습니다.")
                    return []
        else:
            # CPU 방식
            correlation_results = []

            # 현재 패턴 정규화
            current_mean = np.mean(current_pattern)
            current_std = np.std(current_pattern)

            if current_std < 1e-8:
                print("현재 패턴의 변동성이 너무 낮습니다.")
                return []

            current_norm = (current_pattern - current_mean) / current_std

            # 모든 패턴을 배열로 변환
            patterns_array = np.array(patterns)

            # 각 패턴의 평균 및 표준편차 계산
            means = np.mean(patterns_array, axis=1, keepdims=True)
            centered = patterns_array - means
            stds = np.sqrt(np.sum(centered**2, axis=1)).reshape(-1, 1)

            # 유효한 패턴만 선택 (표준편차 > 0)
            valid_indices = np.where(stds.flatten() > 1e-8)[0]

            if len(valid_indices) == 0:
                print("유효한 패턴이 없습니다.")
                return []

            valid_patterns = patterns_array[valid_indices]
            valid_timestamps = [timestamps[i] for i in valid_indices]
            valid_stds = stds[valid_indices]

            # 정규화된 패턴 계산
            normalized_patterns = centered[valid_indices] / valid_stds

            # 상관계수 계산
            correlations = np.dot(normalized_patterns, current_norm) / len(current_pattern)

            # 결과 정리
            for i, corr in enumerate(correlations):
                # 상관계수를 유사도 점수로 변환 (높을수록 더 유사)
                similarity = (corr + 1) / 2  # -1~1 범위를 0~1 범위로 변환
                correlation_results.append((similarity, valid_timestamps[i], valid_patterns[i]))

        # 유사도(상관계수) 기준으로 정렬 (높은 값이 더 유사)
        sorted_correlations = sorted(correlation_results, key=lambda x: x[0], reverse=True)

        # 월별 중복 제거
        unique_months = {}
        for similarity, timestamp, pattern in sorted_correlations:
            month_key = timestamp.strftime("%Y-%m")  # 년-월 형식으로 키 생성
            if month_key not in unique_months:
                unique_months[month_key] = (similarity, timestamp, pattern)

        # 중복이 제거된 결과를 다시 유사도순으로 정렬
        filtered_results = sorted(unique_months.values(), key=lambda x: x[0], reverse=True)

        # top_n 값이 필터링된 결과보다 크면 조정
        return_count = min(top_n, len(filtered_results))
        print(f"{return_count}개의 유사 패턴을 찾았습니다.")

        elapsed = time.time() - start_time
        print(f"유사 패턴 검색 완료: {elapsed:.2f}초")

        return filtered_results[:return_count]
