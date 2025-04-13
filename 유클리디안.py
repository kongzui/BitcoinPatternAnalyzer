from base_analyzer import BasePatternAnalyzer
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import traceback
import time


class EuclideanPatternAnalyzer(BasePatternAnalyzer):
    def __init__(self, window_size=60, use_gpu=True):
        super().__init__(window_size)
        self.name = "유클리디안 거리"
        self.use_gpu = use_gpu

        # GPU 사용 가능 여부 확인
        if self.use_gpu:
            try:
                import cupy

                self.gpu_available = True
                print("CUDA GPU 가속이 활성화되었습니다 (유클리디안 거리)")
            except ImportError:
                self.gpu_available = False
                print("cupy 라이브러리가 설치되지 않아 GPU 가속을 사용할 수 없습니다.")
                print(
                    "GPU 가속을 위해 'pip install cupy-cuda11x'를 실행하세요. (CUDA 버전에 맞게 설치)"
                )
        else:
            self.gpu_available = False

    def calculate_similarity(self, pattern1, pattern2):
        """유클리디안 거리를 사용하여 두 패턴 간의 유사도 계산 (GPU 가속 지원)"""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return float("inf")  # 비교할 수 없는 경우 무한대 거리 반환

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                p1 = cp.array(pattern1)
                p2 = cp.array(pattern2)
                distance = cp.sqrt(cp.sum((p1 - p2) ** 2))
                return float(cp.asnumpy(distance))
            except Exception as e:
                # GPU 계산 실패 시 CPU로 대체
                return np.sqrt(np.sum((pattern1 - pattern2) ** 2))
        else:
            # CPU 계산
            return np.sqrt(np.sum((pattern1 - pattern2) ** 2))

    def find_similar_patterns(self, patterns, timestamps, threshold=1.0):
        """유사한 패턴 클러스터링 (CUDA 가속 지원)"""
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

        # 거리 행렬을 저장할 NumPy 배열
        distances = np.zeros((n_patterns, n_patterns))

        # 청크 크기 설정 (메모리 사용량 조절)
        chunk_size = min(500, n_patterns)

        print("거리 행렬 계산 중...")
        start_time = time.time()

        # GPU 가속 사용
        if self.use_gpu and self.gpu_available:
            try:
                import cupy as cp

                # 청크 단위로 처리하여 메모리 효율화
                for i in range(0, n_patterns, chunk_size):
                    end_i = min(i + chunk_size, n_patterns)
                    patterns_i = cp.array(patterns[i:end_i])

                    for j in range(0, n_patterns, chunk_size):
                        end_j = min(j + chunk_size, n_patterns)
                        patterns_j = cp.array(patterns[j:end_j])

                        # 현재 청크에 대한 거리 계산
                        chunk_i_expanded = patterns_i[:, cp.newaxis, :]
                        chunk_distances = cp.sqrt(
                            cp.sum((chunk_i_expanded - patterns_j) ** 2, axis=2)
                        )

                        # CPU로 결과 전송
                        distances[i:end_i, j:end_j] = cp.asnumpy(chunk_distances)

                        # 메모리 명시적 해제
                        del chunk_distances
                        cp.get_default_memory_pool().free_all_blocks()

                    # 진행 상황 보고
                    if i % (chunk_size * 2) == 0 or i + chunk_size >= n_patterns:
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
                    for j in range(n_patterns):
                        if i <= j:  # 대칭성 활용하여 계산량 절반으로 감소
                            distances[i, j] = self.calculate_similarity(patterns[i], patterns[j])
                            distances[j, i] = distances[i, j]  # 대칭 값 복사

                    if i % 100 == 0 or i == n_patterns - 1:
                        elapsed = time.time() - start_time
                        print(
                            f"진행 중 (CPU): {i+1}/{n_patterns} 패턴 처리됨 ({(i+1)/n_patterns*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                        )
        else:
            # CPU 버전 (GPU가 없는 경우)
            for i in range(n_patterns):
                for j in range(n_patterns):
                    if i <= j:  # 대칭성 활용하여 계산량 절반으로 감소
                        distances[i, j] = self.calculate_similarity(patterns[i], patterns[j])
                        distances[j, i] = distances[i, j]  # 대칭 값 복사

                if i % 100 == 0 or i == n_patterns - 1:
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (CPU): {i+1}/{n_patterns} 패턴 처리됨 ({(i+1)/n_patterns*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

        # DBSCAN 클러스터링
        print("클러스터링 시작...")
        try:
            # 거리 행렬이 비어있는지 확인
            if distances.size == 0 or np.all(distances == 0):
                print("경고: 거리 행렬이 비어있거나 모든 값이 0입니다.")
                return {}, distances

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
        """현재 패턴과 가장 유사한 과거 패턴 찾기 (GPU 가속 지원)"""
        if len(patterns) == 0:
            print("비교할 과거 패턴이 없습니다.")
            return []

        if len(current_pattern) == 0:
            print("현재 패턴이 유효하지 않습니다.")
            return []

        print("유클리디안 거리 기반으로 가장 유사한 패턴 검색 중...")
        start_time = time.time()

        # GPU 가속 사용
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

                    # 배치 처리로 모든 거리 한번에 계산
                    diff = cp_patterns - cp_current.reshape(1, -1)
                    batch_distances = cp.sqrt(cp.sum(diff * diff, axis=1))

                    # 결과를 CPU로 가져와서 리스트에 추가
                    batch_distances_cpu = cp.asnumpy(batch_distances)

                    for j in range(len(batch_distances_cpu)):
                        distances.append(
                            (batch_distances_cpu[j], timestamps[i + j], patterns[i + j])
                        )

                    # GPU 메모리 해제
                    del cp_patterns, diff, batch_distances
                    cp.get_default_memory_pool().free_all_blocks()

                    # 진행 상황 보고
                    elapsed = time.time() - start_time
                    print(
                        f"진행 중 (GPU): {end_i}/{len(patterns)} 패턴 처리됨 ({end_i/len(patterns)*100:.1f}%) - 경과 시간: {elapsed:.2f}초"
                    )

                # GPU 메모리 해제
                del cp_current
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU 계산 중 오류 발생: {e}, CPU로 전환합니다.")
                # CPU 방식으로 대체
                distances = []
                for i, pattern in enumerate(patterns):
                    distance = self.calculate_similarity(current_pattern, pattern)
                    distances.append((distance, timestamps[i], pattern))

                    if i % 500 == 0 or i == len(patterns) - 1:
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
