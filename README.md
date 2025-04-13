# Bitcoin Pattern Analyzer

비트코인과 암호화폐 시장의 반복되는 가격 패턴을 다양한 알고리즘으로 분석하고 시각화하는 파이썬 애플리케이션입니다.

## 소개

Bitcoin Pattern Analyzer는 과거 비트코인 가격 데이터에서 현재 패턴과 유사한 과거 패턴을 발견하여 미래 가격 움직임을 예측하는 데 도움을 주는 도구입니다. 다양한 거리 측정 알고리즘을 사용하여 패턴 간의 유사성을 식별합니다.

![프로그램 스크린샷](https://github.com/user-attachments/assets/1ca52119-6285-47d3-b5f7-37edab01468a)


## 주요 기능

- **다양한 유사도 측정 방식**
  - 유클리디안 거리 - 직접적인 좌표 간 거리 비교
  - 상관계수 - 패턴 형태의 상관관계 측정
  - Matrix Profile (STUMPY) - 효율적인 시계열 패턴 검색
  - 형태 기반 거리 - 시간 축 왜곡을 고려한 패턴 유사성

- **GPU 가속화**
  - CUDA 지원으로 계산 속도 대폭 향상
  - 대용량 데이터 처리에 최적화

- **사용자 친화적 인터페이스**
  - 직관적인 GUI로 쉬운 조작
  - 실시간 분석 진행 상황 모니터링
  - 결과 시각화 및 내보내기 기능

- **데이터 관리**
  - 캐싱 기능으로 반복 분석 시 속도 향상
  - 다양한 시간프레임 지원 (15분, 1시간, 4시간, 일봉)

## 설치 방법

### 필수 요구사항

- Python 3.8 이상
- 주요 라이브러리:
  ```
  numpy>=1.19.0
  pandas>=1.1.0
  matplotlib>=3.3.0
  scikit-learn>=0.23.0
  stumpy>=1.11.0
  ccxt>=2.0.0
  tkinter (Python 기본 패키지)
  ```

### GPU 가속을 위한 추가 요구사항
- CUDA 지원 NVIDIA GPU
- CuPy 라이브러리

### 설치 단계

1. **저장소 클론**
   ```bash
   git clone https://github.com/kongzui/BitcoinPatternAnalyzer.git
   cd BitcoinPatternAnalyzer
   ```

2. **필요한 패키지 설치**
   ```bash
   pip install numpy pandas matplotlib scikit-learn stumpy ccxt
   ```

3. **GPU 가속을 위한 CuPy 설치 (선택사항)**
   ```bash
   # CUDA 버전에 맞게 설치
   pip install cupy-cuda11x
   ```

## 사용 방법

### 기본 사용법

1. **애플리케이션 실행**
   ```bash
   python main.py
   ```

2. **분석 설정**
   - 암호화폐 심볼 선택 (BTC/USDT, ETH/USDT 등)
   - 시간프레임 선택 (15m, 1h, 4h, 1d)
   - 유사도 측정 방법 선택
   - GPU 가속 활성화/비활성화

3. **분석 시작**
   - "분석 시작" 버튼 클릭
   - 로그 창에서 진행 상황 확인
   - 결과 그래프 분석

### 결과 해석

![분석 결과 예시](https://github.com/user-attachments/assets/6070102e-40ba-42ed-9f13-3e5eefe0b71d)


- **파란색 선**: 현재 가격 패턴
- **색상 점선**: 과거 유사 패턴
- **색상 실선**: 과거 패턴이 보여준 미래 가격 움직임
- **회색 영역**: 미래 예측 구간

## 구현 상세

### 패턴 추출 및 비교 알고리즘

```python
# 패턴 추출 기본 코드
def extract_patterns(df, window_size=60, target_col="Close"):
    patterns = []
    timestamps = []
    
    for i in range(len(df) - window_size + 1):
        window = df[target_col].iloc[i:i+window_size].values
        normalized_window = (window - window.mean()) / window.std()
        patterns.append(normalized_window)
        timestamps.append(df.index[i])
    
    return patterns, timestamps
```

### 유사도 측정 방식

- **유클리디안 거리**
  ```python
  np.sqrt(np.sum((pattern1 - pattern2) ** 2))
  ```

- **상관계수**
  ```python
  np.corrcoef(pattern1, pattern2)[0, 1]
  ```

- **Matrix Profile**
  ```python
  import stumpy
  matrix_profile = stumpy.stump(time_series, window_size)
  ```

- **형태 기반 거리**
  ```python
  # FFT를 이용한 크로스 상관관계 계산
  fft_p1 = fft(p1_norm)
  fft_p2 = fft(p2_norm)
  ccr = np.abs(ifft(fft_p1 * np.conj(fft_p2)))
  ```

## 향후 계획

- 딥러닝 기반 패턴 인식 추가
- 다중 자산 상관관계 분석
- 백테스팅 기능 구현
- 웹 애플리케이션 버전 개발

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새 기능 브랜치를 만듭니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 연락처

프로젝트 관련 문의:
- GitHub: [kongzui](https://github.com/kongzui)
