# FCN(2015) PyTorch 논문구현
https://arxiv.org/pdf/1411.4038
## 구조
* model - FCN.py(16s, 8s)
* train - main.ipynb(train/validation/Inference)

## 회고
* 논문을 보고 FCN구조를 코드로 작성해보았습니다.
* 처음으로 Segmentation학습을 기준으로 진행한 코드이며, 진행과정에서 많은 에러와 CUDA Error도 접해보았습니다.
* GPU 환경이 안되어서 코랩 T4 기준으로 10 에포크로만 돌렸지만 역시... 좋은 결과가 나오지는 못했지만 Segmentation 학습을 진행한것에 큰 의의를 두도록 합니다.
* [수정] **2025.1.20**  Kaggle notebooks T4x2  환경으로 에포크 50으로 돌렸지만 과적합을 확인하였고 과적합을 해결해보는 시간을 가졌다.
https://velog.io/@kndh2914/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B3%BC%EC%A0%81%ED%95%A9-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95

