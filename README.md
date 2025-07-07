# 11일차

## 머신러닝 학습 방법 비교
[머신러닝 학습 방법에 대해 정리](ML.md)

## tensorflow, K-means 클러스터링 사용 예제
[예제 코드](0707_python_tensorflow,_K_means_클러스터링_사용_예제.ipynb)

**신호등 딥러닝 예제**
- 딥러닝 시행을 의미하는 Epoch 에포크시, ADAM을 많이 쓴다.
- 예제코드는 정확도가 너무 낮아 실전에는 안쓰고, tensorflow와 cnn모델을 사용해보는데 의의를 둠.

**비지도학습 자율주행운동 패턴 예제**
- 예제 내부에서 운전 유형과 정체상황등의 자율주행 데이터를 미리 정의함.

## openCV와 matplotlib 비교
| openCV에서 많이 등장하는 용어                                       | 설명                                                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Mat**                                  | 이미지 데이터를 담는 OpenCV 기본 객체. 다차원 배열(matrix) 형태로 픽셀 값을 저장.                                             |
| **BGR/RGB**                              | OpenCV 기본 컬러 순서는 BGR(Blue, Green, Red). Matplotlib 등은 RGB 순서를 쓰므로 변환이 필요함.                         |
| **ROI**                                  | Region Of Interest. 이미지에서 관심 있는 부분(영역)을 잘라내어 처리할 때 사용.                                             |
| **픽셀(Pixel)**                            | 영상의 최소 단위. Mat 객체의 각 요소(element)에 해당하며, 그레이스케일은 1채널, 컬러는 3채널(BGR)로 표현.                             |
| **채널(Channel)**                          | 각 픽셀이 갖는 값의 축. 그레이스케일=1채널, 컬러=3채널(B, G, R).                                                        |
| **스레시홀드(Threshold)**                     | 임계값을 기준으로 이진화(흑/백) 이미지를 만드는 연산. `cv2.threshold()`에서 다양한 모드 사용 가능.                                  |
| **가우시안 블러(Gaussian Blur)**               | `cv2.GaussianBlur()`로 노이즈 제거용 블러 처리. 커널 크기와 표준편차를 조정하여 부드러운 흐림 효과 적용.                              |
| **모폴로지 연산(Morphology)**                  | 침식(Erode), 팽창(Dilate), 열기(Open), 닫기(Close) 등을 통해 이진 이미지를 정제하거나 모양을 변화시킴.                           |
| **엣지 검출(Edge Detection)**                | Canny(`cv2.Canny()`), Sobel, Laplacian 등 경계선(엣지)을 검출하는 알고리즘.                                       |
| **컨투어(Contour)**                         | `cv2.findContours()`로 이미지 내 객체 경계를 추출한 뒤, `cv2.drawContours()`로 윤곽선을 그릴 수 있음.                      |
| **히스토그램(Histogram)**                     | `cv2.calcHist()`로 영상의 픽셀 밝기 분포를 계산. 히스토그램 평활화(`cv2.equalizeHist()`)로 대비 개선.                        |
| **컬러 공간 변환**                             | `cv2.cvtColor()`로 BGR↔RGB, BGR↔HSV, BGR↔GRAY 등 색공간 변환. HSV는 색상 기반 분할에 자주 사용.                       |
| **기하 변환(Geometric Transform)**           | `cv2.resize()`, `cv2.warpAffine()`, `cv2.warpPerspective()` 등으로 크기 조절, 회전, 원근(투시) 변환 수행.           |
| **피처 검출/매칭(Feature Detection/Matching)** | SIFT, SURF, ORB, AKAZE 등 알고리즘으로 키포인트와 디스크립터를 추출하고, BFMatcher나 FLANN으로 매칭.                          |
| **카메라 보정(Camera Calibration)**           | 왜곡 보정, 카메라 매트릭스 산출을 위해 chessboard 패턴 촬영 후 `cv2.calibrateCamera()` 사용.                              |
| **비디오 캡처(VideoCapture)**                 | `cv2.VideoCapture()`로 카메라나 비디오 파일에서 프레임을 읽어 들이기. `read()`로 프레임 획득 후 `imshow()`로 표시.                |
| **윈도우(Window)**                          | `cv2.namedWindow()`, `cv2.imshow()`, `cv2.waitKey()`, `cv2.destroyAllWindows()`로 영상 출력 및 키 이벤트 처리. |

-> 이 외에도 OpenCV는 머신러닝 통합, 딥러닝 지원(DNN 모듈), 3D 재구성, 영상 추적 등 방대한 기능을 제공함.

![image](https://github.com/user-attachments/assets/e274e111-666e-4cc5-9aa8-e6481cfad954)
![image](https://github.com/user-attachments/assets/6ce90bbf-d45b-4518-adc8-917ace23b728)

OpenCV와 Matplotlib이 이미지를 다루는 방식 차이에서 비롯된 동작입니다.
1. 배열 차원과 채널 수<br>
그레이스케일 (2차원 배열)<br>
shape = (높이, 너비)<br>
채널이 1개이므로, “숫자값 하나”가 픽셀의 밝기를 바로 나타냄

컬러 (3차원 배열)<br>
shape = (높이, 너비, 3)<br>
채널 3개(R, G, B)는 각각 빨강·초록·파랑 강도를 뜻함

2. Matplotlib imshow의 동작 원리<br>
2D 배열을 주면 픽셀값(스칼라)을 컬러맵(cmap)으로 “색깔에 매핑”<br>
ex) cmap='jet' → 파랑에서 빨강까지 색이 변하는 그라디언트 맵

3D 배열을 주면
“이미 완성된 컬러 이미지(RGB)”로 간주하고,
cmap 파라미터는 무시 → 입력된 R/G/B 값을 그대로 보여줌

3. OpenCV(BGR) vs. Matplotlib(RGB) 컬러 순서<br>
OpenCV로 읽은 이미지는 기본적으로 BGR 순서<br>
Matplotlib에 그대로 넘기면 색상이 뒤바뀔 수 있으므로, 올바른 컬러 출력을 위해<br>
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)<br>
plt.imshow(gray, cmap='jet')<br>
코드를 추가해야함.

4. 요약<br>
그레이스케일: 2D → 값 → 컬러맵 적용 ✔️<br>
컬러: 3D → 이미 완성된 RGB → 컬러맵 무시 ✔️<br>
컬러맵을 적용하고 싶다면, 3채널 이미지를 한 번 그레이스케일로 변환(또는 특정 채널만 뽑아서) 2D 배열로 만들어야 합니다.

