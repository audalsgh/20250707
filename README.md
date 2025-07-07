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
Matplotlib에 그대로 넘기면 색상이 뒤바뀔 수 있으므로

**올바른 컬러 출력**<br>
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)<br>
plt.imshow(rgb)

4. 요약<br>
그레이스케일: 2D → 값 → 컬러맵 적용 ✔️<br>
컬러: 3D → 이미 완성된 RGB → 컬러맵 무시 ✔️<br>
컬러맵을 적용하고 싶다면, 3채널 이미지를 한 번 그레이스케일로 변환(또는 특정 채널만 뽑아서) 2D 배열로 만들어야 합니다.

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)<br>
plt.imshow(gray, cmap='jet')
