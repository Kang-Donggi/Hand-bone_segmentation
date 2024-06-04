# hand bone segmentation

### 개요:
- 손을 촬영한 X-Ray 이미지에서 sementic segmentation을 수행

### 데이터:
- 400명의 좌, 우 손 X-Ray 데이터 총 800장의 이미지 존재
  
|split|Images|Label|
|---|---|---|
|Train|800|O|
|Test|288|X|

- 29개의 class 존재<br/>
  [finger-1, finger-2, finger-3, finger-4, finger-5, finger-6, finger-7, finger-8, finger-9, finger-10<br/>
  finger-11, finger-12, finger-13, finger-14, finger-15, finger-16, finger-17, finger-18, finger-19,<br/>
  Trapezium, Trapezoid, Capotate, Hamate, Scaphoid, Lunate, Triquetrum, Pisiform, Radius, Ulan]
![output](https://github.com/KANG-dg/hand-bone-segmentation/assets/121837927/5eabd5f8-812c-4698-9895-f57dc78ab3d9)

### 사용기법 & 결과:
- 해당 task는 하나의 픽셀에 여러게의 label이 존재하는 multi label segmentation
- pytorch 기반 fcn_resnet50을 기반으로 실험 진행
- mmsegmentation을 통해 segformer_mit를 통해 추가 실험
- object detection과 동일하게 입력 이미지의 크기가 원본에 가까울 수록 성능에 긍정적 영향

### 결과 & 회고
- mmsegmentation을 공부하는것에 초점을 두고 프로젝트를 진행 하였다
- document를 따라하며 실행에는 성공하였지만 전체적인 구조 파악,
  필요한 부분에 대해 <br/> customize 할 수 있도록 더 공부할 필요를 느꼈다.
- segmentation task가 기본적은 augmenatation을 적용할때도 
  많은 부분을 고려해야해서 모델 최적화에 특히 어려움을 느꼈다.
  따라서 전처리와 적절한 모델 탐색이 더욱 중요할 것으로 생각된다. 
