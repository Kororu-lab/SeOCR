컴퓨팅 리소스가 충분하다면, 더 강력한 모델 조합을 제안드립니다:
Detection: CRAFT + Transformer
장점:
CRAFT의 정확한 문자 영역 감지
Transformer의 글로벌 컨텍스트 이해
복잡한 레이아웃 처리 능력 향상
구현:
CRAFT로 초기 텍스트 영역 감지
Transformer로 컨텍스트 기반 후처리
멀티스케일 피처 추출
Classification: ViT-Large + CRNN
장점:
ViT의 강력한 피처 추출 능력
CRNN의 시퀀스 처리 능력
높은 정확도와 강건성
구현:
ViT로 글로벌 컨텍스트 이해
CRNN으로 시퀀스 기반 문자 인식
앙상블 기법으로 최종 결과 도출