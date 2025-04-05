# 테스트 하이퍼파라미터 설정
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

# 모델 설정
TEST_CONFIG = {
    'model_path': 'seocr_checkpoints/best_model',  # 테스트할 모델 경로
    'num_samples': 5,                              # 테스트할 샘플 수
    'font_path': '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'  # 옛한글에 더 적합한 명조체 사용
}

# 폰트 설정
font_path = TEST_CONFIG['font_path']
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumMyeongjo'
else:
    print("Warning: NanumMyeongjo font not found. Please install it: sudo apt-get install fonts-nanum")

# 추론 테스트
def predict_text(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(Config.DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

def display_predictions(dataset, num_samples=TEST_CONFIG['num_samples']):
    # 랜덤하게 샘플 선택
    indices = random.sample(range(len(dataset)), num_samples)
    
    # 서브플롯 생성
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4*num_samples))
    
    for idx, i in enumerate(indices):
        # 원본 이미지 경로와 텍스트 가져오기
        sample = dataset.examples[i]
        image_path = sample['image_path']
        original_text = sample['text']
        
        # 이미지 로드 및 표시
        image = Image.open(image_path)
        axes[idx, 0].imshow(image)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title('원본 이미지')
        
        # OCR 예측 수행
        predicted_text = predict_text(image_path)
        
        # 결과 텍스트 표시
        axes[idx, 1].text(0.1, 0.5, 
                         f'원본 텍스트:\n{original_text}\n\n예측 텍스트:\n{predicted_text}',
                         fontsize=12, 
                         verticalalignment='center')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 저장된 모델 불러오기
if os.path.exists(TEST_CONFIG['model_path']):
    model.load_model(TEST_CONFIG['model_path'])
    print(f"모델을 불러왔습니다: {TEST_CONFIG['model_path']}")
else:
    print(f"Warning: 모델을 찾을 수 없습니다: {TEST_CONFIG['model_path']}")
    print("기본 모델을 사용합니다.")

print("추론 테스트 시작...")
display_predictions(dataset)
print("추론 테스트 완료")