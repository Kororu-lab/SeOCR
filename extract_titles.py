import os, re
from pathlib import Path

def extract_title(filename):
    # 파일 확장자 제거
    filename = filename.replace('.json', '')
    
    # 언더스코어 위치 찾기
    first = filename.find('_')
    second = filename.find('_', first + 1)
    third = filename.find('_', second + 1)
    last = filename.rfind('_')
    
    print(f"Processing: {filename}.json")
    print(f"Underscores at: {first}, {second}, {third}")
    
    # 두 번째와 세 번째 언더스코어 사이의 문자열 추출
    if second != -1 and third != -1:
        title = filename[second+1:third]
        # 권 번호와 숫자 제거
        title = re.sub(r'\s*권\s*\w+$|\s*\d+$', '', title)
        print(f"Extracted title: {title}")
        return title.strip()
    return None

def process_json_files(directory):
    titles = set()
    count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                count += 1
                title = extract_title(file)
                if title:
                    titles.add(title)
    
    # 결과를 파일에 저장
    with open('/home/kororu/SeOCR/data/doc.txt', 'w', encoding='utf-8') as f:
        for title in sorted(titles):
            f.write(title + '\n')
    
    print(f"\n추출된 문헌 수: {len(titles)}")
    print("\n처음 10개 문헌:")
    for title in sorted(list(titles))[:10]:
        print(title)

titles = set()
for type_dir in ["목판본", "필사본", "활자본"]:
    path = Path("/home/kororu/SeOCR/data") / type_dir
    print(f"\nProcessing directory: {path}")
    if path.exists():
        for file in path.glob("*.json"):
            title = extract_title(file.name)
            if title:
                titles.add(title)

# 정렬된 목록을 doc.txt에 저장
with open("/home/kororu/SeOCR/data/doc.txt", "w", encoding="utf-8") as f:
    for title in sorted(titles):
        f.write(title + "\n")

print("\n추출된 문헌 수:", len(titles))
print("\n처음 10개 문헌:")
for title in sorted(list(titles))[:10]:
    print(title) 