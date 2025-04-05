# Data Directory Structure

This directory should contain the AI Hub's Old Korean OCR Dataset. Due to licensing restrictions, the data is not included in this repository.

## Data Source
- **Provider**: AI Hub
- **Dataset**: [고문헌 OCR 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71295)
- **License**: AI Hub License (Requires Registration)

## Dataset Characteristics

### 1. Manuscripts (필사본)
- Handwritten historical documents
- Contains both Hangeul and Hanja
- Characteristics:
  * Various writing styles and calligraphy
  * Different ink densities and paper conditions
  * Potential degradation and damage
- Format: `필사본_[ID]_[Category]_[Title]_[Page]_[Image].{json,png}`

### 2. Woodblock Prints (목판본)
- Printed using traditional woodblock printing methods
- Characteristics:
  * Consistent character shapes within same document
  * Possible wood grain patterns
  * Varying ink distributions
- Format: `목판본_[ID]_[Category]_[Title]_[Page]_[Image].{json,png}`

### 3. Movable Type Prints (활자본)
- Printed using traditional movable type printing
- Characteristics:
  * Regular and consistent character shapes
  * Clear character boundaries
  * Potential alignment variations
- Format: `활자본_[ID]_[Category]_[Title]_[Page]_[Image].{json,png}`

## Directory Structure
```
data/
├── manuscripts/      # 필사본
│   ├── [ID]/
│   │   ├── images/
│   │   └── labels/
├── woodblocks/      # 목판본
│   ├── [ID]/
│   │   ├── images/
│   │   └── labels/
└── movable_types/   # 활자본
    ├── [ID]/
    │   ├── images/
    │   └── labels/
```

## File Format

### Image Files (*.png)
- Resolution: Varies by document
- Color Space: RGB
- Format: PNG

### Label Files (*.json)
```json
{
    "Typeface": "필사본|목판본|활자본",
    "Info_id": "문서 ID",
    "Category_name": "문서 분류",
    "Info_name": "문서명",
    "Period": "시대",
    "Author": "저자",
    "Difficulty": "난이도(A,B,C)",
    "License_name": "라이선스",
    "Info_Data_created": "데이터 생성일",
    "Image_id": "이미지 ID",
    "Image_filename": "이미지 파일명",
    "Image_Data_captured": "이미지 촬영일",
    "Image_Width": "이미지 너비",
    "Image_Height": "이미지 높이",
    "Image_dpi": "이미지 DPI",
    "Image_Char_no": "문자 수",
    "Text_Coord": [
        {
            "bbox": [x1, y1, x2, y2, line_idx, char_idx],
            "annotate": "문자"
        }
    ]
}
```

## Data Preparation

1. Register at [AI Hub](https://aihub.or.kr)
2. Download the Old Korean OCR Dataset
3. Extract the downloaded files
4. Place the data in the appropriate directories following the structure above
5. Run the data verification script:
   ```bash
   python src/utils/verify_data.py
   ```

## Notes
- The dataset is large (~100GB+)
- Some images may require preprocessing due to quality issues
- Text annotations follow reading order (top-to-bottom, right-to-left)
- Character bounding boxes include line and character position information 