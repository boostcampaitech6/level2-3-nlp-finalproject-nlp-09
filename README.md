# 📖ABCDE(A BoostCamp Diary for Emotions) : 대화형 감정 일기


https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/assets/81287077/949b75bb-c02f-4ad0-8322-393ea13cf2a2



**ABCDE**는 하루의 일상을 마무리하면서 🦆**Dr.부덕이**와 나눈 대화를 바탕으로 일기를 생성해주는 감정 일기 서비스입니다.
- **[직접 체험하기](https://m2af-abcde.streamlit.app/)** (서비스 기간: ~24.04.02)
- 발표 영상: [youtube](https://youtu.be/r7ngZ25C5qg)
- 더 많은 정보: [report.pdf](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/report.pdf)
## Quick Start
각 폴더의`README` 파일을 참하세요.
- data: [data/README.md](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/data/README.md)
- streamlit: [streamlit/README.md](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/streamlit/README.md)
- models
  - chat: [models/chat/README.md](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/models/chat/README.md)
  - emotion classifier: [models/classifier/README.md](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/models/classifier/README.md)
  - summary: [models/summary/README.md](https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-09/blob/develop/models/summary/README.md)


## Datasets
- Crawled Data from 5 Platforms : [link](https://huggingface.co/datasets/m2af/ko-emotion-dataset)
- AIHub 공감형 대화 : [link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71305)
- AIHub 감성 대화 : [link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86)

## 😋 TEAM Moth2AFlame (역삼동불나방)
### About us
- 네이버 커넥트재단 부스트캠프 AI Tech 6기 NLP 9조
- **[Team Hub](https://huggingface.co/m2af)**
### Members
| [전현욱](https://github.com/enearsist) | [김가영](https://github.com/garongkim) | [김신우](https://github.com/kimsw9703) | [안윤주](https://github.com/nyunzoo) | [곽수연(명예팀원)](https://github.com/gongree) |
| --- | --- | --- | --- | --- |
| <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0a2cc555-e3fc-4fb1-9c05-4c99038603b3)" width="140px" height="140px" title="Hyunwook Jeon" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0fb3496e-d789-4368-bbac-784aeac06c89)" width="140px" height="140px" title="Gayoung Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/77b3a062-9199-4d87-8f6e-70ecf42a1df3)" width="140px" height="140px" title="Shinwoo Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/f3b42c80-7b82-4fa1-923f-0f11945570e6)" width="140px" height="140px" title="Yunju An" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/d500e824-f86d-4e72-ba59-a21337e6b5a3)" width="140px" height="140px" title="Suyeon Kwak" /> |

- **팀 공통**
  - 데이터 구축 및 라벨링
- **전현욱**
  - 팀 리더, LLM QLoRA Fine-tuning, Prompt Engineering
- **김가영**
  - AI 모델 서빙 구현, 사용자별 통계 지표 산출, Active Learning 및 Classifier 모델 구축
- **김신우**
  - DBA, Front-End 구현, Active Learning 및 Classifier 모델 구축
- **안윤주**
  - PM, 데이터 전처리 및 EDA, 검증 시나리오 셋 구축
 
## References
- [polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)
- [OrionStarAI](https://huggingface.co/OrionStarAI)
