# Streamlit

streamlit 앱 실행 방법

## Dependencies

- python== 3.8.10
- streamlit==1.32.2
- pandas==2.1.1
- matplotlib==3.8.3
- koreanize_matplotlib==0.1.1
- scipy==1.11.3

```bash
$ pip3 install -r requirements.txt
```

## Run streamlit app

```bash
$ cd streamlit
$ streamlit run main.py
```

### 8080포트가 아니라 다른 포트를 사용할 경우

/.streamlit/config.toml 파일의 port를 원하는 포트로 수정해주시면 됩니다.

```python
[server]
port = 8080
```