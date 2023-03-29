Build
```bash
docker build -t aaalexlit/cc-omdena-streamlit .
docker run -d --name cc-omdena-streamlit -p 8501:80 -e EVIDENCE_API_IP=35.204.182.201 aaalexlit/cc-omdena-streamlit
```

Push
```bash
docker tag aaalexlit/cc-omdena-streamlit:latest aaalexlit/cc-omdena-streamlit:0.0.x
docker push aaalexlit/cc-omdena-streamlit:0.0.x
```