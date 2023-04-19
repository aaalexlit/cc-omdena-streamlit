## Local development

1. If needed update `EVIDENCE_API_IP` in [docker-compose.yml](docker-compose.yml)
1. start up locally on Docker with
    ```bash
   docker compose up --build --force-recreate
   ```
1. for the first time container will take a while to start 
   because it's downloading all the used models
   after the container is started the app is accessible on [http://127.0.0.1:8501/](http://127.0.0.1:8501/)
1. you can change the streamlit app .py files and the changes will be picked up
1. Important: if a page is importing functions eg
   ```python
   main_page.split_into_sentences(text_input)
   ```
   the changes in the imported function won't be picked up
   automatically, you'd need to restart the container for them to be 
   picked up
1. clean up when done
    ```bash
   docker compose down
   ```

## Build and push docker image
Build
```bash
docker build -t aaalexlit/cc-omdena-streamlit .
```

Run
```shell
docker run -d --name cc-omdena-streamlit -p 8501:80 -e EVIDENCE_API_IP=35.204.71.90 aaalexlit/cc-omdena-streamlit
```

Push
```bash
docker tag aaalexlit/cc-omdena-streamlit:latest aaalexlit/cc-omdena-streamlit:0.0.x
docker push aaalexlit/cc-omdena-streamlit:0.0.x
```

## Accelerator
For the time being the application is deployed on GCP (from a docker image)
on a CPU-powered machine.
Theoretically, with no or little change it can also be deployed to a 
GPU-powered machine and take advantage of it using any Cloud provider as
well as locally. Unfortunately, I don't have access to a machine with
GPU to test it out.