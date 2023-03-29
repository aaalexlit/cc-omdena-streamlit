#!/usr/bin/env bash
python check_gpu.py
streamlit run streamlit_app.py --server.port=80 --server.address=0.0.0.0