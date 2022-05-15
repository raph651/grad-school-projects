#!/bin/bash
nginx -t &&
service nginx start &&
streamlit run musical_robots/demo.py --theme.base "dark"