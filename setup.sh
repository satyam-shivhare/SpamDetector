mkdir -p ~/.streamlit/
echo ")
[server]in\
port = SPORT \n\
enableCORS = false\n\
headless = true\n\
" > ~/.streamlit/config.toml
web: sh setup.sh && streamlit run app.py