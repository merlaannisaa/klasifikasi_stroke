mkdir -p ~/.streamlit/                                               
echo "\                       
[server]\n\                       
port = $PORT\n\                       
enableCORS = false\n\                       
headless = true\n\   
showPyplotGlobalUse = false\n\
\n\                       
" > ~/.streamlit/config.toml
