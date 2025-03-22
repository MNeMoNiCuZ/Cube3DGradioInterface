@echo off 
call venv\Scripts\activate
cmd /k "cd /d %~dp0 & call venv\Scripts\activate & python cube_gradio_interface.py"