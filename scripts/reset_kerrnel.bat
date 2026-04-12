@echo off
cd /d D:\projects_py\fastapi\ai-bootcamp

jupyter kernelspec uninstall ai-bootcamp -f
uv run python -m ipykernel install --user --name ai-bootcamp --display-name "Python (ai-bootcamp)"

echo Kernel ai-bootcamp reinstalled
pause