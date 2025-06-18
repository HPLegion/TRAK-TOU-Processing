mkdir _build
del tou_batch.pyz
xcopy /e tct .\_build\tct\
rmdir /S /Q .\_build\tct\__pycache__
python -m zipapp --main tct.tou_batch:main _build
ren _build.pyz tou_batch.pyz
rmdir /S /Q .\_build