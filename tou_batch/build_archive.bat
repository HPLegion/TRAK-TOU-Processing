mkdir _archive
del tou_batch.pyz
xcopy tou_batch.py .\_archive\
xcopy __main__.py .\_archive\
xcopy /e ..\tct .\_archive\tct\
rmdir /S /Q .\_archive\tct\__pycache__
python -m zipapp _archive
ren _archive.pyz tou_batch.pyz
rmdir /S /Q .\_archive