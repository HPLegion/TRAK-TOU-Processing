mkdir _archive
del tou_batch.pyz
copy tou_batch.py .\_archive\
copy tou_particle.py  .\_archive\
copy import_tou.py .\_archive\
copy __main__.py .\_archive\
python -m zipapp _archive
ren _archive.pyz tou_batch.pyz
rmdir /S /Q .\_archive