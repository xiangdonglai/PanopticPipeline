cd ./panopticstudio/Application/syncTableGeneratorVHK
seqName=190211_tent
nasIdx=12a
idx=1
category=specialEvents
# ./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables ~/tmp/${seqName}$idx 15 15 1 t f 1000
mv ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* ~/tmp/${seqName}$idx/
rm -r ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
rm -r ~/tmp/${seqName}$idx/originalImgs
idx=2
# ./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables ~/tmp/${seqName}$idx 15 15 1 t f 1000
mv ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* ~/tmp/${seqName}$idx/
rm -r ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
rm -r ~/tmp/${seqName}$idx/originalImgs/
idx=3
# ./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables ~/tmp/${seqName}$idx 15 15 1 t f 1000
mv ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* ~/tmp/${seqName}$idx/
rm -r ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
rm -r ~/tmp/${seqName}$idx/originalImgs/
