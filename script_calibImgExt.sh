cd ./panopticstudio/Application/syncTableGeneratorVHK
seqName=$1  # e.g. 190211_tent
nasIdx=$2 # e.g. 12a
outputDate=$3  # e.g. 190211
rootPath=$4  # e.g. /mnt/sda/donglaix/
idx=$5  # e.g. 3

basePath=$rootPath/$outputDate/
category=specialEvents
./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables $basePath/${seqName}$idx 15 15 1 t f 1000
mv $basePath/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* $basePath//${seqName}$idx/
rm -r $basePath/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
rm -r $basePath/${seqName}$idx/originalImgs

# idx=2
# ./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables ~/tmp/${seqName}$idx 15 15 1 t f 1000
# mv ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* ~/tmp/${seqName}$idx/
# rm -r ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
# rm -r ~/tmp/${seqName}$idx/originalImgs/
# 
# idx=3
# ./ImageExtractorVHK VHK_25 /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx /media/posefs${nasIdx}/Captures/${category}/${seqName}$idx/sync/syncTables ~/tmp/${seqName}$idx 15 15 1 t f 1000
# mv ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/* ~/tmp/${seqName}$idx/
# rm -r ~/tmp/${seqName}$idx/originalImgs/vga_25/000XX/00000015/
# rm -r ~/tmp/${seqName}$idx/originalImgs/
