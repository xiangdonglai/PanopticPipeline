BASENAME='190611_asl'
BASE_PROCESSED_PATH='/media/posefs5b/Processed/specialEvents/'

for i in 1 2 3 4 5 6 7 8 10 11 12 13 14 15
do
    SEQNAME="${BASENAME}${i}"
    echo $SEQNAME
    PROCESSED_PATH="$BASE_PROCESSED_PATH${SEQNAME}/"
    python copy_to_domdbweb.py $SEQNAME $PROCESSED_PATH op25
done
