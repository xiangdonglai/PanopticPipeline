BASENAME='190419_asl'
BASE_PROCESSED_PATH='/media/posefs11b/Processed/specialEvents/'

for i in 2 4 5
do
    SEQNAME="${BASENAME}${i}"
    echo $SEQNAME
    PROCESSED_PATH="$BASE_PROCESSED_PATH${SEQNAME}/"
    python copy_to_domdbweb.py $SEQNAME $PROCESSED_PATH op25
done
