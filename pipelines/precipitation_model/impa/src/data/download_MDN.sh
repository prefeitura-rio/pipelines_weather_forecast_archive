#!/bin/bash

first_dt="$1"
last_dt="$2"

# Check if another instance of this script is running
lockdir=/tmp/download_hdf_PPI_MDN.lock
if mkdir -- "$lockdir"
then
    trap 'rm -rf -- "$lockdir"' 0
else
    echo "Permission denied: cannot acquire lock."
    exit 0
fi

DOWNLOADED_FILES_LOG="data/raw/radar_PPI_MDN/downloaded_files.log"
rm -f $DOWNLOADED_FILES_LOG

# Set path variable to gsutil and poetry
PATH=/impa/home/a/antonio.catao/google-cloud-sdk/bin/:/opt/poetry/bin:$PATH
export PYTHONPATH=/impa/home/a/antonio.catao/rio_rain/

first_date=${first_dt:0:8}
last_date=${last_dt:0:8}

# Iterate over every folder indicating vol
for i in $(gsutil ls gs://rj-escritorio-scp/mendanha/odimhdf5/vol_cor | grep -e "$first_date" -e "$last_date"); do
    # Get the name of current file
    filename=$(basename "$i")

    # Get the date of the file
    file_dt=$(echo $filename | sed 's/.*MDN.\([^ ]*\).PVOL.*/\1/')

    # Get full filepath
    filepath="data/raw/radar_PPI_MDN/vol_cor/vol_cor_$filename"
    ext="${filepath##*.}"

    # Check if the file is in the desired range
    if [[ "$file_dt" < "$first_dt" ]] || [[ "$file_dt" > "$last_dt" ]]; then
        continue
    fi

    # Check if the file is a .gz file

    echo "$filepath" >> $DOWNLOADED_FILES_LOG

    [ $ext == "gz" ] || continue
    if ! test -f "$filepath"; then
        # Download the file from the cloud
        gsutil cp gs://rj-escritorio-scp/mendanha/odimhdf5/vol_cor/$filename $filepath
    fi
done
