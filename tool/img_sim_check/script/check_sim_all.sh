db_folder_root=/media/sf_E_DRIVE/share/train_div
query_img_root=/media/sf_E_DRIVE/share/chamo_00000/chamo_00000
exe_addr=../build/check_simi

for folder in ${db_folder_root}/*
do
    if [[ -d ${folder} ]]; then
        db_img_root=${folder}
        echo ${folder}
	    if [[ -f img_list.txt ]]; then
	        rm img_list.txt
	    fi

	    for f in $(find ${db_img_root} -name '*.jpg')
	    do
	        echo ${f} >> img_list.txt
	    done
	    
	    if [[ -f query_list.txt ]]; then
            rm query_list.txt
        fi

        for f in $(find ${query_img_root} -name '*.jpg')
        do
            echo ${f} >> query_list.txt
        done
        
        if [[ -d re ]]; then
            rm -r re
        fi
        mkdir re
	    
	    ${exe_addr} test ${training_img_root} img_list.txt query_list.txt
	fi
done
