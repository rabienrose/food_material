db_img_root=/media/sf_E_DRIVE/share/train_div
exe_addr=../build/check_simi
#if [[ -f img_list.txt ]]; then
#    rm img_list.txt
#fi

#for f in $(find ${db_img_root} -name '*.jpg')
#do
#    echo ${f} >> img_list.txt
#done

${exe_addr} gen img_list.txt 
