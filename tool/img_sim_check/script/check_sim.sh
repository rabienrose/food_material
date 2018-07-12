db_img_root=$1
query_img_root=$2

exe_addr=/home/leo/Documents/chamo/transfer/tool/img_sim_check/build/check_simi

for f in $(find ${db_img_root} -name '*.jpg')
do
    echo ${f} >> img_list.txt
done

for f in $(find ${query_img_root} -name '*.jpg')
do
    echo ${f} >> query_list.txt
done

${exe_addr} test ${training_img_root} img_list.txt query_list.txt
