db_img_root=/home/rviz/Documents/img_sim_check/script/test1
query_img_root=/home/rviz/Documents/img_sim_check/script/test2

exe_addr=../build/check_simi
if [[ -f img_list.txt ]]; then
    rm img_list.txt
fi

if [[ -f query_list.txt ]]; then
    rm query_list.txt
fi

for f in $(find ${db_img_root} -name '*.jpg')
do
    echo ${f} >> img_list.txt
done

for f in $(find ${query_img_root} -name '*.jpg')
do
    echo ${f} >> query_list.txt
done

${exe_addr} test ${training_img_root} img_list.txt query_list.txt
