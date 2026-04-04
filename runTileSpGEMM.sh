#/bin/bash

# AAT="/home/stu1/Dataset/simple"
AA="/home/stu1/Dataset/TileSpGEMMDataset"

mkdir -p ./TileSpGEMMlog_0304
# for mtx_file in "$AAT"/*.mtx; do
#     base_name_=$(basename $mtx_file .mtx)
#     for ((i=0; i<length; i++)); do
#         m=${TILE_SIZE_M[i]}
#         n=${TILE_SIZE_N[i]}
#         ./bin/test_m${m}_n${n} -d 0 -aat 1 $mtx_file > log/aat1_${base_name_}_m${m}_n${n}.log
#     done
# done

for mtx_file in "$AA"/*.mtx; do
    base_name_=$(basename $mtx_file .mtx)
    # /home/stu1/TileSpGEMM/src/test -d 0 -aat 0 $mtx_file > TileSpGEMMlog/${base_name_}.log
    /home/stu1/chenruiyang/TileSpGEMM/src/test -d 0 -aat 0 $mtx_file > TileSpGEMMlo_0304/${base_name_}.log
    echo "${base_name_} Finished!"
done