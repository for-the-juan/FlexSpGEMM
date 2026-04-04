#/bin/bash

TILE_SIZE_M=(8 16 32 8 16 32 16 8 32)
TILE_SIZE_N=(8 16 32 16 8 16 32 32 8)

length=${#TILE_SIZE_M[@]}

# AAT="/home/stu1/Dataset/simple"
AA="/home/stu1/Dataset/HYTEDataset"

mkdir -p ./log_dseHYTE
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
    mkdir -p ./log_dseHYTE/${base_name_}
    for ((i=0; i<length; i++)); do
        m=${TILE_SIZE_M[i]}
        n=${TILE_SIZE_N[i]}
        ./bin/test_m${m}_n${n} -d 0 -aat 0 $mtx_file > log_dseHYTE/${base_name_}/aat0_m${m}_n${n}.log
        echo "${base_name_}_aat0_m${m}_n${n} Finished!"
    done
done