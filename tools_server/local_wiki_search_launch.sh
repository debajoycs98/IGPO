
# /mnt/xhunter是挂载的nas
# mkdir /mnt/xhunter
# mount.nfs4 -o "nfsvers=3,nolock,noatime,noacl,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport" 267242491d5-iim80.cn-heyuan-alipay.nas.aliyuncs.com:/ /mnt/xhunter

file_path=/mnt/xhunter/shaobing/Search-R1/data
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/xhunter/shaobing/Search-R1/model/e5-base-v2

# export CUDA_VISIBLE_DEVICES=0,1 # 限制在前2个gpu

python  tools_server/search/local_wiki_search.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
