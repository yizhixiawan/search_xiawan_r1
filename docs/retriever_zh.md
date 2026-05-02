## 搜索引擎

本文档提供了启动不同检索器的示例，包括本地稀疏检索器（例如 BM25）、本地稠密检索器（例如 e5）以及在线搜索引擎。

对于本地检索器，本文以 [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) 语料库为例。对应的语料索引可以在以下位置找到：[bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index)、[e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)、[e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64)。

### 如何选择检索器？

- 如果你有私有语料库或特定领域语料库，选择**本地检索器**。

    - 如果你的领域里没有高质量的基于 embedding 的检索器（稠密检索器），选择**本地稀疏检索器**（例如 BM25）。

    - 否则，选择**本地稠密检索器**。

        - 如果你没有足够的 GPU 来做精确的稠密向量匹配，选择在 CPU 上运行的 **ANN 索引**。

        - 如果你有足够的 GPU，选择在 GPU 上运行的 **flat indexing**。

- 如果你想训练一个通用的大语言模型搜索智能体，并且预算充足，选择**在线搜索引擎**（例如 [SerpAPI](https://serpapi.com/)）。

- 如果你有特定领域的在线搜索引擎（例如 PubMed search），可以参考这个[链接](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py)，自行将它接入 Search-R1。

搜索引擎启动脚本可以在这个[链接](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever)中找到。

### 本地稀疏检索器

稀疏检索器（例如 BM25）是一种传统方法。它的检索过程非常高效，并且不需要 GPU。不过，在某些特定领域里，它的准确率可能不如稠密检索器。

（1）下载索引。

```bash
save_path=/your/path/to/save
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

（2）启动本地 BM25 检索服务。

```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```

### 本地稠密检索器

你也可以使用一些现成的稠密检索器，例如 e5。在某些特定领域里，这些模型通常比稀疏检索器更强。

如果你有足够的 GPU，推荐使用下面的 flat indexing 版本；否则，可以使用 ANN 版本。

#### Flat indexing

Flat indexing 会进行精确的 embedding 匹配，因此速度较慢但准确率很高。为了让它足够高效，从而支持在线 RL，建议通过 `--faiss_gpu` 启用 **GPU**。

（1）下载索引和语料库。

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

（2）启动本地 flat e5 检索服务。

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu
```

#### ANN indexing（HNSW64）

如果只使用 **CPU**，为了提升搜索效率，可以采用近似最近邻（Approximate Nearest Neighbor，ANN）索引，例如 HNSW64。

这种方式非常高效，但准确率可能不如 flat indexing，尤其是在召回 passage 数量较少时更明显。

（1）下载索引。

```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```

（2）启动本地 ANN 稠密检索服务。

```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```

### 在线搜索引擎

项目同时支持 [Google Search API](https://developers.google.com/custom-search/v1/overview) 和 [SerpAPI](https://serpapi.com/)。

更推荐使用 [SerpAPI](https://serpapi.com/)，因为它集成了多个在线搜索引擎 API，包括 Google、Bing、Baidu 等，并且没有月度配额限制。相比之下，[Google Search API](https://developers.google.com/custom-search/v1/overview) 每月有固定的 10k 次配额上限，这通常不足以支撑在线 LLM RL 训练。

#### SerpAPI 在线搜索服务

```bash
search_url=https://serpapi.com/search
serp_api_key="" # 在这里填写你的 serp api key（https://serpapi.com/）

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google 在线搜索服务

```bash
api_key="" # 在这里填写你的 google custom API key（https://developers.google.com/custom-search/v1/overview）
cse_id="" # 在这里填写你的 google cse API key（https://developers.google.com/custom-search/v1/overview）

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```
