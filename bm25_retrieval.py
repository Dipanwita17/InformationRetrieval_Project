import sys
from pyserini.search.lucene import LuceneSearcher

ROLL_NO = 'mtc2414' 

def parse_fire_topics(filepath):
    queries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    qid = None
    for line in lines:
        line = line.strip()
        if line.startswith('<num>'):
            qid = line.split(':')[-1].strip()
        elif line.startswith('<title>') and qid is not None:
            title = line.replace('<title>', '').strip()
            queries[qid] = title
    return queries

def run_bm25(index_dir, topic_file, output_file='bm25_results.txt'):
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)

    queries = parse_fire_topics(topic_file)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for qid, query_text in queries.items():
            hits = searcher.search(query_text, k=2000)
            for rank, hit in enumerate(hits):
                fout.write(f"{qid}\tQ0\t{hit.docid}\t{rank + 1}\t{hit.score:.4f}\t{ROLL_NO}\n")

    print(f"BM25 results saved to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python bm25_retrieval_singlefile.py <index_dir> <topic_file>")
        sys.exit(1)

    run_bm25(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 bm25_retrieval.py <index_dir> <query_dir>")
        sys.exit(1)
