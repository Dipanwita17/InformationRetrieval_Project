import argparse
import math
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
import numpy as np
from scipy.stats import gumbel_r  # frechet_r removed since it doesn't exist

class MVDRetriever:
    def __init__(self, index_path):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(0.9, 0.4)
        self.alpha = 0.6
        self.beta = 1.5
        self.k = 1.5
        self.z1 = 2.5
        self.z2 = 0.04
        self.avg_doc_len = None
        self.collection_size = None

    def initialize_collection_stats(self):
        """Fallback defaults: Pyserini LuceneSearcher doesn't expose collection stats directly"""
        self.avg_doc_len = 500  # Adjust if known
        self.collection_size = 1000000  # Adjust if known

    def extract_features(self, query_terms, doc_id):
        """Extract features for a given document"""
        try:
            doc = self.searcher.doc(doc_id)
            if not doc or not doc.raw():
                return None

            terms = doc.raw().lower().split()
            doc_len = len(terms)
            tf_stats = defaultdict(int)

            for term in terms:
                tf_stats[term] += 1
            max_tf = max(tf_stats.values()) if tf_stats else 1

            features = {}
            for term in query_terms:
                tf = tf_stats.get(term, 0)
                ritf = math.log(1 + tf) / math.log(self.k + max_tf) if max_tf > 0 else 0
                lrtf = tf * math.log(1 + self.avg_doc_len / doc_len) if doc_len > 0 else 0

                # Document frequency: using workaround
                try:
                    df = self.searcher.get_doc_frequency(term)
                except:
                    df = 1

                features[term] = {
                    'tf': tf,
                    'ritf': ritf,
                    'lrtf': lrtf,
                    'df': df
                }

            return features
        except Exception as e:
            print(f"Error processing doc {doc_id}: {str(e)}")
            return None

    def score_document(self, query_terms, doc_features):
        """Score document based on MVD"""
        total_score = 0.0

        for term in query_terms:
            if term not in doc_features:
                continue

            features = doc_features[term]
            ritf = features['ritf']
            lrtf = features['lrtf']
            df = features['df']
            idf = math.log(self.collection_size / (df + 1)) if df > 0 else 0

            # Use only Gumbel for now, Frechet removed
            gumbel_score = gumbel_r.cdf(ritf, scale=1.0)
            p = (self.beta * idf) / (1 + self.beta * idf)
            weight = self.alpha * gumbel_score  # Only Gumbel CDF now
            total_score += weight * idf

        return total_score

    def process_query(self, query_text, qid, top_k=1000):
        """Process a single query and rerank documents"""
        self.initialize_collection_stats()
        query_terms = query_text.lower().split()

        hits = self.searcher.search(query_text, k=2000)
        scored_docs = []

        for hit in hits:
            doc_features = self.extract_features(query_terms, hit.docid)
            if not doc_features:
                continue

            score = self.score_document(query_terms, doc_features)
            scored_docs.append((hit.docid, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (docid, score) in enumerate(scored_docs[:top_k], 1):
            results.append(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\tmtc2414")
        return results


def parse_fire_queries(filepath):
    """Parse FIRE query file with <num> and <title> tags"""
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    qid = None
    for line in lines:
        line = line.strip()
        if line.startswith('<num>'):
            qid = ''.join(filter(str.isdigit, line))
        elif line.startswith('<title>') and qid is not None:
            title = line.replace('<title>', '').strip()
            queries.append((qid, title))
    return queries


def main():
    parser = argparse.ArgumentParser(description='MVD Re-ranking with Pyserini')
    parser.add_argument('index_path', help='Path to index')
    parser.add_argument('query_file', help='Path to FIRE topic file')
    args = parser.parse_args()

    retriever = MVDRetriever(args.index_path)
    queries = parse_fire_queries(args.query_file)

    for qid, query in queries: 
        results = retriever.process_query(query, qid)
        for line in results:
            print(line)


if __name__ == '__main__':
    main()

