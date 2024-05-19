import re
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def load_paper(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def chunk_text(docs, chunk_size=300):
    chunks = []
    for page in docs:
        content = page.page_content
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = []
        current_length = 0
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current_length + paragraph_length > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    return chunks


def segment_chunks(chunks, num_segments=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    kmeans = KMeans(n_clusters=num_segments, random_state=42).fit(X)
    labels = kmeans.labels_
    segmented_texts = {i: [] for i in range(num_segments)}
    for label, chunk in zip(labels, chunks):
        segmented_texts[label].append(chunk)
    return segmented_texts


def aggregate_segments(segmented_texts, min_length=500):
    aggregated_sections = []
    for key in sorted(segmented_texts.keys()):
        segment = segmented_texts[key]
        aggregated_section = " ".join(segment)
        aggregated_sections.append(aggregated_section)
    return aggregated_sections


# Example usage
if __name__ == "__main__":
    file_path = "./assets/Week8_CC536_Cybersecurity.pdf"
    docs = load_paper(file_path)
    chunks = chunk_text(docs)
    segmented_texts = segment_chunks(chunks)
    final_sections = aggregate_segments(segmented_texts)

    for i, section in enumerate(final_sections):
        print(f"Section {i + 1}:\n{section}\n{'-' * 20}")
