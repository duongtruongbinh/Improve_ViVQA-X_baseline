import nltk

try:
    nltk.data.find('corpora/wordnet')
    print("NLTK 'wordnet' resource found.")
except nltk.downloader.DownloadError:
    print("NLTK 'wordnet' resource not found. Downloading...")
    nltk.download('wordnet')
    print("'wordnet' downloaded.")
except LookupError: # For older NLTK versions or different error types
    print("NLTK 'wordnet' resource not found (LookupError). Downloading...")
    nltk.download('wordnet')
    print("'wordnet' downloaded.")


try:
    nltk.data.find('corpora/omw-1.4')
    print("NLTK 'omw-1.4' resource found.")
except nltk.downloader.DownloadError:
    print("NLTK 'omw-1.4' resource not found. Downloading...")
    nltk.download('omw-1.4')
    print("'omw-1.4' downloaded.")
except LookupError:
    print("NLTK 'omw-1.4' resource not found (LookupError). Downloading...")
    nltk.download('omw-1.4')
    print("'omw-1.4' downloaded.")

print("NLTK resource check/download complete.")