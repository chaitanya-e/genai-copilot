from typing import List

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, 0)
    return chunks

# Explanation
# text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
# chunks = chunk_text(text, chunk_size=4, overlap=2)
# print(chunks)

# words = ['Word1', 'Word2', 'Word3', 'Word4', 
#          'Word5', 'Word6', 'Word7', 'Word8', 
#          'Word9', 'Word10']

# Step 2 – First chunk (start = 0):
# end = min(0 + 4, 10) = 4
# Take words [0:4] → 'Word1 Word2 Word3 Word4'
# Append to chunks:
# ['Word1 Word2 Word3 Word4']
# New start = max(4 - 2, 0) = 2

# [
#   'Word1 Word2 Word3 Word4',
#   'Word3 Word4 Word5 Word6',
#   'Word5 Word6 Word7 Word8',
#   'Word7 Word8 Word9 Word10'
# ]
