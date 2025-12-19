from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def plagiarism_check(text1, text2):
    documents = [text1, text2]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors)[0][1]
    return similarity * 100

# ---- User Input ----
print("PLAGIARISM DETECTOR\n")

text1 = input("Enter first text:\n")
text2 = input("\nEnter second text:\n")

result = plagiarism_check(text1, text2)

print(f"\nPlagiarism Percentage: {result:.2f}%")

if result > 70:
    print("⚠ High Plagiarism Detected")
elif result > 40:
    print("⚠ Moderate Plagiarism Detected")
else:
    print("✅ Low Plagiarism")