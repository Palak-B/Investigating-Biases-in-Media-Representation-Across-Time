import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import words
nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load pre-trained GloVe embeddings
def load_embeddings(file_path):
    return KeyedVectors.load_word2vec_format(file_path, binary=False)

# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Compute the differential association between two sets of words
def differential_association(word, A, B, embeddings):
    mean_cosine_A = np.mean([cosine_similarity(embeddings[word], embeddings[a]) for a in A])
    mean_cosine_B = np.mean([cosine_similarity(embeddings[word], embeddings[b]) for b in B])
    return mean_cosine_A - mean_cosine_B

# Compute WEAT statistic
def weat_score(X, Y, A, B, embeddings):
    score_X = sum([differential_association(x, A, B, embeddings) for x in X])
    score_Y = sum([differential_association(y, A, B, embeddings) for y in Y])
    return score_X - score_Y

# Compute effect size
def effect_size(X, Y, A, B, embeddings):
    mean_X = np.mean([differential_association(x, A, B, embeddings) for x in X])
    mean_Y = np.mean([differential_association(y, A, B, embeddings) for y in Y])
    std_dev = np.std([differential_association(w, A, B, embeddings) for w in X+Y])
    return (mean_X - mean_Y) / std_dev

def compute_p_value(X, Y, A, B, embeddings, iterations=10000):
    observed_weat = weat_score(X, Y, A, B, embeddings)
    combined = X + Y
    count = 0
    for _ in range(iterations):
        random.shuffle(combined)
        new_X = combined[:len(X)]
        new_Y = combined[len(X):]
        permuted_weat = weat_score(new_X, new_Y, A, B, embeddings)
        if permuted_weat > observed_weat:
            count += 1
    p_value = count / iterations
    return p_value    

def display_gender_bias(X, Y, A, B, embeddings):
    male_vec = np.mean([embeddings[word] for word in A if word in embeddings], axis=0)
    female_vec = np.mean([embeddings[word] for word in B if word in embeddings], axis=0)
    
    for occupation in X + Y:
        if occupation in embeddings:
            occ_vec = embeddings[occupation]
            male_similarity = cosine_similarity(occ_vec, male_vec)
            female_similarity = cosine_similarity(occ_vec, female_vec)
            print(f"{occupation}: Male bias={male_similarity:.4f}, Female bias={female_similarity:.4f}")

def plot_gender_bias(X, Y, A, B, embeddings):
    male_biases = []
    female_biases = []
    labels = []

    # Compute the average vector for male and female attributes
    male_vec = np.mean([embeddings[word] for word in A if word in embeddings], axis=0)
    female_vec = np.mean([embeddings[word] for word in B if word in embeddings], axis=0)

    # Calculate biases for each occupation
    for occupation in X + Y:
        if occupation in embeddings:
            occ_vec = embeddings[occupation]
            male_bias = cosine_similarity(occ_vec, male_vec)
            female_bias = cosine_similarity(occ_vec, female_vec)
            male_biases.append(male_bias)
            female_biases.append(female_bias)
            labels.append(occupation)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(male_biases, female_biases, alpha=0.6)

    # Annotate some points
    for i, label in enumerate(labels):
        if i % 5 == 0:  # Annotate every 5th label for clarity
            plt.annotate(label, (male_biases[i], female_biases[i]))

    plt.axvline(x=0, color='grey', linestyle='--')
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.title('Scatter Plot of Gender Bias by Occupation')
    plt.xlabel('Cosine Similarity to Male Vector')
    plt.ylabel('Cosine Similarity to Female Vector')
    plt.grid(True)
    plt.show()

         

X = [
    'janitor', 'bailiff', 'geologist', 'athlete', 'physicist', 
    'blacksmith', 'psychologist', 'mathematician', 'surveyor', 'mechanic', 
    'laborer', 'postmaster', 'broker', 'chemist', 'scientist', 'carpenter', 
    'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'mason', 
    'architect', 'collector', 'operator', 'surgeon', 'driver', 'engineer', 
    'lawyer', 'clergy', 'physician', 'farmer', 'manager', 'guard', 'smith', 
    'official', 'police', 'doctor', 'professor', 'judge', 'author', 'soldier','astronomer',
    'pharmacist', 'anatomist', 'linguist', 'mineralogist', 'comedian', 'swimmer',
    'entertainer', 'politician', 'astrologer', 'theoretician', 'philosopher', 'technician', 'deputy', 'sculptor', 'businessman',
    'industrialist', 'historian' ,'critic', 'financier', 'president', 'actor', 'singer', 'filmmaker', 'cartographer', 'botanist',
    "statistician", "statistician",  'captaincy', 'chauffeur', 'divers'
]

Y = [
    'midwife', 'photographer', 'shoemaker', 'cashier', 'dancer', 'housekeeper', 
    'accountant', 'gardener', 'dentist', 'weaver', 'tailor', 'designer', 
    'economist', 'librarian', 'attendant', 'clerical', 'musician', 'porter', 
    'baker', 'administrator', 'nurse', 'cook', 'retired', 'sales', 'clerk', 
    'artist', 'secretary', 'teacher', 'student', 'folklorist' ,'kitchener', 'zoologist',
    'stripper', 'educator', 'journalist', 'writer', 'poet', 
    'essayist', 'hymnist', 'philologist', 'midwife', 'caretaker', 'bissette', 'songwriter',
    'designers', 'actress', 'performer', 'painter', 'sculptor', 'choreographer', 'ballet', 'paymaster', 'quartermaster'
]

# Attribute Sets
A = ['man', 'male', 'boy', 'he', 'his', 'brother']
B = ['woman', 'female', 'girl', 'she', 'her', 'sister']

# Load your embeddings (replace 'path_to_embeddings' with your file path)
embeddings = KeyedVectors.load_word2vec_format('/Users/nitish/Downloads/vectors.word2vec.txt', binary=False)

# Calculate WEAT score and effect size
weat_result = weat_score(X, Y, A, B, embeddings)
effect_size_result = effect_size(X, Y, A, B, embeddings)
p_value = compute_p_value(X, Y, A, B, embeddings)

display_gender_bias(X, Y, A, B, embeddings)
plot_gender_bias(X, Y, A, B, embeddings)   
print("WEAT Score:", weat_result)
print("Effect Size:", effect_size_result)
print("P-Value:", p_value)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get a list of English words to filter meaningful words
english_vocab = set(words.words())
# Filter words from the model's vocabulary
filtered_words = {word for word in embeddings.key_to_index if word in english_vocab}
# Example: further filter to get nouns or adjectives using NLTK's POS tagger (this requires the words to be meaningful in English)
nltk.download('averaged_perceptron_tagger')
word_pos_tags = nltk.pos_tag(list(filtered_words))
nouns = {word for word, pos in word_pos_tags if pos.startswith('NN')}  # All nouns
adjectives = {word for word, pos in word_pos_tags if pos.startswith('JJ')}  # All adjectives
# Print some of the filtered words
print("Sample nouns:", list(nouns)[:10])
print("Sample adjectives:", list(adjectives)[:10])

stop_words = set(stopwords.words('english'))
print("Stop Words:", stop_words)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def filter_words_from_embeddings(embeddings, word_list):
    """Filter and return words that exist in the embeddings."""
    return [word for word in word_list if word in embeddings.key_to_index]

# Example occupation and gender lists
occupations = [
    "janitor", "statistician", "midwife", "bailiff", "auctioneer", "photographer", 
    "geologist", "shoemaker", "athlete", "cashier", "dancer", "housekeeper", 
    "accountant", "physicist", "gardener", "dentist", "weaver", "blacksmith", 
    "psychologist", "supervisor", "mathematician", "surveyor", "tailor", "designer", 
    "economist", "mechanic", "laborer", "postmaster", "broker", "chemist", 
    "librarian", "attendant", "clerical", "musician", "porter", "scientist", 
    "carpenter", "sailor", "instructor", "sheriff", "pilot", "inspector", "mason", 
    "baker", "administrator", "architect", "collector", "operator", "surgeon", 
    "driver", "painter", "conductor", "nurse", "cook", "engineer", "retired", 
    "sales", "lawyer", "clergy", "physician", "farmer", "clerk", "manager", 
    "guard", "artist", "smith", "official", "police", "doctor", "professor", 
    "student", "judge", "teacher", "author", "secretary", "soldier"
]
gender_terms = ["man", "woman", "male", "female", "boy", "girl", "he", "she", "his", "her", "sister", "brother"]
# Assuming 'model' is your pre-loaded embeddings model (e.g., loaded via gensim)
filtered_occupations = filter_words_from_embeddings(embeddings, occupation_list)
filtered_gender_terms = filter_words_from_embeddings(embeddings, gender_terms)
print("Filtered Occupations:", filtered_occupations)
print("Filtered Gender Terms:", filtered_gender_terms)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

