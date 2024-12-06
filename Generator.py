import random
import numpy as np
from Reasoner import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json

# List of nouns (50 elements)
nouns = [
    "Jack", "Mike", "The sky", "The pig", "Stanford University", "Sarah",
    "The cat", "The car", "The book", "Alice", "The house", "The dog",
    "Emma", "The mountain", "The river", "Charlie", "The tree",
    "The computer", "The phone", "Henry", "The ocean", "Lucy", "The sun",
    "The moon", "Ethan", "The train", "The city", "The forest",
    "Mia", "The sandwich", "The globe", "The wall", "The bridge",
    "Liam", "The plane", "The shop", "The garden", "The chef",
    "The teacher", "The artist", "The musician", "James", "The cloud",
    "The bakery", "The stadium", "The zoo", "Ava", "The festival",
    "Sophie", "The pumpkin", "The fountain", "The playground", "Lucas", "Jianzhi", "Chuan"
]

# List of verbs (50 elements)
verbs = [
    "is blue", "is happy", "goes to the hospital", "is fast", "is interesting",
    "is fluffy", "jumps high", "swims well", "reads books", "paints pictures",
    "sings beautifully", "runs quickly", "dances gracefully", "is tall",
    "is loud", "is quiet", "makes pizza", "cooks well", "draws maps",
    "plays games", "writes stories", "teaches math", "is colorful",
    "carries bags", "becomes famous", "explores cities", "is friendly",
    "grows plants", "hosts parties", "enjoys music", "travels far",
    "is magical", "rides bikes", "plays football", "hikes mountains",
    "fishes in lakes", "collects stamps", "creates art", "solves puzzles",
    "plays instruments", "takes pictures", "is funny", "does yoga",
    "volunteers", "runs marathons", "is a doctor", "is a scientist",
    "is a chef", "is an engineer", "is an artist"
]

# Function to generate k Fact objects
def generate_facts(k, nouns, verbs):
    facts = []
    for i in range(k):
        noun = random.choice(nouns)
        verb = random.choice(verbs)
        fact = Fact(i + 1, f"{noun} {verb}.")
        facts.append(fact)
    return list(set(facts))


# Function to generate k rules from facts
def generate_rules_from_facts(k, selected_facts, facts, n_of_premises=2):
    rules = []
    for _ in range(k):
        # Randomly determine the number of premises for this rule (between 1 and n_of_premises)
        num_premises = random.randint(1, n_of_premises)
        premises = random.sample(selected_facts, min(num_premises, len(facts)))  # Randomly select premises
        # Select one fact as the conclusion from the available facts
        conclusion = random.choice([i for i in facts if i not in premises])

        # Create a Rule object
        rule = Rule(premises, conclusion)
        rules.append(rule)
    return list(set(rules))


def generate_facts_and_rules(k1, k2, k3, n_nouns=4, n_verbs=4, n_of_premises=2):
    facts = generate_facts(k1, random.sample(nouns, n_nouns), random.sample(verbs, n_verbs))
    selected_facts = random.sample(facts, min(k3, len(facts)))  # Select up to k3 facts
    non_initial_facts = [fact for fact in facts if fact not in selected_facts]
    rules = generate_rules_from_facts(random.randint(k2-1, k2), selected_facts, non_initial_facts, n_of_premises=n_of_premises) + generate_rules_from_facts(random.randint(k2-1, k2), facts, non_initial_facts, n_of_premises=n_of_premises)
    return selected_facts, rules, facts




# print(string_from_facts_and_rules(selected_facts, rules))

def generate_a_datapoint(k1, k2, k3, n_of_premises=1):
    selected_facts, rules, facts = generate_facts_and_rules(k1, k2, k3, n_of_premises=n_of_premises)
    return string_from_facts_and_rules(selected_facts, rules, facts)

def generate_wrapper(args):
    return generate_a_datapoint(*args)

def parallel_generate(num_samples, num_processes=None):
    num_processes = num_processes or cpu_count()
    args = [(k1, k2, k3, 1)] * num_samples
    with Pool(processes=num_processes) as pool:
        results = pool.map(generate_wrapper, args)
    return results
    

if __name__ == "__main__":
    k1 = 11
    k2 = 5
    k3 = 3

    with open('data.json', "r") as f:
        data = json.load(f)

    dataset = parallel_generate(10000000)

    depth_5 = [i for i in dataset if i["depth"] > 4]
    depth_4 = [i for i in dataset if i["depth"] == 4]
    depth_3 = [i for i in dataset if i["depth"] == 3]
    depth_2 = [i for i in dataset if i["depth"] == 2]
    depth_1 = [i for i in dataset if i["depth"] == 1]


    depth_4_true = [i for i in depth_4 if i["label"]=="True"]
    depth_4_false = [i for i in depth_4 if i["label"]=="False"]
    depth_4_false = random.sample(depth_4_false, len(depth_4_true))


    depth_3_true = [i for i in depth_3 if i["label"]=="True"]
    depth_3_true = random.sample(depth_3_true, min(250, len(depth_3_true)))
    depth_3_false = [i for i in depth_3 if i["label"]=="False"]
    depth_3_false = random.sample(depth_3_false, min(250, len(depth_3_false)))

    print(len(depth_3_true), len(depth_4_true), len(depth_5))
    
    depth_2_true = [i for i in depth_2 if i["label"]=="True"]
    depth_2_true = random.sample(depth_2_true, min(250, len(depth_2_true)))
    depth_2_false = [i for i in depth_2 if i["label"]=="False"]
    depth_2_false = random.sample(depth_2_false, min(250, len(depth_2_false)))

    depth_1_true = [i for i in depth_1 if i["label"]=="True"]
    depth_1_true = random.sample(depth_1_true, min(250, len(depth_1_true)))
    depth_1_false = [i for i in depth_1 if i["label"]=="False"]
    depth_1_false = random.sample(depth_1_false, min(250, len(depth_1_false)))

    dataset = depth_5 + depth_4_false + depth_4_true + depth_3_false + depth_3_true + depth_2_false + depth_2_true + depth_1_true + depth_1_false
    with open("data.json", "w") as f:
        json.dump(dataset, f)

    print("OK!")