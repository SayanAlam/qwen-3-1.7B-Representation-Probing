import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model and tokenizer
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval().to(device)

# 10 diverse reasoning prompts
reasoning_prompts = [
    "Every morning, Aya goes for a 9-kilometer-long walk and then stops at a coffee shop afterward. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks at a speed of s + 2 kilometers per hour, the walk takes her 2 hours and 24 minutes, again including t minutes in the coffee shop. Now, suppose Aya walks at a speed of s + 0.5 kilometers per hour. Find the total number of minutes the walk takes her, including the time spent in the coffee shop.",
    "There exist real numbers x and y, both greater than 1, such that logₓ(yˣ) = logᵧ(x⁴ʸ) = 10. Find the value of xy.",
    "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.",
    "Jen enters a lottery by picking 4 distinct numbers from S = {1, 2, 3, ..., 10}. Four numbers are randomly chosen from S. She wins a prize if at least two of her numbers were two of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is m/n, where m and n are relatively prime positive integers. Find m + n.",
    "Rectangles ABCD and EFGH are drawn such that D, E, C, and F are collinear. Also, A, D, H, and G all lie on a circle. If BC = 16, AB = 107, FG = 17, and EF = 184, what is the length of CE?",
    "Consider the paths of length 16 that follow the lines from the lower left corner to the upper right corner on an 8 x 8 grid. Find the number of such paths that change direction exactly four times.",
    "Eight circles of radius 34 are sequentially tangent, and two of the circles are tangent to AB and BC of triangle ABC, respectively. 2024 circles of radius 1 can be arranged in the same manner. The inradius of triangle ABC can be expressed as m/n, where m and n are relatively prime positive integers. Find m + n.",
    "Let triangle ABC have side lengths AB = 5, BC = 9, and CA = 10. The tangents to the circumcircle of triangle ABC at points B and C intersect at point D, and line segment AD intersects the circumcircle again at point P (other than A). The length of AP is equal to m/n, where m and n are relatively prime integers. Find m + n.",
    "Each vertex of a regular octagon is independently colored either red or blue with equal probability. The probability that the octagon can then be rotated so that all of the blue vertices end up at positions where there had been red vertices is m/n, where m and n are relatively prime positive integers. Find m + n.",
    "Let p be the least prime number for which there exists a positive integer n such that n^4 + 1 is divisible by p^2. Find the least positive integer m such that m^4 + 1 is divisible by p^2."
]

# 10 diverse instruction-following prompts
instruction_prompts = [
    "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
    "Write a resume for a fresh high school graduate who is seeking their first job. Make sure to include at least 12 placeholder represented by square brackets, such as [address], [name].",
    "Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)'",
    "Given the sentence \"Two young boys with toy guns and horns.\" can you ask a question? Please ensure that your response is in English, and in all lowercase letters. No capital letters are allowed.",
    "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
    "Write a 2 paragraph critique of the following sentence in all capital letters, no lowercase letters allowed: \"If the law is bad, you should not follow it\". Label each paragraph with PARAGRAPH X.",
    "Write me a resume for Matthias Algiers. Use words with all capital letters to highlight key abilities, but make sure that words with all capital letters appear less than 10 times. Wrap the entire response with double quotation marks.",
    "Write a letter to a friend in all lowercase letters ask them to go and vote.",
    "Write a long email template that invites a group of participants to a meeting, with at least 500 words. The email must include the keywords \"correlated\" and \"experiencing\" and should not use any commas.",
    "Write a story of exactly 2 paragraphs about a man who wakes up one day and realizes that he's inside a video game. Separate the paragraphs with the markdown divider: ***"
]

all_prompts = instruction_prompts + reasoning_prompts
instruction_labels = [1] * len(instruction_prompts) + [0] * len(reasoning_prompts)
reasoning_labels = [0] * len(instruction_prompts) + [1] * len(reasoning_prompts)

# Function to extract hidden states
def extract_features(prompts, batch_size=5, use_mean_pool=True):
    X_layers = None

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.hidden_states

        attention_mask = tokens["attention_mask"].unsqueeze(-1)

        batch_features = []
        for layer in hidden_states:
            if use_mean_pool:
                masked = layer * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1)
                pooled = summed / counts
            else:
                pooled = layer[:, -1, :]
            batch_features.append(pooled.cpu())

        if X_layers is None:
            X_layers = [layer_chunk.clone().detach() for layer_chunk in batch_features]
        else:
            for j in range(len(X_layers)):
                X_layers[j] = torch.cat((X_layers[j], batch_features[j]), dim=0)

    return X_layers

# Extract features from all prompts
X_layers = extract_features(all_prompts)

# Prepare results table
table_data = {
    "Layer": [],
    "Instruction Accuracy": [],
    "Reasoning Accuracy": []
}

# Train and evaluate linear probes layer by layer
for i, X in enumerate(X_layers):
    X = X.numpy()
    X_train, X_test, y_instr_train, y_instr_test = train_test_split(X, instruction_labels, test_size=0.3, random_state=42)
    _, _, y_reasn_train, y_reasn_test = train_test_split(X, reasoning_labels, test_size=0.3, random_state=42)

    clf_instr = LogisticRegression(max_iter=1000)
    clf_reasn = LogisticRegression(max_iter=1000)

    clf_instr.fit(X_train, y_instr_train)
    clf_reasn.fit(X_train, y_reasn_train)

    acc_instr = accuracy_score(y_instr_test, clf_instr.predict(X_test))
    acc_reasn = accuracy_score(y_reasn_test, clf_reasn.predict(X_test))

    table_data["Layer"].append(i)
    table_data["Instruction Accuracy"].append(round(acc_instr, 4))
    table_data["Reasoning Accuracy"].append(round(acc_reasn, 4))

# Convert to pandas DataFrame
df = pd.DataFrame(table_data)

# Display table
print("\nLayer-wise Accuracy Table:\n")
print(df.to_string(index=False))
