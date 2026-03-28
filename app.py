# COMPLETE PROTEIN SOLUBILITY PREDICTOR - RUN THIS ENTIRE CODE IN ONE CELL
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🔬 PROTEIN SOLUBILITY PREDICTOR")
print("=" * 60)

# STEP 1: Create dataset
print("\n📊 STEP 1: Creating dataset...")
data = {
    'sequence': [
        # SOLUBLE proteins (1)
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF",
        "MKTIIALSYIFCLVFADKTDVDTLVLEGSDGRPKRILTISQDPKVPK",
        "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDG",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALV",
        # INSOLUBLE proteins (0)
        "MSTAGKQRPVKVSRSEEIMKLTYKEKVAELQEETLKKIKESLKVEQE",
        "MLRAALLLLLLLLPLLAAPAAAVEGGGEARVKIGYNYARTWGGVTA",
        "MKKLLPILTGLPLFLLVLASVFSQIVQGNQTHRDRDPTPFETADKPG",
        "MVSVWGPYGGSPSTFSTLGVGGSGVGATVGGALGGGLAAASLAPVK",
    ],
    'solubility': [1, 1, 1, 1, 0, 0, 0, 0]
}

df = pd.DataFrame(data)
print(f"✅ Dataset created: {len(df)} proteins")
print(f"   Soluble: {df['solubility'].sum()}")
print(f"   Insoluble: {len(df) - df['solubility'].sum()}")

# STEP 2: Convert sequences to features
print("\n🔢 STEP 2: Converting sequences to numbers...")
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

def get_amino_acid_composition(sequence):
    """Count percentage of each amino acid in the sequence"""
    sequence = sequence.upper()
    length = len(sequence)
    composition = []
    for aa in amino_acids:
        count = sequence.count(aa)
        percentage = (count / length) * 100
        composition.append(percentage)
    return composition

features = []
for seq in df['sequence']:
    comp = get_amino_acid_composition(seq)
    features.append(comp)

X = np.array(features)
y = df['solubility'].values
print(f"✅ Features created: {X.shape[0]} proteins with {X.shape[1]} features each")

# STEP 3: Train the model
print("\n🤖 STEP 3: Training AI model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"✅ Model trained on {len(X_train)} proteins")

# STEP 4: Test the model
print("\n📈 STEP 4: Testing model accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# Show test results
print("\n🔬 Test Results:")
for i in range(len(X_test)):
    pred = "Soluble 💧" if y_pred[i] == 1 else "Insoluble ⚠️"
    actual = "Soluble 💧" if y_test[i] == 1 else "Insoluble ⚠️"
    confidence = max(model.predict_proba([X_test[i]])[0])
    status = "✅" if y_pred[i] == y_test[i] else "❌"
    print(f"  {status} Protein {i+1}: Predicted={pred}, Actual={actual}, Confidence={confidence:.1%}")

# STEP 5: Create prediction function
print("\n🎯 STEP 5: Creating prediction function...")

def predict_protein(sequence):
    """Predict if a protein sequence is soluble or insoluble"""
    sequence = sequence.upper().strip()

    # Validation
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(char in valid_chars for char in sequence):
        invalid = [c for c in sequence if c not in valid_chars]
        return {"error": f"Invalid amino acids: {invalid}"}

    if len(sequence) < 5:
        return {"error": "Sequence too short (minimum 5 amino acids)"}

    # Get features and predict
    features = get_amino_acid_composition(sequence)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]

    result = "Soluble 💧" if prediction == 1 else "Insoluble ⚠️"
    confidence = probabilities[prediction]

    # Get top 3 amino acids
    comp_dict = dict(zip(amino_acids, features))
    top_3 = sorted(comp_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "prediction": result,
        "confidence": f"{confidence:.1%}",
        "length": len(sequence),
        "top_amino_acids": top_3
    }

# Test the function
test_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
result = predict_protein(test_seq)
print(f"✅ Prediction function ready!")
print(f"\n   Test prediction on known soluble protein:")
print(f"   Sequence: {test_seq[:20]}...")
print(f"   Result: {result['prediction']}")
print(f"   Confidence: {result['confidence']}")

# STEP 6: Feature importance visualization
print("\n📊 STEP 6: Analyzing important features...")
feature_importance = model.feature_importances_

plt.figure(figsize=(12, 5))
sorted_idx = np.argsort(feature_importance)[::-1][:10]
top_features = [list(amino_acids)[i] for i in sorted_idx]
top_importance = feature_importance[sorted_idx]

plt.bar(top_features, top_importance, color='lightblue', edgecolor='navy')
plt.xlabel('Amino Acid', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title('Top 10 Most Important Amino Acids for Solubility Prediction', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n📝 Interpretation:")
for aa, imp in zip(top_features, top_importance):
    if aa in 'RKDENQ':
        print(f"  {aa}: {imp:.3f} → Hydrophilic (increases solubility)")
    elif aa in 'AILMFWYV':
        print(f"  {aa}: {imp:.3f} → Hydrophobic (decreases solubility)")
    else:
        print(f"  {aa}: {imp:.3f}")

# STEP 7: Interactive predictor
print("\n" + "=" * 60)
print("🎯 INTERACTIVE PROTEIN PREDICTOR")
print("=" * 60)
print("Enter any protein sequence to predict solubility")
print("Example: MVLSPADKTNVKAAWG")
print("Type 'quit' to exit")
print("-" * 60)

# Create interactive loop
while True:
    print("\n📝 Enter protein sequence:")
    user_input = input().strip()

    if user_input.lower() == 'quit':
        print("\n👋 Goodbye! Thanks for using Protein Predictor!")
        break

    if not user_input:
        print("⚠️ Please enter a sequence!")
        continue

    result = predict_protein(user_input)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        print("   Use only letters: A C D E F G H I K L M N P Q R S T V W Y")
    else:
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   {'=' * 40}")
        print(f"   Sequence length: {result['length']} amino acids")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   {'=' * 40}")
        print(f"\n   Top 3 amino acids in sequence:")
        for aa, perc in result['top_amino_acids']:
            print(f"     • {aa}: {perc:.1f}%")

        # Give simple advice
        if "Soluble" in result['prediction']:
            print(f"\n   💡 Tip: This protein should dissolve well in water!")
        else:
            print(f"\n   💡 Tip: This protein may be difficult to work with in solution")

print("\n✅ Program complete!")
