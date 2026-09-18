import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. HP 15s DATASET
# ==========================================

data = {
    "Model": [
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s",
        "HP 15s"
    ],

    "Budget": [
        45000, 48000, 50000, 52000, 55000,
        47000, 50000, 53000, 49000, 55000
    ],

    "RAM": [
        8, 8, 16, 16, 16,
        8, 16, 16, 8, 16
    ],

    "Storage": [
        512, 512, 512, 512, 512,
        512, 512, 512, 512, 512
    ],

    "Processor": [
        "i5", "i5", "i5", "i5", "i5",
        "i5", "i5", "i5", "i5", "i5"
    ],

    "Recommendation": [
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

# ==========================================
# 2. CONVERT PROCESSOR INTO NUMBER
# ==========================================

df["Processor_Code"] = df["Processor"].map({
    "i5": 5
})

# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df[
    [
        "Budget",
        "RAM",
        "Storage",
        "Processor_Code"
    ]
]

y = df["Recommendation"]

# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 5. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 6. TEST THE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("===================================")
print(" MACHINE LEARNING MODEL")
print("===================================")
print("Accuracy:", accuracy * 100, "%")

# ==========================================
# 7. USER INPUT
# ==========================================

print("\n===================================")
print(" HP 15s LAPTOP SYSTEM")
print("===================================")

user_model = input("Enter Laptop Model: ").strip()

# ==========================================
# 8. CHECK USER INPUT
# ==========================================

if user_model.lower() == "hp 15s":

    # HP 15s details

    budget = 50000
    ram = 16
    storage = 512
    processor_code = 5

    # ======================================
    # 9. ML PREDICTION
    # ======================================

    user_data = [[
        budget,
        ram,
        storage,
        processor_code
    ]]

    prediction = model.predict(user_data)

    # ======================================
    # 10. DISPLAY DETAILS
    # ======================================

    print("\n===================================")
    print("       HP 15s DETAILS")
    print("===================================")

    print("Brand       : HP")
    print("Model       : HP 15s")
    print("Budget      : ₹40,000 - ₹55,000")
    print("Usage       : IT / Software Development / Programming")
    print("Processor   : Intel Core i5")
    print("Generation  : 12th Gen or newer")
    print("Example CPU : Intel Core i5-1235U")
    print("OS          : Windows 11 Home")
    print("RAM         : 16 GB")
    print("Storage     : 512 GB SSD")
    print("Graphics    : Intel Iris Xe / Integrated")
    print("Display     : 15.6-inch (39.6 cm)")
    print("Resolution  : Full HD 1920 x 1080")
    print("Display Type: Anti-glare, Micro-edge")
    print("Colour      : Natural Silver")
    print("Weight      : Around 1.7 kg")
    print("Speakers    : Dual Speakers")
    print("Battery     : 3-cell, around 41 Wh")
    print("Charger     : 45 W")

    # ======================================
    # 11. ML RECOMMENDATION
    # ======================================

    print("\n===================================")

    if prediction[0] == 1:
        print("ML Recommendation : RECOMMENDED")
    else:
        print("ML Recommendation : NOT RECOMMENDED")

    print("===================================")

else:
    print("\nnot available.")