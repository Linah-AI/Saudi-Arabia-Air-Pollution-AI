import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from src import preprocessing, training, evaluation

def main():
    print("Starting Makkah Air Pollution Risk Classification Pipeline...")
    print("-" * 60)

    # 1. Data Preprocessing Phase
    # ---------------------------------
    data_path = "data/makkah_pollution.xlsx"
    clean_df = preprocessing.load_and_clean_data(data_path)
    
    print("Data preprocessing completed.")
    print("\nSample of the cleaned dataset (first 5 rows):")
    print(clean_df.head())
    print("-" * 60)

    # 2. Model Training Phase
    # ---------------------------------
    model, X, y = training.train_model(clean_df)
    print(f"Model configured with {model.get_n_leaves()} decision leaves.")
    print("-" * 60)

   # 3. Model Visualization Phase
    # ---------------------------------
    print("Generating Decision Tree visualization...")
    
    # تثبيت الأسماء باللغة الإنجليزية
    feature_names = ["Year", "Pollution Type"]
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
    )

    # 4. Evaluation Phase (Cross-Validation)
    # ---------------------------------
    print("Initiating Cross-Validation Evaluation...")
    evaluation.evaluate_with_cv(model, X, y, labels=model.classes_)

    print("\n" + "=" * 60)
    print("Pipeline execution finished successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()