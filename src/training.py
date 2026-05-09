import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# قراءة البيانات
df = pd.read_excel("makkah_pollution.xlsx")


def train_model(df):
    # تحويل النصوص إلى أرقام
    le_pollution = LabelEncoder()
    le_suffering = LabelEncoder()

    df["أنواع التلوث الهوائي"] = le_pollution.fit_transform(df["أنواع التلوث الهوائي"])
    df["عانت من التلوث"] = le_suffering.fit_transform(df["عانت من التلوث"])

    # Features
    X = df[["السنة", "أنواع التلوث الهوائي", "التوزيع النسبي"]]

    # Target
    y = df["عانت من التلوث"]

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # إنشاء الموديل
    model = DecisionTreeClassifier()

    # تدريب الموديل
    model.fit(X_train, y_train)

    print("Model Trained Successfully")

    return model, X_test, y_test
