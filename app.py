import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค",
    page_icon="🚢",
    layout="centered",
)

# ---------------------------------------------------------
# Minimal styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: "Helvetica Neue", Arial, sans-serif;
        }
        .main {
            background-color: #FAFAFA;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 720px;
        }
        h1 {
            font-weight: 600;
            font-size: 1.9rem;
            color: #1A1A1A;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #8A8A8A;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }
        div.stButton > button {
            width: 100%;
            background-color: #2C3E50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 0;
            font-size: 1rem;
            font-weight: 500;
        }
        div.stButton > button:hover {
            background-color: #34495E;
            color: white;
        }
        .result-card {
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1.2rem;
        }
        .survive {
            background-color: #EAF6EC;
            border: 1px solid #B7E4C7;
            color: #1E7B3A;
        }
        .not-survive {
            background-color: #FBEAEA;
            border: 1px solid #F1B0B0;
            color: #B3261E;
        }
        .footer {
            text-align: center;
            color: #A0A0A0;
            font-size: 0.85rem;
            margin-top: 3rem;
            border-top: 1px solid #EAEAEA;
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("<h1>🚢 ระบบทำนายการรอดชีวิตผู้โดยสารไททานิค</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Titanic Survival Prediction System</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("titanic_tree.joblib")

model = load_model()
FEATURES = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]


# ---------------------------------------------------------
# Compute accuracy on the public Titanic dataset (for display only)
# ---------------------------------------------------------
@st.cache_data
def compute_accuracy():
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        df["Sex_female"] = (df["Sex"] == "female").astype(int)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        X = df[FEATURES]
        y = df["Survived"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    except Exception:
        return None

accuracy = compute_accuracy()

if accuracy is not None:
    st.metric(label="ความแม่นยำของโมเดล (Accuracy)", value=f"{accuracy * 100:.2f}%")
else:
    st.info("ไม่สามารถคำนวณค่าความแม่นยำได้ในขณะนี้ (ต้องการการเชื่อมต่ออินเทอร์เน็ต)")

st.divider()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("กรอกข้อมูลผู้โดยสาร")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "ชั้นโดยสาร (Pclass)",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: f"ชั้น {x}",
    )
    sex = st.radio("เพศ", options=["หญิง", "ชาย"], horizontal=True)
    age = st.slider("อายุ (ปี)", min_value=0, max_value=80, value=30)

with col2:
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("จำนวนพ่อแม่/ลูกที่ร่วมเดินทาง (Parch)", min_value=0, max_value=10, value=0)

family_size = sibsp + parch + 1
sex_female = 1 if sex == "หญิง" else 0

st.caption(f"ขนาดครอบครัวที่ร่วมเดินทาง (FamilySize): {family_size}")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("ทำนายผล"):
    input_df = pd.DataFrame(
        [[pclass, sex_female, age, fare, family_size]], columns=FEATURES
    )
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card survive">
                <h3>✅ รอดชีวิต</h3>
                <p>ความน่าจะเป็นที่จะรอดชีวิต: {proba[1] * 100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card not-survive">
                <h3>❌ ไม่รอดชีวิต</h3>
                <p>ความน่าจะเป็นที่จะไม่รอดชีวิต: {proba[0] * 100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    "<div class='footer'>พัฒนาโดย นางสาววรรณวิสา สุขาบูรณ์</div>",
    unsafe_allow_html=True,
)
