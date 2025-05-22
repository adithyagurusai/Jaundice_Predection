
# 🩺 Jaundice Prediction Using Machine Learning

This project aims to **predict the risk of jaundice** using a machine learning model trained on clinical parameters. It is a proactive healthcare initiative to help identify high-risk individuals, especially newborns and patients with underlying health conditions, **without relying on liver enzyme values**.

---

## 📌 Features

- Predicts jaundice risk based on:
  - Total Bilirubin
  - Direct Bilirubin
  - Indirect Bilirubin
  - Albumin-Globulin (A/G) Ratio
  - Age
  - Sex
- Excludes liver enzymes for a non-invasive prediction approach.
- Designed with scalability for healthcare systems and integration with mobile/web apps.

## 🧠 Machine Learning

- **Algorithms Used**: [e.g., Random Forest, Logistic Regression, etc.]
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Tools & Libraries**:
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib / Seaborn (for visualizations)

## 🗂️ Project Structure


jaundice\_prediction/
├── data/                   # Dataset files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Core logic
├── streamlit\_app/          # Streamlit UI files
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

## ⚙️ Setup Instructions

1. **Clone the repository**:
   bash
   git clone https://github.com/adithyagurusai/Jaundice_Predection.git
   cd Jaundice_Predection

2. **Create and activate a virtual environment**:


   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**:

   pip install -r requirements.txt


4. **Run the Streamlit app**:

   cd streamlit_app
   streamlit run app.py


## 📊 Results

* Model Accuracy: 90%
* Precision / Recall: **XX% / XX%**
* ROC-AUC Score: **XX**

## 🚀 Future Work

* Deploy via Streamlit Cloud
* Connect to Firebase or cloud storage
* Mobile integration (React Native / Kotlin)
* Expand dataset and improve generalization


## 👨‍💻 Authors

* Adithya Guru Sai K – [GitHub](https://github.com/adithyagurusai)
* Adduri Sathwik
* Adina Rohith Reddy

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## ❤️ Acknowledgements

* Scikit-learn team
* OpenML & healthcare data contributors
* Faculty guidance and support
