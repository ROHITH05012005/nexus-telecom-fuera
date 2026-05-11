from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "Telecom Churn Project Details", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def chapter_title(self, num, title):
        self.set_font("helvetica", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f"{num}. {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 11)
        self.multi_cell(0, 6, body)
        self.ln(6)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

sections = [
    (
        "What dataset is used and where did you get that?",
        "The project uses a synthetically generated Telecom Churn dataset. Instead of relying on a static external dataset (such as the standard IBM Telco dataset), the data is dynamically generated using the `data_generator.py` script provided in the project.\n\nThis script utilizes Python's Numpy and Pandas libraries to generate realistic customer features (such as tenure, monthly charges, and contract type). It applies realistic business logic to simulate churn probabilities based on factors like sudden price shocks, usage spikes, and frequent tech support calls."
    ),
    (
        "What ML algorithms are used?",
        "The core machine learning algorithm used is XGBoost (specifically `XGBClassifier`), which is highly effective for tabular data and churn prediction.\n\nAdditionally, the project integrates SHAP (SHapley Additive exPlanations) via the `TreeExplainer` for Explainable AI (XAI). SHAP is used to break down each prediction and identify the specific features contributing to a customer's churn risk, providing transparent reasoning for the model's outputs."
    ),
    (
        "What evaluation metrics are used?",
        "The model is evaluated using a comprehensive suite of classification metrics:\n\n"
        "- ROC AUC Score (Area Under the Receiver Operating Characteristic Curve) to measure the overall ability to distinguish between churners and non-churners.\n"
        "- Logloss (Logarithmic Loss), which is used as the primary evaluation metric during the XGBoost training phase.\n"
        "- Classification Report metrics, including Precision, Recall, and F1-Score, to evaluate the balance of the predictions.\n"
        "- Confusion Matrix, to visualize True/False Positives and Negatives."
    ),
    (
        "Where did you deploy this?",
        "The application is natively containerized using Docker, with both a `Dockerfile` and a `docker-compose.yml` file included in the repository. This makes it cloud-ready for deployment on any platform that supports Docker containers, such as Render, Heroku, or Hugging Face Spaces.\n\nFor local environments, the project is deployed using a Flask development server (or Gunicorn) accessible at `http://localhost:5000`."
    ),
    (
        "How did you make this website?",
        "The website is built as a responsive web dashboard using a lightweight stack:\n\n"
        "- Backend/API: Built with Python and the Flask framework to handle routing and serve machine learning predictions.\n"
        "- Database: TinyDB (a lightweight, document-oriented database) is used to persist high-risk customer data, prediction history, and model drift metrics in JSON format (`at_risk_customers.json`).\n"
        "- Frontend: The user interface is built using HTML, CSS, and JavaScript, served through Flask's Jinja templates from the `templates` directory. This creates an interactive dashboard for monitoring churn metrics and SHAP explanations in real time."
    )
]

for i, (title, body) in enumerate(sections, 1):
    pdf.chapter_title(i, title)
    pdf.chapter_body(body)

pdf.output("project_details.pdf")
print("PDF generated successfully: project_details.pdf")
