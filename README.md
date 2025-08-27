 AI-Powered Interactive Heart Disease Risk Prediction with Explainable AI

This project develops a secure, interactive web application that predicts an individual's risk of heart disease using an advanced machine learning model. Unlike traditional tools, it provides transparent, personalized explanations for its predictions, empowering users to understand their risk factors and explore how lifestyle changes could impact their future health.

The application is built as a full-stack solution, featuring a high-performance FastAPI backend, a responsive HTML/CSS/JS frontend, and an XGBoost machine learning model. It leverages cutting-edge Explainable AI (XAI) techniques to demystify the AI's predictions and make them actionable for end-users.



Core Features

The application is built around five key features that work together to create a powerful, user-centric health tool.

1.  Transparent Explanations (SHAP)
    Provides a detailed breakdown of the model's prediction, showing exactly which health parameters (e.g., cholesterol, blood pressure) are contributing to or reducing a user's risk. This demystifies the AI, building trust and providing clear, personalized insights.

2.  Interactive Explanations (What-if Analysis)
    A powerful exploratory tool that allows users to directly manipulate input values (e.g., change cholesterol levels from 200 to 180). The application immediately re-runs the prediction and explanation, allowing users to see how specific, hypothetical changes would affect their risk score. This feature is the foundation for understanding the model's sensitivity.

3.  Personalized Risk Score Dashboard
    Displays the prediction in an easy-to-understand format, including a probability score (0-100%), a confidence level, and a clear risk category (Low, Medium, or High). The dashboard provides an immediate, actionable summary of the user's health status.

4.  Gamification: Lifestyle Improvements
    This feature is a motivational wrapper around "What-if Analysis." Instead of free-form changes, users can select predefined lifestyle goals (e.g., "Reduce Cholesterol," "Increase Physical Activity"). The application simulates the positive impact of these changes and visually demonstrates the resulting reduction in predicted risk, encouraging and rewarding healthier habits.

5.  Healthcare-Ready Touches (PDF Report)
    Allows users to download a comprehensive PDF report that includes their health data, the final prediction, and the SHAP explanation. This document is designed to be easily shareable with healthcare professionals, providing a practical link between the application's insights and real-world medical consultations.



How It Works

The project follows a standard three-tier architecture:

* Frontend: A responsive web interface built with pure HTML, CSS, and JavaScript. It handles user input, displays the prediction dashboard, and visualizes the SHAP and "What-if" results.
* Backend: A FastAPI application that serves as the central API. It handles user authentication, receives health data from the frontend, runs the ML model prediction, generates SHAP values, and handles the PDF report creation.
* ML Model: A pre-trained XGBoost model serialized using joblib. It takes the 13 heart disease features as input to produce a risk prediction.

Data Features

The machine learning model is trained on a dataset containing 13 key health parameters. The user provides values for these parameters through the application's interface.

* age: Age in years
* sex: 1 = Male, 0 = Female
* cp: Chest pain type
    * 0 = Typical angina
    * 1 = Atypical angina
    * 2 = Non-anginal pain
    * 3 = Asymptomatic
* trestbps: Resting blood pressure (in mm Hg)
* chol: Serum cholesterol (in mg/dl)
* fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
* restecg: Resting ECG results
    * 0 = Normal
    * 1 = ST-T wave abnormality
    * 2 = Left ventricular hypertrophy
* thalach: Maximum heart rate achieved during exercise
* exang: Exercise-induced angina (1 = yes, 0 = no)
* oldpeak: ST depression induced by exercise relative to rest
* slope: The slope of the peak exercise ST segment
    * 0 = Upsloping
    * 1 = Flat
    * 2 = Downsloping
* ca: Number of major vessels (0-3) colored by fluoroscopy
* thal: A blood disorder called thalassemia
    * 0 = Normal
    * 1 = Fixed defect (a non-reversible part of the heart)
    * 2 = Reversible defect (a temporary part of the heart that gets worse with exercise)

Technologies Used

* Backend: Python 3.9+, FastAPI, XGBoost, SHAP, NumPy, Pandas
* Frontend: HTML, CSS, JavaScript (with libraries like Chart.js for visualizations)
* Database: SQLite (for local development)
* Project Management: Git, GitHub, Trello
* Deployment (Planned): Docker, Cloud Platform (AWS/Azure)

Getting Started

1.  Clone the repository:
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
2.  Set up the Python virtual environment and install dependencies:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
3.  Run the FastAPI server:
    uvicorn main:app --reload
4.  Open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the application.

Project Status & Timeline

This project is currently in the Development Phase. We are following an agile methodology, with progress tracked on our Trello board.

* Start Date: August 2025
* Planned Completion: April 2026

Contribution & Support

This is an academic  Project. For any questions or inquiries, please contact [Your Email Address].

License

This project is licensed under the MIT License. See the LICENSE file for details.
