# ⚡ lead-scoring-gcp - Simple Lead Scoring With Explainability

[![Download lead-scoring-gcp](https://img.shields.io/badge/Download-Here-brightgreen?style=for-the-badge)](https://github.com/Ryl3/lead-scoring-gcp)

---

## ℹ️ What is lead-scoring-gcp?

lead-scoring-gcp is a tool that helps you score sales leads using machine learning. It shows you why it gives each lead a certain score. This explains where the most important factors are, making it easy to understand. The tool runs as a web application, so you can use it through your browser. It uses FastAPI and Streamlit, two common components for modern apps. The app is hosted on Google Cloud, which means it runs online and does not slow down your computer.

Using lead-scoring-gcp, you can prioritize which leads to contact first. It sorts leads by their chance to convert. This can help sales teams work more efficiently.

---

## 🎯 Features

- Scores sales leads with machine learning.
- Shows clear explanations for scores using SHAP.
- Easy-to-use web interface via Streamlit.
- Runs fast with FastAPI backend.
- Hosted on Google Cloud Platform (Cloud Run).
- Works on Windows, Mac, or Linux (Windows setup detailed below).
- Requires no coding knowledge to use.

---

## 🖥️ System Requirements

To use lead-scoring-gcp on Windows, you need:

- Windows 10 or later.
- Internet connection.
- At least 4 GB of free memory.
- 2 GHz or faster processor.
- A modern web browser (Edge, Chrome, Firefox).
- No additional hardware or software required, as the app runs in the cloud.

---

## 🔽 Download and Install lead-scoring-gcp on Windows

To get started, follow these steps carefully.

### 1. Visit the Download Page

Click the big green button below to open the download page:

[![Download lead-scoring-gcp](https://img.shields.io/badge/Download-Here-blue?style=for-the-badge)](https://github.com/Ryl3/lead-scoring-gcp)

This link takes you to the official repository where you can get all the files you need.

### 2. Getting the Application

Since lead-scoring-gcp runs online, you do not need to install heavy software. Instead, you will run a simple command in Windows PowerShell or Command Prompt that downloads and runs the app in your browser.

To make this easier, the repository contains a ready-made package that sets everything up.

### 3. Download Docker Desktop for Windows (if necessary)

The application uses Docker, a tool for running programs easily. If you do not have Docker installed:

- Open your browser and go to https://www.docker.com/products/docker-desktop/
- Download Docker Desktop for Windows.
- Run the installer and follow the on-screen steps.
- After installing, Docker will ask you to restart your computer. Do that.

> Docker is needed only for running the app locally. If you want to skip this, you can use the cloud-hosted app online instead.

### 4. Clone or Download the Repository Files

On the download page linked above:

- Click on the green **Code** button near the top.
- Select **Download ZIP**.
- Save the file on your desktop or another easy location.
- Right-click the ZIP file and choose **Extract All**.
- Open the extracted folder.

### 5. Start the Application

Now you will run the app on your computer:

- Open the folder with the app files.
- Hold Shift and right-click in a blank space inside the folder.
- Click **Open PowerShell window here**.

In the PowerShell window, type:

```
docker compose up
```

and press Enter.

Docker will download the app components and launch the service.

### 6. Open the Lead Scoring Dashboard

Once the app is running:

- Open your web browser.
- Go to http://localhost:8501

This will open the lead scoring dashboard. Here, you can upload your lead data and get instant scores and explanations.

### 7. Stop the Application

To stop the app:

- Go back to the PowerShell window where the app is running.
- Press **Ctrl + C** to stop the service.
- Close the PowerShell window.

---

## 🚀 Quick Use Guide

After opening the dashboard in your browser:

- Upload your leads data file (such as a CSV).
- Click the **Run** button.
- View the scores and explanations for every lead.
- Use this information to decide which leads to contact first.

The interface shows clear charts and lists, so you can easily understand the results.

---

## ⚙️ How It Works

lead-scoring-gcp combines machine learning and explainability:

- It uses a trained machine learning model to predict how likely a lead is to become a customer.
- It applies SHAP (SHapley Additive exPlanations) to explain how each feature of the lead affects the score.
- The backend is built on FastAPI, a fast and simple web server.
- The frontend uses Streamlit to create interactive dashboards.

Google Cloud Platform Cloud Run hosts the app so it can scale and run smoothly without you needing special servers.

---

## 🛠 Troubleshooting Tips

- If `docker compose up` shows an error, make sure Docker Desktop is running.
- If the browser cannot connect to `localhost:8501`, check that firewall settings allow local connections.
- Restart Docker if the app does not start.
- Use a supported web browser and clear your browser cache if the app doesn’t load correctly.

---

## 🔐 Privacy and Data Handling

Your data stays on your computer when you run the app locally. The cloud-hosted version processes data on Google Cloud. No data is shared beyond your session. Data privacy depends on how you use the app.

---

## 🔗 Useful Links

- Main repository: https://github.com/Ryl3/lead-scoring-gcp
- Docker Desktop (Windows): https://www.docker.com/products/docker-desktop/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/

[![Download lead-scoring-gcp](https://img.shields.io/badge/Download-Here-brightgreen?style=for-the-badge)](https://github.com/Ryl3/lead-scoring-gcp)

---

## 🧰 Additional Information

lead-scoring-gcp supports CSV files with fields like:

- Lead name
- Contact info
- Company size
- Industry type
- Previous interactions

The model uses these to calculate a score from 0 (low chance) to 1 (high chance) for each lead.

Use the SHAP explanations to investigate why a lead’s score is high or low. This helps understand the data better.

---

## 👥 Support

For help, check the repository’s **Issues** page. Open a new issue if you find bugs or have questions.

---

## 📝 License

This project uses the MIT License. You can freely use, modify, and share it.