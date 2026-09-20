# EV Battery Mission Control — Manual Run Guide

This guide provides step-by-step instructions for manually demonstrating the EV Battery Mission Control project. It is based entirely on the current, implemented functionality.

---

## 1. PROJECT STARTUP

Follow these exact steps to start the application:

1. Open your terminal.
2. Navigate to the project folder:
   `cd C:\Users\samar\.gemini\antigravity-ide\scratch\ev-battery-mission-control`
3. Activate your virtual environment (if you created one for this project).
4. Install dependencies (only if you haven't already):
   `pip install -r requirements.txt`
5. Start the Streamlit application by running the exact command:
   `streamlit run app/main.py`
6. The application will start and provide a local URL. You should open your browser to the expected localhost URL:
   `http://localhost:8501`
7. **What you should see:** A professional engineering dashboard titled "EV Battery Mission Control" with a clean, light-themed industrial interface (avoiding neon AI glows).
8. **Terminal errors to watch for:** If you see `Models not found. Run: python scripts/train_model.py` in the UI or terminal, it means the pre-trained ML models are missing. You must run `python src/train_model.py` first.

---

## 2. PROJECT NAVIGATION

Use the dropdown menu in the left sidebar to navigate between the following pages:

- **Mission Control**: High-level overview of battery telemetry, predicted temperature, thermal risk, and LangGraph recommendations. (No manual input required).
- **Digital Twin**: 2D spatial heatmap of the battery pack. (Requires input for what-if scenarios).
- **Thermal Analysis**: Live simulation tracking time-series thermal behavior. (Requires clicking 'Start Live Simulation').
- **What-If Simulator**: Scenario testing to compare baseline vs. hypothetical parameters. (Requires adjusting sliders and clicking 'Run What-If Simulation').
- **Dataset Explorer**: Raw synthetic telemetry dataset viewer. (No manual input required).
- **Model Performance**: Evaluation metrics and feature importance charts for the XGBoost model. (No manual input required).
- **GenAI Generator**: cGAN interface for generating synthetic telemetry based on risk conditions. (Requires selecting a condition and number of samples).
- **VAE Generator**: Latent space compression demonstration. (Requires adjusting signal parameters and clicking 'Encode & Reconstruct').
- **XAI**: Explainable AI interface featuring SHAP and LIME. (Requires clicking 'Generate' buttons).
- **Responsible AI**: Static documentation of system limitations and safety boundaries. (No manual input required).
- **RAG Assistant**: Conversational interface for querying the knowledge base. (Requires typing a query).
- **Prompt Lab**: Engineering console to execute LLM prompts against the RAG pipeline. (Requires typing a prompt and clicking 'Execute Prompt Pipeline').
- **AI Agent Decision**: LangGraph deterministic safety agent workflow demonstration. (Requires clicking 'Simulate thermal stress').

---

## 3. MISSION CONTROL

**How to demonstrate:**
No manual input is required.

**What to inspect:**
- **Battery Temperature**: The actual telemetry temperature.
- **Predicted Temperature**: The XGBoost prediction for maximum battery temperature.
- **Thermal Risk**: The rule-based classification (NORMAL, CAUTION, HIGH, CRITICAL).
- **Hotspot Risk**: The localized hotspot variance metric (NORMAL or CAUTION).
- **XGBoost Status**: Prediction active status.
- **MLP Status**: Secondary model tracking status.
- **Isolation Forest Status**: Anomaly detection status (NORMAL or ANOMALY).
- **Risk Evaluator**: Currently assessed risk category.
- **LangGraph Recommendation**: The autonomous agent's prescribed cooling action.
- **Thermal Forecast**: The 15-step projection chart showing historical and predicted trajectory.

**Successful run:**
A successful run shows all metrics loaded correctly, the forecast chart rendered, and a logically consistent recommendation (e.g., if risk is NORMAL, recommendation should be to maintain cooling). 

**Screenshot:**
Capture the entire Mission Control overview, ensuring the anomaly badges and forecast chart are visible.

---

## 4. DIGITAL TWIN

**A. Normal Test**
- **Input values:** Battery Current: `5.0 A`, Ambient Temperature: `25.0 °C`.
- **Where to enter:** Sliders under "Operating Parameters".
- **Action:** Click "Update Digital Twin".
- **Observe:** The 2D thermal grid updates to show a stable, low-temperature profile. Record the cell temperatures, Pack Max Temp, Hotspot Risk, Thermal State, and Cell Variance.

**B. Stress Test**
- **Input values:** Battery Current: `15.0 A`, Ambient Temperature: `45.0 °C`, Discharge Rate: `5.0 C`, Coolant Flow: `0.005 kg/s`.
- **Why these represent stress:** High current and discharge rate maximize internal heat generation, high ambient limits external dissipation, and low coolant flow minimizes active cooling.
- **Action:** Click "Update Digital Twin".
- **Observe:** Record the new cell temperatures. Verify that the Pack Max Temperature EXACTLY matches the text inside the hottest cell in the 2D grid. Record the Hotspot Risk, Thermal State (which should now be elevated), and Cell Variance.

---

## 5. THERMAL ANALYSIS

**Steps:**
1. Under "Simulation Profile", select **NORMAL**.
2. Click **Start Live Simulation**.
3. Observe the charts updating dynamically.
4. **Wait for completion:** The simulation will run for exactly 50 steps.
5. **Observe successful completion:** The running indicator will change to "✅ COMPLETE", the loop will stop, and the charts will remain visible on screen.

**What to observe during the run:**
- The telemetry charts (Current, Voltage, Battery Temperature, Ambient Temperature, Coolant Flow, Inlet Temperature).
- The Temperature Forecast graph advancing step-by-step.
- The Risk Status badges.

*(If you wish to test stress, select HIGH LOAD or THERMAL STRESS from the profile dropdown, click Stop (if running), and click Start Live Simulation again).*

---

## 6. WHAT-IF SIMULATOR

**TEST A — NORMAL**
- **Input values:** Use the default slider values (e.g., Current: 5.0A, Ambient: 28.0°C, Coolant: 0.020 kg/s, Inlet: 23.0°C, SOC: 60.0%, Discharge: 1.5C).
- **Action:** Click "Run What-If Simulation".
- **Observe:** Note the Baseline Prediction and the What-If Prediction. They should be identical or very close. Record the temperature difference and risk state.

**TEST B — STRESS**
- **Input values:** Battery Current: `15.0 A`, Ambient Temperature: `45.0 °C`, Discharge Rate: `5.0 C`, Coolant Flow Rate: `0.005 kg/s`.
- **Action:** Click "Run What-If Simulation".
- **Observe:** Record the actual output values. The What-If Prediction should be significantly higher than the Baseline, and the Impact Arrow should be red, indicating a large positive temperature difference. Record the new, elevated Thermal Risk state.

---

## 7. DATASET EXPLORER

**How to demonstrate:**
Navigate to the "Dataset Explorer" page.

**What to inspect:**
- Check the top metric cards to verify the total number of rows and features.
- Inspect the raw data table to see the 14 `FEATURE_COLUMNS`.
- Note the target column: `max_battery_temperature_C`.
- Inspect the interactive correlation heatmap and feature distribution histograms.

---

## 8. MODEL PERFORMANCE

**What to check:**
- Navigate to the page. It evaluates the XGBoost predictive model.
- **Metrics to record:** Record the displayed MAE (Mean Absolute Error), RMSE (Root Mean Square Error), and R² metrics from the test set.
- **Charts:** Inspect the "Actual vs Predicted" scatter plot, the "Residuals" distribution, and the "Feature Importance" bar chart.
- Note the warning panel explicitly clarifying that metrics reflect the XGBoost model on synthetic data.

---

## 9. GENAI GENERATOR

**How to demonstrate:**
1. Select a Target Risk Level condition (e.g., "CRITICAL").
2. Set "Number of samples to generate" to a supported value like **10** or **50** (the slider allows 10 to 500).
3. Click "Generate Synthetic Data".
4. **Output to inspect:** Verify the table populates with new telemetry data. Check the feature distributions in the second tab to ensure they look like valid sensor distributions (no NaN/Inf).
5. **Export:** Note the "Download Synthetic Data (CSV)" button that appears, allowing export of the cGAN-generated data.

---

## 10. VAE GENERATOR

**How to demonstrate:**
1. Adjust the input signal parameters (Current, Ambient Temp, Discharge Rate, etc.).
2. Click "Encode & Reconstruct".
3. **Output to inspect:** View the 4-dimensional latent space representation (the 4 compressed values). 
4. **Charts to inspect:** Look at the Reconstruction Error line chart showing the original vs. reconstructed standardized values.
5. **Metrics to record:** Record the "Raw-scale MSE" and "Raw-scale MAE". Do not make claims of high accuracy; present the raw-scale errors objectively as the capability of the 4D bottleneck.

---

## 11. XAI

**A. SHAP Local**
- **Action:** Select the "Local Explanation (SHAP)" tab. Click "Generate SHAP Explanation".
- **What appears:** A waterfall plot detailing how individual features pushed the prediction up or down from the base value.
- **Record:** Note the top contributing features and capture a screenshot of the waterfall plot.

**B. LIME**
- **Action:** Select the "Local Explanation (LIME)" tab. Click "Generate LIME Explanation".
- **What appears:** Visual, colored cards showing feature weights (red for positive impact on temperature, green for negative).
- **Record:** Capture a screenshot showing the clean visualization cards (ensure no raw HTML tags are visible).

**C. Global SHAP**
- **Action:** Navigate to the "Model Performance" page and click the "SHAP Summary" tab. Click "Compute Global SHAP Summary".
- **What appears:** A bar chart showing mean absolute SHAP values across a sample of the dataset.
- **Record:** Note which features have the highest global impact. Capture a screenshot.

---

## 12. RESPONSIBLE AI

**How to demonstrate:**
Navigate to the "Responsible AI" page. No interaction is required.

**What to inspect:**
Scroll through and present the documented guardrails:
- The limitation of using synthetic simulation data.
- The boundary between deterministic safety logic and the LLM (emphasizing that the LLM has zero control over cooling).
- Limitations of model explainability.

---

## 13. RAG ASSISTANT

**How to demonstrate:**
Navigate to the "Prompt Lab" page. In the Prompt Engineering Console, execute the following queries:

**TEST 1 — Relevant technical question**
- **Type:** `What is battery thermal runaway?`
- **Action:** Click "Execute Prompt Pipeline".
- **Inspect:** Verify the LLM generates a relevant answer and lists valid context sources from the BTMS knowledge base.

**TEST 2 — BTMS specific question**
- **Type:** `How does coolant flow affect battery temperature?`
- **Action:** Click "Execute Prompt Pipeline".
- **Inspect:** Verify the technical answer and cited sources.

**TEST 3 — Unrelated question**
- **Type:** `Who won the FIFA World Cup?`
- **Action:** Click "Execute Prompt Pipeline".
- **Inspect:** Verify the RAG distance threshold rejects the query and returns the exact fallback message: `"This question is outside the available EV battery thermal-management knowledge base."`

---

## 14. PROMPT LAB

**Example Prompts:**
- `Explain the relationship between high discharge rate and thermal stress.`
- `What cooling strategies can mitigate hotspot formation?`

**How to demonstrate:**
- **Action:** Enter the prompt and click "Execute Prompt Pipeline".
- **Expected response:** A generated technical explanation based on retrieved markdown files.
- **Verify:** Point out the "Context Sources" expander to show the exact chunks retrieved. Emphasize that the LLM is acting strictly as an informational assistant and does not execute any deterministic code or control logic for the battery itself.

---

## 15. AI AGENT DECISION

This page demonstrates the core agentic loop.

**TEST A — NORMAL CONDITIONS**
- **Inputs:** Battery Current: `5.0 A`, Ambient Temp: `25.0 °C`, Coolant Flow: `0.020 kg/s`.
- **Action:** Click "Simulate thermal stress".
- **Record:** Note the XGBoost prediction, the NORMAL risk level, and the candidate table. Verify the agent selects "Maintain cooling" because it minimizes energy flow while staying in the NORMAL band.

**TEST B — THERMAL STRESS CONDITIONS**
- **Inputs:** Battery Current: `15.0 A`, Ambient Temp: `45.0 °C`, Coolant Flow: `0.010 kg/s`.
- **Action:** Click "Simulate thermal stress".
- **Record:**
  - XGBoost Prediction
  - MLP Prediction
  - Isolation Forest anomaly status
  - Initial Risk Level (Likely CRITICAL or HIGH)
  - The evaluated candidates in the table (Candidate Actions, Predicted Temps, Risk Levels, Flows).
  - The **Selected Action** (e.g., Emergency Cooling).
  - The **Decision Reason** (e.g., "Increased cooling to X% is required to safely manage thermal stress.").

---

## 16. DETERMINISM TEST

**How to verify:**
The AI Agent utilizes deterministic logic (XGBoost + sorting rules) rather than an LLM to make its final selection, ensuring repeatable safety.

Run **Test B (Thermal Stress)** from Section 15 exactly three times in a row without changing the sliders.

Fill out a comparison table for your records:

| Run | Initial Risk | Selected Action | Decision Reason |
|-----|--------------|-----------------|-----------------|
| 1   |              |                 |                 |
| 2   |              |                 |                 |
| 3   |              |                 |                 |

**What to compare:** You should observe that the selected action and the decision reason are exactly identical across all three runs, proving that the safety control loop is purely deterministic.

---

## 17. COMPLETE DEMONSTRATION FLOW

For a 10–15 minute presentation to a professor, use this order:

1. **Mission Control**: Start here to show the high-level professional overview.
2. **Thermal Analysis**: Run the 50-step live simulation to demonstrate the dynamic time-series capabilities.
3. **Digital Twin**: Perform the Stress Test to show spatial heatmap updates and UI consistency.
4. **AI Agent Decision (Crucial)**: Run the Normal and Stress tests to demonstrate the core observe -> predict -> simulate -> decide agentic loop. Show the determinism test.
5. **XAI**: Generate the LIME or SHAP local explanations to prove the model's decisions are transparent.
6. **Prompt Lab**: Run a relevant query and the FIFA query to demonstrate the RAG guardrails.
7. **Responsible AI**: Briefly flash this page to show awareness of safety and synthetic data limitations.

*(You can skip Dataset Explorer, VAE Generator, GenAI Generator, and Model Performance if time is short, as they are supporting ML tools rather than the core control logic).*

---

## 18. SCREENSHOT CHECKLIST

Check these off as you document your run:
- [ ] Mission Control
- [ ] Digital Twin Normal
- [ ] Digital Twin Stress
- [ ] Thermal Analysis (Status: ✅ COMPLETE)
- [ ] What-If Normal
- [ ] What-If Stress
- [ ] Dataset Explorer
- [ ] Model Performance
- [ ] GenAI
- [ ] VAE
- [ ] SHAP (Waterfall plot)
- [ ] LIME (Colored cards)
- [ ] Global SHAP (Bar chart)
- [ ] Responsible AI
- [ ] RAG / Prompt Lab (Relevant response)
- [ ] RAG / Prompt Lab (Out-of-domain rejection)
- [ ] AI Agent Stress

---

## 19. TEST RESULT TABLE

Print or copy this blank table to fill manually during your demonstration:

| Module | Test | Input | Result | PASS/FAIL | Screenshot |
|--------|------|-------|--------|-----------|------------|
| Digital Twin | Stress Test | Current: 15A, Amb: 45°C | Pack Max == Max Cell | | |
| Thermal Analysis | Live Run | NORMAL Profile | Completes 50 steps | | |
| What-If | Stress Case | Current 15A, Flow 0.005 | Temp spikes significantly | | |
| XAI | LIME Gen | Default | Cards render cleanly | | |
| Prompt Lab | Out-of-Domain | "FIFA World Cup" | Rejected by system | | |
| AI Agent | Stress Eval | Current 15A, Flow 0.010 | Increases cooling | | |
| AI Agent | Normal Eval | Current 5A, Flow 0.020 | Maintains cooling | | |
| AI Agent | Determinism | Run Stress 3x | Same result every time | | |
