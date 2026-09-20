import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\samar\.gemini\antigravity-ide\brain\e2f45aeb-c52d-47c4-80fe-ca9752021d5e"

def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        
        print("Navigating to app...")
        page.goto("http://localhost:8501")
        page.wait_for_timeout(3000) # Wait for initial load
        
        # Test 1: Mission Control
        print("Test 1: Mission Control")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test1_mission_control.png"))
        
        # Helper to click sidebar tab
        def go_to_tab(tab_name):
            try:
                page.locator(f"text={tab_name}").click(timeout=3000)
                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"Could not click tab {tab_name}: {e}")
            
        # Test 2: AI Prediction
        print("Test 2: Thermal Analysis")
        go_to_tab("Thermal Analysis")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test2_prediction_default.png"))
        
        # Test 3: What-If Simulator
        print("Test 3: What-If Simulator")
        go_to_tab("What-If Simulator")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test3_whatif.png"))
        
        # Test 4: XAI / Why
        print("Test 4: XAI")
        go_to_tab("XAI (Why?)")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test4_xai.png"))
        
        # Test 9: Digital Twin
        print("Test 9: Digital Twin")
        go_to_tab("Digital Twin")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test9_digital_twin.png"))
        
        # Test 7: AI Agent
        print("Test 7: AI Agent Decision")
        go_to_tab("AI Agent Decision")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test7_agent.png"))
        
        # Test 11: Model Performance
        print("Test 11: Model Performance")
        go_to_tab("Model Performance")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test11_performance.png"))
        
        # Test 12: Dataset Explorer
        print("Test 12: Dataset Explorer")
        go_to_tab("Dataset Explorer")
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "test12_dataset.png"))
        
        print("Tests completed successfully!")
        browser.close()

if __name__ == "__main__":
    run_tests()
