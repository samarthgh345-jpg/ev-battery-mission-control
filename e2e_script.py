from playwright.sync_api import sync_playwright
import time
import os

PAGES = [
    'Mission Control',
    'Digital Twin',
    'Thermal Analysis',
    'What-If Simulator',
    'Dataset Explorer',
    'Model Performance',
    'GenAI Generator',
    'VAE Generator',
    'XAI',
    'Responsible AI',
    'RAG Assistant',
    'AI Agent Decision'
]

os.makedirs('screenshots', exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 800})
    page.goto('http://localhost:8503/')
    time.sleep(5)
    
    print('Testing Mission Control buttons...')
    
    for btn_text in ['Why this prediction?', 'Run What-If', 'Start live simulation', 'Simulate thermal stress']:
        if page.locator(f'button:has-text("{btn_text}")').count() > 0:
            print(f'Button {btn_text} found.')
        else:
            print(f'Button {btn_text} NOT FOUND!')
            
    page.click('button:has-text("Why this prediction?")')
    time.sleep(3)
    page.screenshot(path='screenshots/01_after_click_why.png', full_page=True)
    
    def select_page(page_name):
        print(f'Selecting {page_name}...')
        page.locator('div[data-testid="stSelectbox"]').first.click()
        time.sleep(0.5)
        page.locator(f'li[role="option"]:has-text("{page_name}")').click()
        time.sleep(2)
        
    select_page('Mission Control')
    page.screenshot(path='screenshots/02_mission_control.png', full_page=True)
    
    for i, p_name in enumerate(PAGES):
        select_page(p_name)
        
        if p_name == 'Thermal Analysis':
            if page.locator('button:has-text("Start Live Simulation")').count() > 0:
                page.click('button:has-text("Start Live Simulation")')
                time.sleep(3)
                
        if p_name == 'AI Agent Decision':
            if page.locator('button:has-text("Simulate thermal stress")').count() > 0:
                page.click('button:has-text("Simulate thermal stress")')
                time.sleep(5)
                
        page.screenshot(path=f'screenshots/page_{i:02d}_{p_name.replace(" ", "_")}.png', full_page=True)
        
    select_page('RAG Assistant')
    page.fill('input[type="text"]', 'What are the main causes of thermal runaway in lithium-ion batteries, and how can a BTMS help prevent it?')
    page.click('button:has-text("Ask Assistant")')
    time.sleep(3)
    page.screenshot(path='screenshots/rag_query_1.png', full_page=True)
    
    # clear input
    page.fill('input[type="text"]', '')
    page.fill('input[type="text"]', 'What is the capital of France?')
    page.click('button:has-text("Ask Assistant")')
    time.sleep(3)
    page.screenshot(path='screenshots/rag_query_2.png', full_page=True)

    browser.close()
