from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from datetime import datetime, timedelta
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Pinnacle spreads (would this even work in the US?)
def get_team_points_today():
    url = 'https://www.pinnacle.com/en/basketball/wnba/matchups/#team_total'

    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=chrome_options)

    # Open the URL in the browser
    driver.maximize_window()
    driver.get(url)

    # Define a wait object with a timeout
    wait = WebDriverWait(driver, 10)  #5 seconds timeout

    # Wait until the dates elements are present
    data = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[class*="market-bt"]')))  # Adjust the substring as needed
    
    assert len(data) != 0
    print(len(data))

    for d in data:
        print(d.text)

    # First team should be the first team on WNBA matchups for the day
    totals = []
    relevant = []

    # 10 
    for i, table in enumerate(data):
        # First is moneyline spread
        # Third is the total 
        stuff = data[i].text
        if "GAME" in stuff:
            continue
        if "HALF" in stuff:
            continue
        if "OVER" in stuff:
            continue
        relevant.append(stuff)

    for i, text in enumerate(relevant):
        if i % 2 == 0:
            numbers = text.split("\n")
            totals.append(float(numbers[0]))
            totals.append(float(numbers[4]))

    # can we scrape the team data too?

    
    print(totals)
    driver.quit()
    return totals 

def get_spreads_today():
    # Pinnacle Odds are by far the best ground truth for this
    url = 'https://www.pinnacle.com/en/basketball/wnba/matchups/#period:0'

    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--headless")  # Run in headless mode (without a visible browser window)
    driver = webdriver.Chrome(options=chrome_options)

    # Open the URL in the browser
    driver.maximize_window()
    driver.get(url)

    # Define a wait object with a timeout
    wait = WebDriverWait(driver, 10)  #5 seconds timeout

    # Wait until the dates elements are present
    data = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[class*="style_button-wrapper"]')))  # Adjust the substring as needed
    
    assert len(data) != 0
    print(len(data))

    # First team should be the first team on WNBA matchups for the day
    spreads = []
    totals = []

    # 10 
    for i, table in enumerate(data):
        # First is moneyline spread
        # Third is the total 
        stuff = data[i].text
        print(stuff)
        
        if i % 6 == 0:
            s = float(stuff.split("\n")[0]) 
            # We have a spread
            spreads.append(s)
        elif i % 6 == 5:
            t = float(stuff.split("\n")[0]) 
            totals.append(t)

        #rows = data[i].find_elements(By.CSS_SELECTOR, '[class*="style_button-wrapper"]')
        # Get the style label within each value

    driver.quit()

    print(f"Spreads: {spreads}")
    print(f"Totals: {totals}")

    # Obtain the IMPLIED PPG of each team
    return spreads, totals

# FIX THIS SHIT BUT IT WORKS FOR NOW I GUESS
def handle_popups(driver):
    try:
        # Example of handling a generic popup, can be customized as per the actual popup
        popups = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "modal-with-mask")))
        print(len(popups))
        for popup in popups:
            close_button = popup.find_element(By.CLASS_NAME, "primary")
            close_button.click()
    except Exception as e:
        print("No pop-ups found or unable to close pop-up: ", e)

# Returns: pandas dataframe with odds 
def get_odds_today(league="nba"):
    data = []
    headers = ["Date", "Player", "Line", "Over", "Under"]
    url = "https://sportsbook.draftkings.com/nba-player-props?category=player-points&subcategory=points"
    if league != "nba":
        url = "https://sportsbook.draftkings.com/leagues/basketball/wnba?category=player-points&subcategory=points"
        
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--headless")  # Run in headless mode (without a visible browser window)
    driver = webdriver.Chrome(options=chrome_options)

    # Open the URL in the browser
    driver.maximize_window()
    driver.get(url)

    # Extract data using Selenium locators
    handle_popups(driver)

    # Define a wait object with a timeout
    wait = WebDriverWait(driver, 5)  #5 seconds timeout

    # Wait until the dates elements are present
    dates = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "sportsbook-event-accordion__date")))
    print(len(dates))

    # Wait until the tables elements are present
    tables = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "table")))
    print(len(tables))

    assert len(tables) != 0

    # Print the text content of each element
    for i, table in enumerate(tables):
        print(dates[i].text)

        if "TODAY" in dates[i].text:
            # Locate all rows within the table
            rows = table.find_elements(By.XPATH, ".//tbody/tr")

            # Iterate through rows and extract text content of each cell
            
            for row in rows:
                player_name = row.find_element(By.TAG_NAME, "th").text.split("\n")[0]
                lines = row.find_elements(By.TAG_NAME, "td")

                #print(len(lines))
                #print(lines[0].text)
                #print(lines[1].text)

                over_odds = float(lines[0].text.split("\n")[2].replace('−', '-')) if len(lines[0].text.split("\n")) > 2 else 0
                line = float(lines[0].text.split("\n")[1])
                under_odds = float(lines[1].text.split("\n")[2].replace('−', '-')) if len(lines[1].text.split("\n")) > 2 else 0
                
                row_data = [datetime.today().strftime('%Y-%m-%d'), player_name, line, under_odds, over_odds]
                print(row_data)
                data.append(row_data)
        elif "TOMORROW" in dates[i].text:
            # Locate all rows within the table
            rows = table.find_elements(By.XPATH, ".//tbody/tr")

            # Iterate through rows and extract text content of each cell
            
            for row in rows:
                player_name = row.find_element(By.TAG_NAME, "th").text.split("\n")[0]
                lines = row.find_elements(By.TAG_NAME, "td")

                over_odds = float(lines[0].text.split("\n")[2].replace('−', '-'))
                line = float(lines[0].text.split("\n")[1])
                under_odds = float(lines[1].text.split("\n")[2].replace('−', '-'))
                
                row_data = [(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d'), player_name, line, under_odds, over_odds]
                print(row_data)
                data.append(row_data)
        else:
            continue

    # Close the browser
    #driver.quit()
    new_rows = pd.DataFrame(data, columns=headers)
    return new_rows


def get_matchups():
    today_date = datetime.today()
    # Format the date to "MM/DD/YY"
    formatted_date = today_date.strftime('%m/%d/%Y')
    print(formatted_date)

    url = f'https://stats.wnba.com/scores/{formatted_date}'
    print(url)

    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--window-size=1920,1080")  # Run in headless mode (without a visible browser window)
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Open the WNBA scores page
        driver.get(url)
        
        time.sleep(5)

        # wait = WebDriverWait(driver, 10)  # Increase the wait time as needed
        #matchups_section = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'scores-game__inner'))) 
        matchups_section = driver.find_elements(By.CLASS_NAME, 'scores-game__inner')  # Adjust class name if needed
        
        assert len(matchups_section) != 0

        # Extract team abbreviations from the matchups section
        matchups = []
        for game in matchups_section:
            teams = game.find_elements(By.CLASS_NAME, 'scores-game__team-abbr')  # Adjust class name if needed
            if len(teams) == 2:
                team1 = teams[0].text.strip()
                team2 = teams[1].text.strip()
                matchups.append((team1, team2))
        
        print(matchups)
    except:
        return []

    finally:
        # Close the WebDriver
        driver.quit()

    return matchups

if __name__ == "__main__":
    # Find both odds for NBA and WNBA
    # get_team_points_today()

    file_name = 'data/wnba_odds.csv'
    df = pd.read_csv(file_name, index_col=False)
    headers = ["Date", "Player", "Line", "Over", "Under"]
    new_rows = get_odds_today("wnba")
    if not new_rows.empty:
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)

    """
    file_name = 'data/new_odds_two.csv'
    df = pd.read_csv(file_name, index_col=False)
    headers = ["Date", "Player", "Line", "Over", "Under"]
    new_rows = get_odds_today()
    if not new_rows.empty:
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)
    """