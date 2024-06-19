from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from datetime import datetime
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# SQLite Backend

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
    driver.get(url)
    driver.maximize_window()

    # Extract data using Selenium locators

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

        if "TODAY" not in dates[i].text:
            continue

        # Locate all rows within the table
        rows = table.find_elements(By.XPATH, ".//tbody/tr")

        # Iterate through rows and extract text content of each cell
        
        for row in rows:
            player_name = row.find_element(By.TAG_NAME, "th").text.split("\n")[0]
            lines = row.find_elements(By.TAG_NAME, "td")

            #print(len(lines))
            #print(lines[0].text)
            #print(lines[1].text)

            over_odds = float(lines[0].text.split("\n")[2].replace('−', '-'))
            line = float(lines[0].text.split("\n")[1])
            under_odds = float(lines[1].text.split("\n")[2].replace('−', '-'))
            
            row_data = [datetime.today().strftime('%Y-%m-%d'), player_name, line, under_odds, over_odds]
            print(row_data)
            data.append(row_data)
        
    # Close the browser
    driver.quit()

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
    except:
        return []

    finally:
        # Close the WebDriver
        driver.quit()

    return matchups

if __name__ == "__main__":
    # Find both odds for NBA and WNBA
    print(get_matchups())
    """
    file_name = 'data/wnba_odds.csv'
    df = pd.read_csv(file_name, index_col=False)
    headers = ["Date", "Player", "Line", "Over", "Under"]
    new_rows = get_odds_today("wnba")
    if not new_rows.empty:
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)
    """
    """
    file_name = 'data/new_odds_two.csv'
    df = pd.read_csv(file_name, index_col=False)
    headers = ["Date", "Player", "Line", "Over", "Under"]
    new_rows = get_odds_today()
    if not new_rows.empty:
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)
    """