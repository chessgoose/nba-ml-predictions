from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from datetime import datetime
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# SQLite Backend
league = "wnba"

# Returns: pandas dataframe with odds 
def get_odds_today(league="nba"):
    data = []
    headers = ["Date", "Player", "Line", "Over", "Under"]
    url = "https://sportsbook.draftkings.com/nba-player-props?category=player-points&subcategory=points"
    if league != "nba":
        url = "https://sportsbook.draftkings.com/leagues/basketball/wnba?category=player-points&subcategory=points"
        
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")  # Run in headless mode (without a visible browser window)
    driver = webdriver.Chrome(options=chrome_options)

    # Open the URL in the browser
    driver.get(url)

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
    new_rows = pd.DataFrame(data, columns=headers)
    driver.quit()
    return new_rows

# Create a Pandas DataFrame
# df['Date'] = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
# Print the DataFrame

if __name__ == "__main__":
    if league == "wnba":
        file_name = 'data/wnba_odds.csv'
        df = pd.read_csv(file_name, index_col=False)
        headers = ["Date", "Player", "Line", "Over", "Under"]
        new_rows = get_odds_today("wnba")
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)
    else:
        file_name = 'data/new_odds_two.csv'
        df = pd.read_csv(file_name, index_col=False)
        headers = ["Date", "Player", "Line", "Over", "Under"]
        new_rows = get_odds_today()
        df = pd.concat([df, new_rows])
        df.to_csv(file_name, index=False)