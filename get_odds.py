from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

url = "https://sportsbook.draftkings.com/nba-player-props?wpsrc=Organic%20Search&wpaffn=Google&wpkw=https%3A%2F%2Fsportsbook.draftkings.com%2Fnba-player-props&wpcn=nba-player-props"  # Replace this with your desired URL

# SQLite Backend
# Set up the Chrome WebDriver (you need to download chromedriver.exe and specify its path)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # Run in headless mode (without a visible browser window)
driver = webdriver.Chrome(options=chrome_options)

# Open the URL in the browser
driver.get(url)

# Extract data using Selenium locators (modify as needed)
# For example, let's extract all the text from paragraph elements (change the locator accordingly)
tables = driver.find_elements(By.TAG_NAME, "table")
print(len(tables))

data = []
# Print the text content of each element
for table in tables:
    headers = ["Player", "Over", "Under"]

    # Locate all rows within the table
    rows = table.find_elements(By.XPATH, ".//tbody/tr")

    # Iterate through rows and extract text content of each cell
    
    for row in rows:
        player_name = row.find_element(By.TAG_NAME, "th").text.split("\n")[0]
        lines = row.find_elements(By.TAG_NAME, "td")
        row_data = [player_name] + [line.text for line in lines]
        data.append(row_data)
    
# Close the browser
driver.quit()

# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=headers)

# Print the DataFrame
print(df)

df.to_csv("data/odds.csv")