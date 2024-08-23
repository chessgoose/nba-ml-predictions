import scipy.stats as stats
import pandas as pd

def american_odds_to_probability(odds):
    if odds > 0:
        probability = 100 / (odds + 100)
    else:
        probability = -odds / (odds - 100)
    return probability

def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds >= 100:
        decimal_odds = (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds))
    return round(decimal_odds, 2)

def generate_confidence_score(projected_value, betting_line, projected_value_std):
    # Convert American odds to probability
    betting_line_probability = american_odds_to_probability(betting_line)

    # Calculate the z-score based on the normal distribution
    z_score = (projected_value - betting_line_probability) / projected_value_std

    # Use the cumulative distribution function (CDF) to get the confidence score
    confidence = 1 - min(1, abs(z_score) / 1.25)
    return confidence

def calculate_over_ev(projection, betting_line, std_deviation, over_odds):
    # Convert over odds to probability
    
    # Calculate Z-score based on normal distribution
    z_score = (projection - betting_line) / 2
    print(z_score)

    # Calculate the probability of winning using the cumulative distribution function (CDF)
    win_prob = stats.norm.cdf(z_score)

    # Calculate the probability of losing
    lose_prob = 1 - win_prob

    # Calculate potential profit and loss
    profit = 0
    if over_odds > 0:
        profit = over_odds / 100
    else:
        profit = (1 / over_odds) * 100
    loss = 1  # Assuming the full amount is lost in case of a loss

    # Calculate expected value
    expected_value = (win_prob * profit) - (lose_prob * loss)

    return expected_value

def convert_number_string(input_string): 
    # Check if the string starts with a plus or minus sign
    if input_string[0] == '+':
        # Convert positive number string to integer
        result = int(input_string[1:])
    elif input_string[0] == '−':
        # Convert negative number string to negative integer
        result = -int(input_string[1:])
    else:
        # Convert regular number string to integer
        result = int(input_string)

    return result

df1 = pd.read_csv('data/odds.csv')
df2 = pd.read_csv('data/points.csv')

for (index1, row1), (index2, row2) in zip(df1.iterrows(), df2.iterrows()):
    try:
        player_name = row1['Player']
        over_lst = row1['Over'].split("\n")
        betting_line = float(over_lst[1].strip())
        over_odds = convert_number_string(over_lst[2])
        under_odds = convert_number_string(row1['Under'].split("\n")[2])
        projection = round(float(row2['Projection']), 1)
        std = float(row2['Standard Deviation'])
        print(player_name, betting_line, projection, std, over_odds, under_odds)

        over_ev = calculate_over_ev(projection, betting_line, std, over_odds)

        if projection > betting_line:
            print("over")
        else:
            print('under')

        # Calculate EV of betting the over/under based on odds AND projections

        
    except:
        print("Error proceessing row")
