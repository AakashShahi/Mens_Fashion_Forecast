import pandas as pd
p=pd.read_csv("data/google_trends_1yr_nepal.csv")["Keyword"].unique().tolist()
print(p)
df = pd.read_csv("data/social_media_1yr_nepal.csv")
print(df["Hashtags"].head(20))