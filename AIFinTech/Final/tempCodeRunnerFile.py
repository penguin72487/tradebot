norm_cols = [col for col in features if col in df.columns]
df[norm_cols] = df[norm_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
