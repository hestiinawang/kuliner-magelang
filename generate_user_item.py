import pandas as pd
import numpy as np
import random

# Set seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# Load data asli
df = pd.read_csv('data_clean.csv')
items = df[df['item'].notna() & (df['item'] != '')]['item'].unique().tolist()

print(f"Total item tersedia: {len(items)}")

# Parameter
n_users = 50
ratings_per_user = (10, 20)  # range rating per user
noise_range = (-0.3, 0.3)  # noise untuk rating

# Generate user-item matrix
user_item_data = []

for user_id in range(1, n_users + 1):
    # Acak jumlah item yang akan dirating user ini
    n_ratings = random.randint(*ratings_per_user)
    user_ratings = random.sample(items, n_ratings)
    
    for item in user_ratings:
        # Ambil rating asli + noise
        orig_row = df[df['item'] == item]
        if len(orig_row) == 0:
            continue
        orig_rating = str(orig_row['rating'].values[0]).replace(',', '.')
        orig_rating = float(orig_rating)
        
        noise = random.uniform(*noise_range)
        new_rating = round(max(1, min(5, orig_rating + noise)), 1)
        
        user_item_data.append({
            'user_id': f'U{user_id:03d}',
            'item': item,
            'rating': new_rating
        })

# Simpan ke CSV
user_item_df = pd.DataFrame(user_item_data)
user_item_df.to_csv('user_item_matrix.csv', index=False)

print(f"\n=== Hasil Generate ===")
print(f"Total ratings: {len(user_item_df)}")
print(f"Users: {user_item_df['user_id'].nunique()}")
print(f"Items: {user_item_df['item'].nunique()}")
print(f"\nFile: user_item_matrix.csv")

# Tampilkan sample
print(f"\n=== Sample Data ===")
print(user_item_df.head(20).to_string(index=False))