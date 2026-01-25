from datetime import datetime
import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

matplotlib.use("Agg")

DATA_ROOT = "./"  # sube desde filip/ al root del proyecto

TW_PATH = f"{DATA_ROOT}raw_data_agust_12/"
TM_PATH = f"{DATA_ROOT}raw_data_agust_tm/"
WY_PATH = f"{DATA_ROOT}raw_data_agust_wy/"
TM_FILIP_PATH = f"{DATA_ROOT}tm_data_filip/"

def convert_to_wyid(transfers, wy_tm_players, player_data):
    mapping = dict(zip(wy_tm_players['tm_id'], wy_tm_players['wy_id']))

    transfers['player_id'] = transfers['player_id'].astype(int)
    transfers['player_id'] = transfers['player_id'].map(mapping)

    nameMap = player_data.set_index("player_id")["short_name"]
    transfers['player_name'] = transfers['player_id'].map(nameMap)

    return transfers

def clean_ids(frame, teams, comps):
    mapping = dict(zip(teams["tm_id"], teams["team_name"]))

    frame["team_id_to"] = frame["team_id_to"].astype(float)
    frame["team_id_from"] = frame["team_id_from"].astype(float)
    frame["team_from"] = frame["team_id_from"].map(mapping)
    frame["team_to"] = frame["team_id_to"].map(mapping)

    return frame

def merge_stats_with_transfer(transfers, stats):
    transfers['season'] = transfers['season'].astype(int)
    return transfers.merge(stats, on=['player_id', 'season'], how='left')

def lin_reg(df):
    df = df.dropna(subset=["age_at_transfer", "dateremaining_contract_period", "Minutes", "transfer_value"])
    df = df[df['transfer_value'] > 0]
    df['log_value'] = np.log(df['transfer_value'].astype(int))

    X = df[['age_at_transfer', 'dateremaining_contract_period', 'Minutes']]
    y = df['log_value']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df[['age_at_transfer', 'dateremaining_contract_period', 'Minutes']].corr()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R²:", r2_score(y_test, y_pred))

    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_
    })

    print(coefficients)

if __name__ == "__main__":
    transfer_data = pd.read_parquet(f"{TW_PATH}male_transfers_data.parquet")
    player_data = pd.read_parquet(f"{WY_PATH}players_wyscout.parquet")
    tm_league_links = pd.read_parquet(f"{TM_PATH}tm_league_links.parquet")
    tm_teams = pd.read_parquet(f"{TM_PATH}tm_teams.parquet")
    wy_tm_players = pd.read_parquet(f"{TM_PATH}wy_tm_players_mapping.parquet")
    transfer_history = pd.read_parquet(f"{TM_FILIP_PATH}transfer_history_tm.parquet")

    df = convert_to_wyid(transfer_history, wy_tm_players, player_data)

    df = df.dropna(subset=["contract_until_date", "dateremaining_contract_period"])

    df = clean_ids(df, tm_teams, tm_league_links)

    df = merge_stats_with_transfer(df, transfer_data)

    lin_reg(df)