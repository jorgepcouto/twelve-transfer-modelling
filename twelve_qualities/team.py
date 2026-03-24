from .metric import Metric
from .target import Target

TEAM = Target.TEAM


BALL_POSSESSION_PCT = Metric(
    name="Ball possession %",
    long_name="Ball possession %",
    api_name="ball_possession_pct",
    description="Share of possession of a team per match.",
    calculation="Number of touches a team had divided by the total number of touches of both teams in the match per match.",
    target=TEAM,
    unit="%",
)

BALL_IN_PLAY_MINUTES = Metric(
    name="Ball-in-play minutes",
    long_name="Ball-in-play minutes",
    api_name="ball_in_play_minutes",
    description="Number of minutes that the ball was in play in a match per match.",
    calculation="Number of minutes that the ball was in play in a match per match.",
    target=TEAM,
)

BOX_ENTRIES_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Possessions to box within 10s after recovery",
    long_name="Possessions to box within 10s after recovery",
    api_name="possessions_to_box_within_10s_after_recovery",
    description="Number of possessions of a team that entered the penalty area following an open play ball recovery per match.",
    calculation="Number of possessions of a team that entered the penalty area following an open play ball recovery per match.",
    target=TEAM,
)

BOX_ENTRIES_TO_SHOT_PCT = Metric(
    name="Box to shot %",
    long_name="Box to shot %",
    api_name="box_to_shot_pct",
    description="Percentage of box entering possessions by a team that led to a shot per match.",
    calculation="Number of possessions by a team with a shot inside the penalty area divided by the number of possessions by the team that reached the penalty area per match.",
    target=TEAM,
    unit="%",
)

OPP_BOX_ENTRIES_TO_SHOT_PCT = Metric(
    name="Opp. Box to shot %",
    long_name="Opponent box to shot %",
    api_name="opp_box_to_shot_pct",
    description="Percentage of box entering possessions by the opposing team that led to a shot per match.",
    calculation="Number of possessions by the opposing team with a shot inside the penalty area divided by the number of possessions by the opposing team that reached the penalty area per match.",
    target=TEAM,
    unit="%",
    higher_is_better=False,
)

TEAM_DEFENSIVE_DUELS_WON_PCT = Metric(
    name="Defensive duels won %",
    long_name="Defensive duels won %",
    api_name="defensive_duels_won_pct",
    description="Percentage of defensive duels won by a team per match.",
    calculation="Number of defensive duels won by a team divided by the total amount of defensive duels of the team per match.",
    target=TEAM,
    unit="%",
)

TEAM_DEFENSIVE_INTENSITY = Metric(
    name="Defensive intensity",
    long_name="Defensive intensity",
    api_name="defensive_intensity",
    description="Number of defensive actions (duels, interceptions, tackles and fouls) that a team has per minute out of possession per match.",
    calculation="Number of defensive actions (duels, interceptions, tackles and fouls) that a team has divided by the possession time of their opponents in minutes per match.",
    target=TEAM,
)

TEAM_DEFENSIVE_LINE_HEIGHT = Metric(
    name="Defensive action height (m)",
    long_name="Defensive action height (m)",
    api_name="defensive_action_height_m",
    description="Average height on the pitch of defensive actions by a team in open play per match.",
    calculation="Average height on the pitch of defensive actions by a team in open play per match.",
    target=TEAM,
)

FINAL_THIRD_ENTRIES_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Possessions to final third within 10s after recovery",
    long_name="Possessions to final third within 10s after recovery",
    api_name="possessions_to_final_third_within_10s_after_recovery",
    description="Number of possessions of a team that successfully reached the final third within 10 seconds after an open play ball recovery per match.",
    calculation="Number of possessions of a team that successfully reached the final third within 10 seconds after an open play ball recovery per match.",
    target=TEAM,
)

FINAL_THIRD_TO_BOX_PCT = Metric(
    name="Final third to box %",
    long_name="Final third to box %",
    api_name="final_third_to_box_pct",
    description="Percentage of final third possessions by a team that entered the penalty area per match.",
    calculation="Number of possessions by a team that entered the penalty area divided by the total number of final third possessions by the team per match.",
    target=TEAM,
    unit="%",
)

TEAM_GOALS = Metric(
    name="Goals",
    long_name="Goals",
    api_name="goals",
    description="Number of goals that a team scored per match.",
    calculation="Number of goals that a team scored per match.",
    target=TEAM,
)

GOAL_DIFFERENCE = Metric(
    name="Goal difference",
    long_name="Goal difference",
    api_name="goal_difference",
    description="Difference between the number of goals scored and the number of goals conceded by a team per match.",
    calculation="Number of goals scored minus the number of goals conceded by a team per match.",
    target=TEAM,
)

HIGH_OPPORTUNITY_SHOTS = Metric(
    name="High opportunity shots",
    long_name="High opportunity shots",
    api_name="high_opportunity_shots",
    description="Number of non-penalty shots by a team with expected goals higher than 0.15 per match.",
    calculation="Number of non-penalty shots by a team with expected goals higher than 0.15 per match.",
    target=TEAM,
)

OPP_HIGH_OPPORTUNITY_SHOTS = Metric(
    name="Opp. High opportunity shots",
    long_name="Opponent high opportunity shots",
    api_name="opp_high_opportunity_shots",
    description="Number of non-penalty shots by the opposing team with expected goals higher than 0.15 per match.",
    calculation="Number of non-penalty shots by the opposing team with expected goals higher than 0.15 per match.",
    target=TEAM,
    higher_is_better=False,
)

LONG_BALL_PCT = Metric(
    name="Long ball %",
    long_name="Long ball %",
    api_name="long_ball_pct",
    description="Percentage of passes longer than 32 meters in the defensive half by a team per match.",
    calculation="Number of passess longer than 32 meters in the defensive half by a team divided by the total number of passes in the defensive half by the team per match.",
    target=TEAM,
    unit="%",
    style_of_play_left=True,
    higher_is_better=False,
)

OPP_GOALS = Metric(
    name="Opp. Goals",
    long_name="Opponent goals",
    api_name="opp_goals",
    description="Number of goals that the opposing team scored per match.",
    calculation="Number of goals that the opposing team scored per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_XG = Metric(
    name="Opp. xG",
    long_name="Opponent expected goals",
    api_name="opp_xg",
    description="Number of goals that the opposing team should be expected to score statistically considering the quality of their chances per match. ",
    calculation="Number of goals that the opposing team should be expected to score statistically considering the quality of their chances per match. ",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_POSSESSIONS_TO_FINAL_THIRD_PCT = Metric(
    name="Opp. Possessions to final third %",
    long_name="Opponent possessions to final third %",
    api_name="opp_possessions_to_final_third_pct",
    description="Percentage of possessions by the opposing team that reached the final third per match.",
    calculation="Number of possessions by the opposing team that reached the final third divided by the total number of possessions by the opposing team per match.",
    target=TEAM,
    unit="%",
    higher_is_better=False,
    opponent=True,
)

OPP_XT_TEAM = Metric(
    name="Opp. xT",
    long_name="Opponent expected threat",
    api_name="opp_xt",
    description="Expected offensive contribution by the opposing team considering the quality of their passess and ball carries per match.",
    calculation="Expected offensive contribution by the opposing team considering the quality of their passes and ball carries per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_FINAL_THIRD_ENTRIES_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Opp. Possessions to final third within 10s after recovery",
    long_name="Opponent possessions to final third within 10s after recovery",
    api_name="opp_possessions_to_final_third_within_10s_after_recovery",
    description="Number of possessions of the opposing team that successfully reached the final third within 10 seconds after an open play ball recovery per match.",
    calculation="Number of possessions of the opposing team that successfully reached the final third within 10 seconds after an open play ball recovery per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_FINAL_THIRD_TO_BOX_PCT = Metric(
    name="Opp. Final third to box %",
    long_name="Opponent final third to box %",
    api_name="opp_final_third_to_box_pct",
    description="Percentage of final third possessions by the opposing team that entered the penalty area per match.",
    calculation="Number of possessions by the opposing team that entered the penalty area divided by the total number of final third possessions by the opposing team per match.",
    target=TEAM,
    unit="%",
    higher_is_better=False,
    opponent=True,
)

OPP_BOX_ENTRIES_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Opp. Possessions to box within 10s after recovery",
    long_name="Opponent possessions to box within 10s after recovery",
    api_name="opp_possessions_to_box_within_10s_after_recovery",
    description="Number of possessions of the opposing team that entered the penalty area following an open play ball recovery per matcb.",
    calculation="Number of possessions of the opposing team that entered the penalty area following an open play ball recovery per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_PASS_TEMPO = Metric(
    name="Opp. Pass tempo",
    long_name="Opponent pass tempo",
    api_name="opp_pass_tempo",
    description="Number of passes that the opposing team made per minute of possession per match.",
    calculation="Number of passes by the opposing team divided by their possession time in minutes per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_XG_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Opp. xG within 10s after recovery",
    long_name="Opponent expected goals within 10s after recovery",
    api_name="opp_xg_within_10s_after_recovery",
    description="Number of goals that the opposing team should be expected to score statistically considering the quality of their chances from attacks following an open play ball recovery per match.",
    calculation="Number of goals that the opposing team should be expected to score statistically considering the quality of their chances from attacks following an open play ball recovery per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

OPP_XT_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="Opp. xT within 10s after recovery",
    long_name="Opponent expected threat within 10s after recovery",
    api_name="opp_xt_within_10s_after_recovery",
    description="Expected offensive contribution by the opposing team considering the quality of their actions within 10 seconds after an open play ball recovery per match.",
    calculation="Expected offensive contribution by the opposing team considering the quality of their actions within 10 seconds after an open play ball recovery per match.",
    target=TEAM,
    higher_is_better=False,
    opponent=True,
)

PPDA = Metric(
    name="PPDA",
    long_name="Passes per defensive action",
    api_name="ppda",
    description="Number of passes of the opponent team in their defensive 60% of the field compared to the number of defensive actions of a team in the same region per match.",
    calculation="Number of passes of the opponent team in their defensive 60% of the field compared to the number of defensive actions of a team in the same region per match.",
    target=TEAM,
    higher_is_better=False,
    style_of_play_left=True,
)

PASS_TEMPO = Metric(
    name="Pass tempo",
    long_name="Pass tempo",
    api_name="pass_tempo",
    description="The number of passes that a team made per minute of possession per match.",
    calculation="The number of passes by a team divided by their possession time in minutes per match.",
    target=TEAM,
)

PENALTY_AREA_TOUCHES = Metric(
    name="Box touches",
    long_name="Box touches",
    api_name="box_touches",
    description="Number of touches by a team inside the opponent's penalty area per match.",
    calculation="Number of touches by a team inside the opponent's penalty area per match.",
    target=TEAM,
)

OPP_PENALTY_AREA_TOUCHES = Metric(
    name="Opp. Box touches",
    long_name="Opponent box touches",
    api_name="opp_box_touches",
    description="Number of touches by a team inside the opponent's penalty area per match.",
    calculation="Number of touches by a team inside the opponent's penalty area per match.",
    target=TEAM,
    higher_is_better=False,
)

POSSESSIONS_RETAINED_AFTER_5S = Metric(
    name="Possessions retained after 5s",
    long_name="Possessions retained after 5s",
    api_name="possessions_retained_after_5s",
    description="Number of possessions by a team that lasted longer than 5 seconds after recovering the possession of the ball in open play per match.",
    calculation="Number of possessions by a team that lasted longer than 5 seconds after recovering the possession of the ball in open play per match.",
    target=TEAM,
)

POSSESSIONS_RETAINED_AFTER_5S_PCT = Metric(
    name="Possessions retained after 5s %",
    long_name="Possessions retained after 5s %",
    api_name="possessions_retained_after_5s_pct",
    description="Percentage of possessions by a team that lasted longer than 5 seconds after recovering the possession of the ball in open play per match.",
    calculation="Number of possessions by a team that lasted longer than 5 seconds after recovering the possesssion of the ball in open play divided by the number of ball recoveries by the team in open play per match.",
    target=TEAM,
    unit="%",
    style_of_play_left=True,
)

POSSESSIONS_TO_FINAL_THIRD_PCT = Metric(
    name="Possessions to final third %",
    long_name="Possessions to final third %",
    api_name="possessions_to_final_third_pct",
    description="Percentage of possessions by a team that reached the final third per match.",
    calculation="Number of possessions by a team that reached the final third divided by the total number of possessions by the team per match.",
    target=TEAM,
    unit="%",
)

RECOVERIES = Metric(
    name="Recoveries",
    long_name="Recoveries",
    api_name="recoveries",
    description="Number of ball recoveries of a team in open play per match.",
    calculation="Number of ball recoveries of a team in open play per match.",
    target=TEAM,
)

RECOVERIES_WITHIN_5S_PCT = Metric(
    name="Recoveries within 5s %",
    long_name="Recoveries within 5s %",
    api_name="recoveries_within_5s_pct",
    description="Percentage of ball recoveries made by a team within 5 seconds after losing the possession of the ball in open play per match.",
    calculation="Number of ball recoveries made by a team within 5 seconds after losing the possession of tha ball in open play divided by the number of times that the team lost the ball in open play per match.",
    target=TEAM,
    unit="%",
)

RECOVERY_LINE_HEIGHT = Metric(
    name="Recovery line height (m)",
    long_name="Recovery line height (m)",
    api_name="recovery_line_height_m",
    description="Average height on the pitch that a team recovered the ball per match.",
    calculation="Average height on the pitch that a team recovered the ball per match.",
    target=TEAM,
)

RED_CARDS = Metric(
    name="Red cards",
    long_name="Red cards",
    api_name="red_cards",
    description="Number of red cards received by a team per match.",
    calculation="Number of red cards received by a team per match.",
    target=TEAM,
    higher_is_better=False,
)

OPP_RED_CARDS = Metric(
    name="Opp. Red cards",
    long_name="Opponent red cards",
    api_name="opp_red_cards",
    description="Number of red cards received by the opposing team per match.",
    calculation="Number of red cards received by the opposing team per match.",
    target=TEAM,
)

FIELD_TILT = Metric(
    name="Field tilt %",
    long_name="Field tilt %",
    api_name="field_tilt_pct",
    description="Percentage of possession that a team had in the final third compared to their opponents per match.",
    calculation="Number of touches a team had in the final third divided by the total number of final third touches of both teams in the match per match.",
    target=TEAM,
    unit="%",
)

TIME_TO_DEFENSIVE_ACTION = Metric(
    name="Time to defensive action (s)",
    long_name="Time to defensive action (seconds)",
    api_name="time_to_defensive_action_s",
    description="Average time that it took a team to make a defensive action after losing the possession of the ball in open play per match. ",
    calculation="Average time that it took a team to make a defensive action after losing the possession of the ball in open play per match. ",
    target=TEAM,
    higher_is_better=False,
)

TIME_TO_RECOVERY = Metric(
    name="Time to recovery (s)",
    long_name="Time to recovery (seconds)",
    api_name="time_to_recovery_s",
    description="Average time that it took a team to recover the possession of the ball after losing it in open play per match. ",
    calculation="Average time that it took a team to recover the possession of the ball after losing it in open play per match. ",
    target=TEAM,
    higher_is_better=False,
)

TURNOVER_LINE_HEIGHT = Metric(
    name="Turnover line height (m)",
    long_name="Turnover line height (m)",
    api_name="turnover_line_height_m",
    description="Average height on the pitch that a team lost the possession of the ball in open play per match.",
    calculation="Average height on the pitch that a team lost the possession of the ball in open play per match.",
    target=TEAM,
)

TURNOVERS = Metric(
    name="Turnovers",
    long_name="Turnovers",
    api_name="opp_recoveries",
    description="Number of times that a team lost the possession of the ball in open play per match.",
    calculation="Number of times that a team lost the possession of the ball in open play per match.",
    target=TEAM,
    higher_is_better=False,
)

WIN_PROBABILITY_PCT = Metric(
    name="Win probability %",
    long_name="Win probability %",
    api_name="win_probability_pct",
    description="Probability that a team would win a match based on their and their opponent's expected goals per match.",
    calculation="Proportion of matches won by a team from simulating the results of a game 10000 times based on poisson sampling of the expected goals of the team an their opponents per match.",
    target=TEAM,
    unit="%",
)

NP_GOALS = Metric(
    name="np Goals",
    long_name="Non-penalty goals",
    api_name="np_goals",
    description="Number of non-penalty goals by a team per match.",
    calculation="Number of non-penalty goals by a team per match.",
    target=TEAM,
)

OPP_NP_GOALS = Metric(
    name="Opp. np Goals",
    long_name="Opponent non-penalty goals",
    api_name="opp_np_goals",
    description="Number of non-penalty goals by the opposing team per match.",
    calculation="Number of non-penalty goals by the opposing team per match.",
    target=TEAM,
    higher_is_better=False,
)

NP_SHOTS = Metric(
    name="np Shots",
    long_name="Non-penalty shots",
    api_name="np_shots",
    description="Number of non-penalty shots by a team per match.",
    calculation="Number of non-penalty shots by a team per match.",
    target=TEAM,
)

OPP_NP_SHOTS = Metric(
    name="Opp. np Shots",
    long_name="Opponent non-penalty shots",
    api_name="opp_np_shots",
    description="Number of non-penalty shots by the opposing team per match.",
    calculation="Number of non-penalty shots by the opposing team per match.",
    target=TEAM,
    higher_is_better=False,
)

NP_XG = Metric(
    name="np xG",
    long_name="Non-penalty expected goals",
    api_name="np_xg",
    description="Number of goals that a team should be expected to score statistically in a match considering the quality of their chances, excluding penalties per match.",
    calculation="Number of goals that a team should be expected to score statistically in a match considering the quality of their chances, excluding penalties per match.",
    target=TEAM,
)

OPP_NP_XG = Metric(
    name="Opp. np xG",
    long_name="Opponent non-penalty expected goals",
    api_name="opp_np_xg",
    description="Number of goals that the opposing team should be expected to score statistically in a match considering the quality of their chances, excluding penalties per match.",
    calculation="Number of goals that the opposing team should be expected to score statistically in a match considering the quality of their chances, excluding penalties per match.",
    target=TEAM,
    higher_is_better=False,
)

XG_TEAM = Metric(
    name="xG",
    long_name="Expected goals",
    api_name="xg",
    description="Number of goals that a team should be expected to score statistically considering the quality of their chances per match.",
    calculation="Number of goals that a team should be expected to score statistically considering the quality of their chances per match.",
    target=TEAM,
)

XG_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="xG within 10s after recovery",
    long_name="Expected goals within 10s after recovery",
    api_name="xg_within_10s_after_recovery",
    description="Number of goals that a team should be expected to score statistically considering the quality of their chances within 10 seconds after an open play ball recovery per match.",
    calculation="Number of goals that a team should be expected to score statistically considering the quality of their chances within 10 seconds after an open play ball recovery per match.",
    target=TEAM,
)

TEAM_XG_PER_SHOT = Metric(
    name="np xG per shot",
    long_name="Non-penalty expected goals per shot",
    api_name="np_xg_per_shot",
    description="Average probability that the non-penalty shots of a team resulted in a goal per match.",
    calculation="Average probability that the non-penalty shots of a team resulted in a goal per match.",
    target=TEAM,
)

OPP_TEAM_XG_PER_SHOT = Metric(
    name="Opp. np xG per shot",
    long_name="Opposition non-penalty expected goals per shot",
    api_name="opp_np_xg_per_shot",
    description="Average probability that the non-penalty shots of the opposing team resulted in a goal per match.",
    calculation="Average probability that the non-penalty shots of the opposing team resulted in a goal per match.",
    target=TEAM,
    higher_is_better=False,
)

XPOINTS = Metric(
    name="xPoints",
    long_name="Expected points",
    api_name="xpts",
    description="Number of points that a team should expect to receive statistically from a match based on their and their opponent's expected goals per match.",
    calculation="The probability of winning a match multiplied by three plus the probability of a draw per match.",
    target=TEAM,
)

XT_TEAM = Metric(
    name="xT",
    long_name="Expected threat",
    api_name="xt",
    description="Expected offensive contribution by a team considering the quality of their passess and ball carries per match. ",
    calculation="Expected offensive contribution by a team considering the quality of their passess and ball carries per match. ",
    target=TEAM,
)

XT_WITHIN_10S_AFTER_RECOVERY = Metric(
    name="xT within 10s after recovery",
    long_name="Expected threat within 10s after recovery",
    api_name="xt_within_10s_after_recovery",
    description="Expected offensive contribution by a team considering the quality of their actions within 10 seconds after an open play ball recovery per match.",
    calculation="Expected offensive contribution by a team considering the quality of their actions within 10 seconds after an open play ball recovery per match.",
    target=TEAM,
)

YELLOW_CARDS = Metric(
    name="Yellow cards",
    long_name="Yellow cards",
    api_name="yellow_cards",
    description="Number of yellow cards that a team received per match.",
    calculation="Number of yellow cards that a team received per match.",
    target=TEAM,
)

OPP_YELLOW_CARDS = Metric(
    name="Opp. Yellow cards",
    long_name="Opponent yellow cards",
    api_name="opp_yellow_cards",
    description="The number of yellow cards that the opposing team received per match.",
    calculation="The number of yellow cards that the opposing team received per match.",
    target=TEAM,
)

POINTS_MINUS_XPOINTS = Metric(
    name="Points - xPoints",
    long_name="Points - xPoints",
    api_name="xpts_points_diff",
    description="Difference between the number of points and the expected points of a team per match.",
    calculation="Number of points minus the expected points of a team per match.",
    target=TEAM,
)

POINTS = Metric(
    name="Points",
    long_name="Points",
    api_name="points",
    description="Difference between the number of points and the expected points of a team per match.",
    calculation="Number of points minus the expected points of a team per match.",
    target=TEAM,
)

FINAL_THIRD_THROWINS = Metric(
    name="Final-third throw-ins",
    long_name="Final-third throw-ins",
    api_name="num_throwins_final_third",
    description="Number of throw-ins in the final third received by a team per match.",
    calculation="Number of throw-ins in the final third received by a team per match.",
    target=TEAM,
)

OPP_FINAL_THIRD_THROWINS = Metric(
    name="Opp. Final-third throw-ins",
    long_name="Opponent final-third throw-ins",
    api_name="opp_num_throwins_final_third",
    description="Number of throw-ins in the final third received by the opposing team per match.",
    calculation="Number of throw-ins in the final third received by the opposing team per match.",
    target=TEAM,
)

CORNERS = Metric(
    name="Corners",
    long_name="Corners",
    api_name="corners",
    description="Number of corners received by a team per match.",
    calculation="Number of corners received by a team per match.",
    target=TEAM,
)

OPP_CORNERS = Metric(
    name="Opp. Corners",
    long_name="Opponent corners",
    api_name="opp_corners",
    description="Number of corners received by the opposing team per match.",
    calculation="Number of corners received by the opposing team per match.",
    target=TEAM,
    higher_is_better=False,
)

PENALTIES = Metric(
    name="Penalties",
    long_name="Penalties",
    api_name="penalties",
    description="Number of penalties received by a team per match.",
    calculation="Number of penalties received by a team per match.",
    target=TEAM,
)

OPP_PENALTIES = Metric(
    name="Opp. Penalties",
    long_name="Opponent penalties",
    api_name="opp_penalties",
    description="Number of penalties received by the opposing team per match.",
    calculation="Number of penalties received by the opposing team per match.",
    target=TEAM,
    higher_is_better=False,
)

OWN_GOALS = Metric(
    name="Own goals",
    long_name="Own goals",
    api_name="own_goals",
    description="Number of own goals scored by a team per match.",
    calculation="Number of own goals scored by a team per match.",
    target=TEAM,
)

# Style of play Metrics
BOX_ENTRIES_FROM_CARRIES_PCT = Metric(
    name="Box entries from carries %",
    long_name="Box entries from carries %",
    api_name="box_entries_from_carries_pct",
    description="Percentage of penalty area entries by a team that come through carries per match.",
    calculation="Number of penalty area entries by a team that come through carries divided by the total amount of penalty area entries per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

BOX_ENTRIES_FROM_CROSSES_PCT = Metric(
    name="Box entries from crosses %",
    long_name="Box entries from crosses %",
    api_name="box_entries_from_crosses_pct",
    description="Percentage of penalty area entries by a team that come through crosses per match.",
    calculation="Number of penalty area entries by a team that come through crosses divided by total amount of penalty area entries per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
    unit="%",
)

DRIBBLES_PER_FINAL_THIRD_ENTRY = Metric(
    name="Dribbles per final-third possession",
    long_name="Dribbles per final-third possession",
    api_name="dribbles_per_final_third_possession",
    description="Number of dribbles in the final third by a team divided by the number of possessions that reached the final third by the team per match.",
    calculation=",Number of dribbles in the final third by a team divided by the number of possessions that reached the final third by the team per match.",
    target=TEAM,
    style_of_play_metric=True,
)

CROSSES_PER_FINAL_THIRD_ENTRY = Metric(
    name="Crosses per final-third possession",
    long_name="Crosses per final-third possession",
    api_name="crosses_per_final_third_possession",
    description="Number of crosses in the final third by a team divided by the number of possessions that reached the final third by the team per match.",
    calculation="Number of crosses in the final third by a team divided by the number of possessions that reached the final third by the team per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
)

FORWARD_PASSES_FROM_MIDDLE_THIRD_PCT = Metric(
    name="Forward passes from middle third %",
    long_name="Forward passes from middle third %",
    api_name="forward_passes_from_middle_third_pct",
    description="Percentage of passes in the middle third by a team that are forward per match.",
    calculation="Number of passes in the middle third by a team that are forward divided by total number of passes in the middle third per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
    unit="%",
)

BUILDUP_FROM_GOALKICK_PCT = Metric(
    name="Buildup from goalkick %",
    long_name="Buildup from goalkick %",
    api_name="buildups_from_goalkicks_pct",
    description="Percentage of the possessions that start with goalkicks by a team where less than one third of the passes in the possessions are longer than 32 meters per match.",
    calculation="Percentage of the possessions that start with goalkicks by a team where less than one third of the passes in the possessions are longer than 32 meters per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

BOX_ENTRIES_WITHIN_10S_AFTER_AH_RECOVERY_PCT = Metric(
    name="Possessions to box within 10s after att. half recovery %",
    long_name="Possessions to box within 10s after attacking-half recovery %",
    api_name="box_entry_within_10_after_recovery_att_half_pct",
    description="Percentage of possessions that begin with a ball recovery in the attacking half and result in the team entering the penalty area within 10 seconds per match.",
    calculation="Amount of possessions that begin with a ball recovery in the attacking half and result in the team entering the penalty area within 10 seconds divided by total amount of possessions that begin with a ball recovery in the attacking half per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

FINAL_THIRD_ENTRIES_WITHIN_10S_AFTER_OH_RECOVERY_PCT = Metric(
    name="Possessions to final third within 10s after own-half recovery %",
    long_name="Possessions to final third within 10s after own-half recovery %",
    api_name="final_third_entry_within_10s_after_recovery_own_half_pct",
    description="Percentage of possessions that begin with a ball recovery in their own half and result in the team entering the final third within 10 seconds per match.",
    calculation="Number of possessions that begin with a ball recovery in their own half and result in the team entering the final third within 10 seconds divided by total amount of possessions that begin with a ball recovery in own half per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

FIRST_PASS_FORWARD_PCT = Metric(
    name="First pass forward %",
    long_name="First pass forward %",
    api_name="first_pass_forward_after_recovery_own_half_pct",
    description="Percentage of first passes played forward after a ball recovery by a team per match.",
    calculation="Number of first passes played forward after a ball recovery by a team divided by amount of first passes after a ball recovery per match.",
    target=TEAM,
    unit="%",
    style_of_play_metric=True,
)

TIME_TO_FORWARD_PASS_AFTER_OH_RECOVERY = Metric(
    name="Time to forward pass after own-half recovery (s)",
    long_name="Time to forward pass after own-half recovery (s)",
    api_name="median_time_to_first_forward_pass_own_half_s",
    description="Average time it takes a team to make a forward pass after recovering the ball in their own half per match.",
    calculation="Average time it takes a team to make a forward pass after recovering the ball in their own half per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
)


SHOTS_FROM_OUTSIDE_BOX_PCT = Metric(
    name="Shots from outside the box %",
    long_name="Shots from outside the box %",
    api_name="shots_from_outside_box_pct",
    description="Percentage of shots by a team from outside of the penalty area per match.",
    calculation="Number of shots by a team from outside of the penalty area divided by total number of shots by a team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

SHOTS_PER_FINAL_THIRD_PASS = Metric(
    name="Shots per final-third pass",
    long_name="Shots per final-third pass",
    api_name="shots_per_final_third_pass",
    description="Number of shots by a team divided by their number of passes in the final third per match.",
    calculation="Number of shots by a team divided by their number of passes in the final third per match.",
    target=TEAM,
    style_of_play_metric=True,
)

SHOTS_FROM_DIRECT_ATTACKS_PCT = Metric(
    name="Shots from direct attacks %",
    long_name="Shots from direct attacks %",
    api_name="shots_from_direct_attacks_pct",
    description="Percentage of shots by a team that come from possessions where at least 50% of the total ball movement is forward per match.",
    calculation="Number of shots by a team that come from possessions where at least 50% of the total ball movement is forward divided by total number of shots by a team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

SHOTS_FROM_SUSTAINED_ATTACKS_PCT = Metric(
    name="Shots from sustained attacks %",
    long_name="Shots from sustained attacks %",
    api_name="shots_from_sustained_attacks_pct",
    description="Percentage of shots by a team that come from possessions with 8 or more passes in the attacking half per match.",
    calculation="Number of shots by a team that come from possessions with 8 or more passes in the attacking half divided by total number of shots by a team per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
    unit="%",
)

FOULS_IN_ATTACKING_HALF_PCT = Metric(
    name="Fouls in attacking half %",
    long_name="Fouls in attacking half %",
    api_name="fouls_in_attacking_half_pct",
    description="Percentage of fouls by a team in the attacking half per match.",
    calculation="Number of fouls by a team in the attacking half divided by total number of fouls by a team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

FINAL_THIRD_RECOVERIES_PCT = Metric(
    name="Final-third recoveries %",
    long_name="Final-third recoveries %",
    api_name="final_third_recoveries_pct",
    description="Percentage of recoveries by a team that are in the final third per match.",
    calculation="Number of recoveries by a team that are in the final third divided by total number of recoveries by a team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

OPP_FINAL_THIRD_ENTRIES_WITHIN_10S_AFTER_LOSS_IN_AH_PCT = Metric(
    name="Opp. Possessions to final third within 10s after att. half turnover %",
    long_name="Opp. Possessions to final third within 10s after attacking-half turnover %",
    api_name="opp_final_third_entry_within_10s_after_recovery_own_half_pct",
    description="Percentage of opponent possessions that begin with a loss in open play in the attacking half by a team and result in the opponent entering the final third within 10 seconds per match.",
    calculation="Percentage of opponent possessions that begin with a loss in open play in the attacking half by a team and result in the opponent entering the final third within 10 seconds per match.",
    target=TEAM,
    unit="%",
    style_of_play_metric=True,
)

TIME_TO_DEFENSIVE_ACTION_AFTER_LOSS_IN_AH = Metric(
    name="Time to defensive action after att. half turnover (s)",
    long_name="Time to defensive action after attacking-half turnover (s)",
    api_name="time_to_defensive_action_after_loss_att_half_s",
    description="Average time that it took a team to make a defensive action after losing the possession of the ball in the attacking half in open play per match.",
    calculation="Average time that it took a team to make a defensive action after losing the possession of the ball in the attacking half in open play per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
)

TIME_TO_DEFENSIVE_ACTION_AFTER_LOSS_IN_OH = Metric(
    name="Time to defensive action after own-half turnover (s)",
    long_name="Time to defensive action after own-half turnover (s)",
    api_name="time_to_defensive_action_after_loss_own_half_s",
    description="Average time that it took a team to make a defensive action after losing the possession of the ball in their own half in open play per match.",
    calculation="Average time that it took a team to make a defensive action after losing the possession of the ball in their own half in open play per match.",
    target=TEAM,
    style_of_play_metric=True,
    style_of_play_left=True,
)

OPP_BOX_ENTRIES_FROM_CARRIES_PCT = Metric(
    name="Opp. Box entries from carries %",
    long_name="Opp. Box entries from carries %",
    api_name="opp_box_entries_from_carries_pct",
    description="Percentage of penalty area entries by the opposing team that come through carries per match.",
    calculation="Number of penalty area entries by the opposing team that come through carries divided by the total amount of penalty area entries by the opposing team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

OPP_BOX_ENTRIES_FROM_CROSSES_PCT = Metric(
    name="Opp. Box entries from crosses %",
    long_name="Opp. Box entries from crosses %",
    api_name="opp_box_entries_from_crosses_pct",
    description="Percentage of penalty area entries by the opposing team that come through crosses per match.",
    calculation="Number of penalty area entries by the opposing team that come through crosses divided by the total amount of penalty area entries by the opposing team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

OPP_SHOTS_FROM_OUTSIDE_BOX_PCT = Metric(
    name="Opp. Shots from outside the box %",
    long_name="Opp. Shots from outside the box %",
    api_name="opp_shots_from_outside_box_pct",
    description="Percentage of shots by the opposing team from outside of the penalty area per match.",
    calculation="Number of shots by the opposing team from outside of the penalty area divided by total number of shots by the opposing team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

OPP_SHOTS_PER_FINAL_THIRD_PASS = Metric(
    name="Opp. Shots per final-third pass",
    long_name="Opponent Shots per final-third pass",
    api_name="opp_shots_per_final_third_pass",
    description="Number of shots by the opposing team divided by their number of passes in the final third per match.",
    calculation="Number of shots by the opposing team divided by their number of passes in the final third per match.",
    target=TEAM,
    style_of_play_metric=True,
)

OPP_SHOTS_FROM_DIRECT_ATTACKS_PCT = Metric(
    name="Opp. Shots from direct attacks %",
    long_name="Opp. Shots from direct attacks %",
    api_name="opp_shots_from_direct_attacks_pct",
    description="Percentage of shots by the opposing team that come from possessions where at least 50% of the total ball movement is forward per match.",
    calculation="Number of shots by the opposing team that come from possessions where at least 50% of the total ball movement is forward divided by total number of shots by the opposing team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)

OPP_SHOTS_FROM_SUSTAINED_ATTACKS_PCT = Metric(
    name="Opp. Shots from sustained attacks %",
    long_name="Opp. Shots from sustained attacks %",
    api_name="opp_shots_from_sustained_attacks_pct",
    description="Percentage of shots by the opposing team team that come from possessions with 8 or more passes in the attacking half per match.",
    calculation="Number of shots by the opposing team that come from possessions with 8 or more passes in the attacking half divided by total number of shots by the opposing team per match.",
    target=TEAM,
    style_of_play_metric=True,
    unit="%",
)


ALL_TEAM_METRICS = [
    var for var in globals().values() if isinstance(var, Metric) and var.target == TEAM
]
TEAM_SEASON_METRICS = [
    var
    for var in globals().values()
    if isinstance(var, Metric)
    and var.target == TEAM
    and var.style_of_play_metric == False
]

TEAM_METRIC_MAP = {metric.name: metric for metric in ALL_TEAM_METRICS}
TEAM_SEASON_METRIC_MAP = {metric.name: metric for metric in TEAM_SEASON_METRICS}
TEAM_API_METRIC_MAP = {metric.api_name: metric for metric in ALL_TEAM_METRICS}
TEAM_API_TO_NAME_MAP = {metric.api_name: metric.name for metric in ALL_TEAM_METRICS}
TEAM_API_TO_METRIC_NAME_MAP = {metric: metric.name for metric in ALL_TEAM_METRICS}
