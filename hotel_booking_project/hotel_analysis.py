"""
Project 13: Hotel Booking & Occupancy Analysis
Libraries used: pandas and numpy ONLY (plus 'os' just to build file paths).

How to run:
    python hotel_analysis.py
(Run it from inside the project folder, or from anywhere - the paths below
are built from the location of this file.)

What it produces:
    - Prints every task's result on the screen
    - output/analysis_report.txt           (full printed report)
    - output/room_type_summary.csv         (deliverable 1)
    - output/channel_cancellation_table.csv(deliverable 2)
    - output/monthly_occupancy_trend.csv   (deliverable 3)
    - output/cleaned_hotel_bookings.csv    (cleaned data with new columns)
    - five recommendations (deliverable 4) are at the end of the report
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Setup: file paths and a small helper that prints AND stores the report
# --------------------------------------------------------------------------
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_FOLDER, "data", "13_hotel_bookings.csv")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

report_lines = []          # every printed line is also saved here


def show(text=""):
    """Print text on screen and keep a copy for the report file."""
    text = str(text)
    print(text)
    report_lines.append(text)


def heading(title):
    show()
    show("=" * 80)
    show(title)
    show("=" * 80)


# ==========================================================================
# TASK 1: Load and inspect the booking records
# ==========================================================================
heading("TASK 1: LOAD AND INSPECT THE BOOKING RECORDS")

df = pd.read_csv(DATA_FILE)

show(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
show("\nFirst 5 records:")
show(df.head())
show("\nColumn data types:")
show(df.dtypes)
show("\nStatistical summary of numeric columns:")
show(df.describe().round(2))

# ==========================================================================
# TASK 2: Missing values and duplicate Booking_ID records
# ==========================================================================
heading("TASK 2: MISSING VALUES AND DUPLICATE BOOKING_IDs")

show("Missing values per column:")
show(df.isnull().sum())

dup_id_rows = df[df.duplicated(subset="Booking_ID", keep=False)]
dup_extra_copies = df.duplicated(subset="Booking_ID").sum()
exact_duplicate_rows = df.duplicated().sum()

show(f"\nRows that share a Booking_ID with another row : {len(dup_id_rows)}")
show(f"Extra (repeated) copies of a Booking_ID         : {dup_extra_copies}")
show(f"Of these, fully identical rows (every column)   : {exact_duplicate_rows}")
show("\nDuplicated Booking_ID records (sorted by ID):")
show(dup_id_rows.sort_values("Booking_ID").to_string())

# The repeated rows are exact copies, so they would double-count bookings
# and revenue. We keep the first occurrence and remove the rest.
rows_before = len(df)
df = df.drop_duplicates(subset="Booking_ID", keep="first").reset_index(drop=True)
show(f"\nDuplicates removed. Rows before: {rows_before}, rows after: {len(df)}")

# ==========================================================================
# TASK 3: Handle missing Lead_Time_Days and Room_Rate (justified approach)
# ==========================================================================
heading("TASK 3: HANDLING MISSING Lead_Time_Days AND Room_Rate")

missing_lead = df["Lead_Time_Days"].isnull().sum()
missing_rate = df["Room_Rate"].isnull().sum()
show(f"Missing Lead_Time_Days after removing duplicates: {missing_lead}")
show(f"Missing Room_Rate after removing duplicates     : {missing_rate}")

# Room_Rate  -> fill with the MEDIAN rate of the SAME room type.
#   Why: price depends on the room category (a Suite costs more than a
#   Standard room), and the median is not pulled around by extreme prices.
# Lead_Time  -> fill with the overall MEDIAN lead time.
#   Why: lead time is a count of days that is spread over a wide range, so
#   the median is a safe "typical" value and does not change the shape much.
# We do NOT drop these rows because that would throw away valid bookings
# (nights, channel, cancellation status are still known).
rate_median_by_room = df.groupby("Room_Type")["Room_Rate"].transform("median")
df["Room_Rate"] = df["Room_Rate"].fillna(rate_median_by_room)

lead_median = df["Lead_Time_Days"].median()
df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(lead_median)

show(f"\nMedian lead time used for filling: {lead_median} days")
show("Median Room_Rate by room type (used for filling):")
show(df.groupby("Room_Type")["Room_Rate"].median().round(2))
show("\nMissing values after cleaning:")
show(df.isnull().sum())

# ==========================================================================
# TASK 4: Convert Checkin_Date to datetime, extract arrival month & weekday
# ==========================================================================
heading("TASK 4: DATE CONVERSION - ARRIVAL MONTH AND WEEKDAY")

df["Checkin_Date"] = pd.to_datetime(df["Checkin_Date"])
df["Arrival_Month"] = df["Checkin_Date"].dt.month_name()           # e.g. May
df["Arrival_Month_Num"] = df["Checkin_Date"].dt.month              # 1-12
df["Arrival_Year_Month"] = df["Checkin_Date"].dt.strftime("%Y-%m")  # e.g. 2025-05
df["Arrival_Weekday"] = df["Checkin_Date"].dt.day_name()           # e.g. Friday

show(f"Checkin_Date data type is now: {df['Checkin_Date'].dtype}")
show(f"Date range in data: {df['Checkin_Date'].min().date()} to {df['Checkin_Date'].max().date()}")
show("\nSample of new columns:")
show(df[["Checkin_Date", "Arrival_Month", "Arrival_Year_Month", "Arrival_Weekday"]].head())

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_counts = df["Arrival_Weekday"].value_counts().reindex(weekday_order)
show("\nNumber of arrivals by weekday (useful for front-desk staffing):")
show(weekday_counts)

# ==========================================================================
# TASK 5: Total bookings, total room nights, overall cancellation %
# ==========================================================================
heading("TASK 5: TOTAL BOOKINGS, ROOM NIGHTS, CANCELLATION PERCENTAGE")

df["Is_Canceled"] = np.where(df["Canceled"] == "Yes", 1, 0)

total_bookings = len(df)
total_room_nights = df["Nights"].sum()
canceled_bookings = df["Is_Canceled"].sum()
cancellation_pct = canceled_bookings / total_bookings * 100

realized_df = df[df["Canceled"] == "No"].copy()     # non-canceled bookings only
realized_bookings = len(realized_df)
realized_room_nights = realized_df["Nights"].sum()

show(f"Total bookings (all)                  : {total_bookings}")
show(f"Total room nights booked (all)        : {total_room_nights}")
show(f"Canceled bookings                     : {canceled_bookings}")
show(f"Overall cancellation percentage       : {cancellation_pct:.2f}%")
show(f"Non-canceled (realized) bookings      : {realized_bookings}")
show(f"Room nights actually stayed (realized): {realized_room_nights}")

# ==========================================================================
# TASK 6: Row-wise total booking revenue (Nights * Room_Rate), non-canceled
# ==========================================================================
heading("TASK 6: ROW-WISE BOOKING REVENUE (NON-CANCELED BOOKINGS ONLY)")

# Canceled bookings earn no revenue, so they get 0.
df["Total_Revenue"] = np.where(df["Canceled"] == "No", df["Nights"] * df["Room_Rate"], 0)
realized_df = df[df["Canceled"] == "No"].copy()      # refresh with the revenue column

total_revenue = realized_df["Total_Revenue"].sum()
show(df[["Booking_ID", "Nights", "Room_Rate", "Canceled", "Total_Revenue"]].head(8))
show(f"\nTotal realized revenue: {total_revenue:,.2f}")
show(f"Average revenue per non-canceled booking: {realized_df['Total_Revenue'].mean():,.2f}")

# Revenue lost because of cancellations (what those bookings would have earned)
potential_revenue_lost = (df.loc[df["Canceled"] == "Yes", "Nights"]
                          * df.loc[df["Canceled"] == "Yes", "Room_Rate"]).sum()
show(f"Revenue exposed to cancellations   : {potential_revenue_lost:,.2f}")

# ==========================================================================
# TASK 7: Average room rates and occupancy across room types
# ==========================================================================
heading("TASK 7: ROOM TYPE COMPARISON (RATE, OCCUPANCY, REVENUE)")

# The data has no hotel capacity (number of rooms), so a true occupancy %
# cannot be computed. We use "occupancy" = room nights actually stayed
# (non-canceled) and its percentage share of all stayed room nights.
room_summary = realized_df.groupby("Room_Type").agg(
    Bookings=("Booking_ID", "count"),
    Occupied_Room_Nights=("Nights", "sum"),
    Avg_Room_Rate=("Room_Rate", "mean"),
    Avg_Stay_Nights=("Nights", "mean"),
    Total_Revenue=("Total_Revenue", "sum"),
)
room_summary["Occupancy_Share_%"] = (room_summary["Occupied_Room_Nights"]
                                     / room_summary["Occupied_Room_Nights"].sum() * 100)
room_summary["Revenue_Share_%"] = (room_summary["Total_Revenue"]
                                   / room_summary["Total_Revenue"].sum() * 100)
cancel_by_room = df.groupby("Room_Type")["Is_Canceled"].mean() * 100
room_summary["Cancellation_%"] = cancel_by_room
room_summary = room_summary.round(2).sort_values("Total_Revenue", ascending=False)

show("Room-type revenue summary (non-canceled bookings):")
show(room_summary)
room_summary.to_csv(os.path.join(OUTPUT_FOLDER, "room_type_summary.csv"))

best_room_rate = room_summary["Avg_Room_Rate"].idxmax()
best_room_revenue = room_summary["Total_Revenue"].idxmax()
best_room_occ = room_summary["Occupied_Room_Nights"].idxmax()
show(f"\nHighest average rate : {best_room_rate}")
show(f"Most occupied nights : {best_room_occ}")
show(f"Highest revenue      : {best_room_revenue}")

# Multi-level summary hinted in the assignment: Room_Type x Booking_Channel
show("\nMulti-level summary - revenue by Room_Type and Booking_Channel:")
multi_level = realized_df.groupby(["Room_Type", "Booking_Channel"]).agg(
    Bookings=("Booking_ID", "count"),
    Room_Nights=("Nights", "sum"),
    Avg_Rate=("Room_Rate", "mean"),
    Revenue=("Total_Revenue", "sum"),
).round(2)
show(multi_level)
multi_level.to_csv(os.path.join(OUTPUT_FOLDER, "room_type_by_channel_summary.csv"))

# ==========================================================================
# TASK 8: Rank booking channels by volume and revenue
# ==========================================================================
heading("TASK 8: RANKING BOOKING CHANNELS")

channel_rank = df.groupby("Booking_Channel").agg(Total_Bookings=("Booking_ID", "count"))
channel_rank["Realized_Bookings"] = realized_df.groupby("Booking_Channel")["Booking_ID"].count()
channel_rank["Total_Revenue"] = realized_df.groupby("Booking_Channel")["Total_Revenue"].sum()
channel_rank["Volume_Rank"] = channel_rank["Total_Bookings"].rank(ascending=False, method="min").astype(int)
channel_rank["Revenue_Rank"] = channel_rank["Total_Revenue"].rank(ascending=False, method="min").astype(int)
channel_rank["Revenue_per_Booking"] = channel_rank["Total_Revenue"] / channel_rank["Realized_Bookings"]
channel_rank = channel_rank.round(2).sort_values("Total_Revenue", ascending=False)

show("Channels ranked (1 = best):")
show(channel_rank)
channel_rank.to_csv(os.path.join(OUTPUT_FOLDER, "channel_ranking.csv"))

top_volume_channel = channel_rank["Total_Bookings"].idxmax()
top_revenue_channel = channel_rank["Total_Revenue"].idxmax()
show(f"\nHighest volume channel : {top_volume_channel}")
show(f"Highest revenue channel: {top_revenue_channel}")

# ==========================================================================
# TASK 14 (done early because Task 9 needs it): Lead-time categories, pd.cut()
# ==========================================================================
heading("TASK 14: LEAD-TIME CATEGORIES USING pd.cut()")

# Bands (days between booking and arrival):
#   0-7      Last-Minute
#   8-30     Short-Term
#   31-90    Medium-Term
#   91+      Advance
lead_bins = [-1, 7, 30, 90, np.inf]
lead_labels = ["Last-Minute (0-7)", "Short-Term (8-30)", "Medium-Term (31-90)", "Advance (91+)"]
df["Lead_Time_Category"] = pd.cut(df["Lead_Time_Days"], bins=lead_bins, labels=lead_labels)
realized_df = df[df["Canceled"] == "No"].copy()      # refresh so it has the new column

lead_summary = df.groupby("Lead_Time_Category", observed=True).agg(
    Bookings=("Booking_ID", "count"),
    Avg_Lead_Days=("Lead_Time_Days", "mean"),
)
lead_summary["Share_%"] = lead_summary["Bookings"] / lead_summary["Bookings"].sum() * 100
show(lead_summary.round(2))

# ==========================================================================
# TASK 9: Cancellation rate by booking channel and by lead-time band
# ==========================================================================
heading("TASK 9: CANCELLATION RATES BY CHANNEL AND LEAD-TIME BAND")

# Mean of the 0/1 column Is_Canceled * 100 = cancellation percentage
channel_cancel = df.groupby("Booking_Channel").agg(
    Total_Bookings=("Booking_ID", "count"),
    Canceled=("Is_Canceled", "sum"),
)
channel_cancel["Cancellation_%"] = channel_cancel["Canceled"] / channel_cancel["Total_Bookings"] * 100
channel_cancel = channel_cancel.round(2).sort_values("Cancellation_%", ascending=False)
show("Channel cancellation table:")
show(channel_cancel)
channel_cancel.to_csv(os.path.join(OUTPUT_FOLDER, "channel_cancellation_table.csv"))

lead_cancel = df.groupby("Lead_Time_Category", observed=True).agg(
    Total_Bookings=("Booking_ID", "count"),
    Canceled=("Is_Canceled", "sum"),
)
lead_cancel["Cancellation_%"] = lead_cancel["Canceled"] / lead_cancel["Total_Bookings"] * 100
lead_cancel = lead_cancel.round(2)
show("\nCancellation by lead-time band:")
show(lead_cancel)
lead_cancel.to_csv(os.path.join(OUTPUT_FOLDER, "lead_time_cancellation_table.csv"))

# Channel x lead-time band cross table (cancellation %)
cross_cancel = (df.groupby(["Booking_Channel", "Lead_Time_Category"], observed=True)["Is_Canceled"]
                .mean() * 100).round(2).unstack()
show("\nCancellation % - Channel (rows) vs Lead-time band (columns):")
show(cross_cancel)
cross_cancel.to_csv(os.path.join(OUTPUT_FOLDER, "channel_by_leadtime_cancellation.csv"))

worst_cancel_channel = channel_cancel["Cancellation_%"].idxmax()
worst_cancel_band = lead_cancel["Cancellation_%"].idxmax()
best_cancel_channel = channel_cancel["Cancellation_%"].idxmin()
best_cancel_band = lead_cancel["Cancellation_%"].idxmin()
show(f"\nHighest cancellation channel : {worst_cancel_channel} ({channel_cancel['Cancellation_%'].max()}%)")
show(f"Lowest cancellation channel  : {best_cancel_channel} ({channel_cancel['Cancellation_%'].min()}%)")
show(f"Highest cancellation band    : {worst_cancel_band} ({lead_cancel['Cancellation_%'].max()}%)")
show(f"Lowest cancellation band     : {best_cancel_band} ({lead_cancel['Cancellation_%'].min()}%)")

# ==========================================================================
# TASK 10: Top 3 peak revenue months
# ==========================================================================
heading("TASK 10: TOP 3 PEAK REVENUE MONTHS")

# The data covers about 22 months (2025-2026), so we look at it two ways.
monthly_revenue = realized_df.groupby("Arrival_Year_Month")["Total_Revenue"].sum()
top3_year_month = monthly_revenue.sort_values(ascending=False).head(3)
show("Top 3 revenue months (specific year-month):")
show(top3_year_month.round(2))

calendar_revenue = (realized_df.groupby(["Arrival_Month_Num", "Arrival_Month"])["Total_Revenue"]
                    .sum().reset_index())
calendar_revenue = calendar_revenue.sort_values("Total_Revenue", ascending=False)
top3_calendar = calendar_revenue.head(3)
show("\nTop 3 calendar months (all years combined - shows seasonality):")
show(top3_calendar[["Arrival_Month", "Total_Revenue"]].round(2).to_string(index=False))
show("\nNote: Jan-Oct appear twice in the data (2025 and 2026) while Nov-Dec appear once,")
show("so combined-year totals favour Jan-Oct. The year-month view above is the fairer one.")

# ==========================================================================
# TASK 11: Average stay duration, leisure vs business travelers
# ==========================================================================
heading("TASK 11: AVERAGE STAY - LEISURE VS BUSINESS")

traveler = realized_df.groupby("Traveler_Type").agg(
    Bookings=("Booking_ID", "count"),
    Avg_Stay_Nights=("Nights", "mean"),
    Median_Stay_Nights=("Nights", "median"),
    Avg_Room_Rate=("Room_Rate", "mean"),
    Total_Revenue=("Total_Revenue", "sum"),
).round(2)
show(traveler)
traveler.to_csv(os.path.join(OUTPUT_FOLDER, "traveler_type_summary.csv"))

longer_stay_type = traveler["Avg_Stay_Nights"].idxmax()
stay_diff = traveler["Avg_Stay_Nights"].max() - traveler["Avg_Stay_Nights"].min()
show(f"\n{longer_stay_type} travelers stay longer on average (difference: {stay_diff:.2f} nights).")

# ==========================================================================
# TASK 12: NumPy - mean, median and standard deviation of daily room rates
# ==========================================================================
heading("TASK 12: NUMPY STATISTICS OF DAILY ROOM RATES")

rates = realized_df["Room_Rate"].to_numpy()        # convert to a NumPy array
rate_mean = np.mean(rates)
rate_median = np.median(rates)
rate_std = np.std(rates)                           # population std (NumPy default)

show(f"Mean daily room rate   : {rate_mean:,.2f}")
show(f"Median daily room rate : {rate_median:,.2f}")
show(f"Standard deviation     : {rate_std:,.2f}")
show(f"Coefficient of variation: {rate_std / rate_mean * 100:.1f}%  (std / mean)")

show("\nSame statistics for each room type:")
for room in ["Standard", "Deluxe", "Suite"]:
    room_rates = realized_df.loc[realized_df["Room_Type"] == room, "Room_Rate"].to_numpy()
    show(f"  {room:9s} mean={np.mean(room_rates):10,.2f}  "
         f"median={np.median(room_rates):10,.2f}  std={np.std(room_rates):10,.2f}")

# ==========================================================================
# TASK 13: Unusually high booking values using the IQR method
# ==========================================================================
heading("TASK 13: OUTLIERS IN BOOKING REVENUE (IQR METHOD)")

revenue_values = realized_df["Total_Revenue"].to_numpy()
q1 = np.percentile(revenue_values, 25)
q3 = np.percentile(revenue_values, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

show(f"Q1 (25th percentile): {q1:,.2f}")
show(f"Q3 (75th percentile): {q3:,.2f}")
show(f"IQR (Q3 - Q1)       : {iqr:,.2f}")
show(f"Lower fence         : {lower_fence:,.2f}")
show(f"Upper fence         : {upper_fence:,.2f}")

high_outliers = realized_df[realized_df["Total_Revenue"] > upper_fence]
low_outliers = realized_df[realized_df["Total_Revenue"] < lower_fence]
show(f"\nUnusually HIGH booking values: {len(high_outliers)}")
show(f"Unusually LOW booking values : {len(low_outliers)}")

if len(high_outliers) > 0:
    show("\nHigh-value bookings:")
    show(high_outliers.sort_values("Total_Revenue", ascending=False)[
        ["Booking_ID", "Room_Type", "Booking_Channel", "Nights", "Room_Rate", "Total_Revenue"]
    ].head(15).round(2).to_string(index=False))
    high_outliers.to_csv(os.path.join(OUTPUT_FOLDER, "high_value_bookings.csv"), index=False)
else:
    show("No booking is above the upper fence, so there are no unusually high bookings.")
    show("(Reason: Nights is limited to 1-7 and rates are limited to a narrow range.)")
    show("\nFor reference, the 5 highest booking values are:")
    show(realized_df.sort_values("Total_Revenue", ascending=False)[
        ["Booking_ID", "Room_Type", "Booking_Channel", "Nights", "Room_Rate", "Total_Revenue"]
    ].head(5).round(2).to_string(index=False))
    show(f"\n95th percentile of booking value (top 5% threshold): {np.percentile(revenue_values, 95):,.2f}")

# ==========================================================================
# EXTRA ANALYSIS 1: Monthly occupancy trend (required deliverable)
# ==========================================================================
heading("EXTRA / DELIVERABLE: MONTHLY OCCUPANCY TREND")

monthly = realized_df.groupby("Arrival_Year_Month").agg(
    Bookings=("Booking_ID", "count"),
    Occupied_Room_Nights=("Nights", "sum"),
    Revenue=("Total_Revenue", "sum"),
)
monthly["Avg_Rate"] = monthly["Revenue"] / monthly["Occupied_Room_Nights"]
monthly["Occupancy_Share_%"] = monthly["Occupied_Room_Nights"] / monthly["Occupied_Room_Nights"].sum() * 100
monthly["3M_Moving_Avg_Nights"] = monthly["Occupied_Room_Nights"].rolling(3).mean()
monthly = monthly.round(2)
show(monthly)
monthly.to_csv(os.path.join(OUTPUT_FOLDER, "monthly_occupancy_trend.csv"))

# A simple text bar chart (no plotting library needed): 1 '#' = 5 room nights
show("\nText chart of occupied room nights per month (1 '#' = 5 nights):")
for month_label, nights in monthly["Occupied_Room_Nights"].items():
    show(f"  {month_label} | {'#' * int(nights // 5)} {int(nights)}")

busiest_month = monthly["Occupied_Room_Nights"].idxmax()
quietest_month = monthly["Occupied_Room_Nights"].idxmin()
show(f"\nBusiest month (room nights) : {busiest_month} ({int(monthly['Occupied_Room_Nights'].max())} nights)")
show(f"Quietest month (room nights): {quietest_month} ({int(monthly['Occupied_Room_Nights'].min())} nights)")

# ==========================================================================
# EXTRA ANALYSIS 2: Weekday arrivals and traveler mix (front-desk staffing)
# ==========================================================================
heading("EXTRA: ARRIVALS BY WEEKDAY (FRONT-DESK STAFFING)")

weekday_summary = realized_df.groupby("Arrival_Weekday").agg(
    Arrivals=("Booking_ID", "count"),
    Room_Nights=("Nights", "sum"),
    Revenue=("Total_Revenue", "sum"),
).reindex(weekday_order).round(2)
weekday_summary["Arrivals_Share_%"] = (weekday_summary["Arrivals"]
                                       / weekday_summary["Arrivals"].sum() * 100).round(2)
show(weekday_summary)
weekday_summary.to_csv(os.path.join(OUTPUT_FOLDER, "weekday_arrivals.csv"))

busiest_day = weekday_summary["Arrivals"].idxmax()
quietest_day = weekday_summary["Arrivals"].idxmin()
show(f"\nBusiest arrival day : {busiest_day} ({int(weekday_summary['Arrivals'].max())} arrivals)")
show(f"Quietest arrival day: {quietest_day} ({int(weekday_summary['Arrivals'].min())} arrivals)")

# ==========================================================================
# TASK 15: Five revenue and operations recommendations
# ==========================================================================
heading("TASK 15: FIVE REVENUE AND OPERATIONS RECOMMENDATIONS")

channel_cancel_rates = channel_cancel["Cancellation_%"]
top_room_share = room_summary.loc[best_room_revenue, "Revenue_Share_%"]

show(f"1. ROOM PRICING: {best_room_rate} rooms earn the highest average rate "
     f"({room_summary.loc[best_room_rate, 'Avg_Room_Rate']:,.0f}), while {best_room_revenue} "
     f"rooms bring the most revenue ({top_room_share:.1f}% of total). Protect the {best_room_revenue} "
     f"rate and use dynamic pricing: raise rates in the peak months "
     f"({', '.join(top3_year_month.index)}) and offer upgrade deals in quiet months "
     f"(lowest month: {quietest_month}).")
show("")
show(f"2. CANCELLATION CONTROL: Overall cancellation is {cancellation_pct:.1f}%, with "
     f"{worst_cancel_channel} the highest ({channel_cancel_rates.max():.1f}%) and "
     f"{best_cancel_channel} the lowest ({channel_cancel_rates.min():.1f}%). Introduce "
     f"stricter cancellation windows or a small deposit for {worst_cancel_channel} bookings, "
     f"and especially for the '{worst_cancel_band}' band ({lead_cancel['Cancellation_%'].max():.1f}% cancelled).")
show("")
lowest_volume_channel = channel_rank["Total_Bookings"].idxmin()
show(f"3. CHANNEL STRATEGY: {top_revenue_channel} brings the most revenue, {top_volume_channel} "
     f"has the most bookings, and {lowest_volume_channel} has the fewest. Compare the commission "
     f"cost of {top_revenue_channel} with its revenue, and run member-rate or loyalty offers to "
     f"grow {lowest_volume_channel} bookings, where the hotel keeps the full room rate.")
show("")
show(f"4. FRONT-DESK STAFFING: Arrivals peak on {busiest_day} ({int(weekday_summary['Arrivals'].max())} "
     f"arrivals) and are lowest on {quietest_day} ({int(weekday_summary['Arrivals'].min())}). "
     f"Schedule extra check-in staff on {busiest_day} and the busiest months "
     f"({busiest_month}), and use lighter rosters on {quietest_day}.")
show("")
show(f"5. STAY-LENGTH AND OVERBOOKING: Business ({traveler.loc['Business', 'Avg_Stay_Nights']:.2f} nights) and "
     f"Leisure ({traveler.loc['Leisure', 'Avg_Stay_Nights']:.2f} nights) stays are almost the same length, "
     f"so use one set of multi-night offers (e.g. stay 5 pay 4) for both instead of separate packages. "
     f"Because about {cancellation_pct:.0f}% of bookings cancel, allow a small, controlled overbooking "
     f"buffer (start at 5%, review monthly) on the high-cancellation channel ({worst_cancel_channel}) "
     f"to reduce empty rooms.")

# ==========================================================================
# Save cleaned data and the full report
# ==========================================================================
cleaned = df.copy()
cleaned["Checkin_Date"] = cleaned["Checkin_Date"].dt.strftime("%Y-%m-%d")
cleaned.to_csv(os.path.join(OUTPUT_FOLDER, "cleaned_hotel_bookings.csv"), index=False)

with open(os.path.join(OUTPUT_FOLDER, "analysis_report.txt"), "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(report_lines))

show("\nAll done. Result files are saved in the 'output' folder.")
